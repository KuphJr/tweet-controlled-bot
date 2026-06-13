"""Read-only TwitterApi.io ingestion: quote-tweets AND comments/replies.

Polls two endpoints for the same source tweet and merges them into one stream of
:class:`RawItem`. Confirmed against the TwitterApi.io docs:

  * Quotes:  ``GET /twitter/tweet/quotes?tweetId=<id>&cursor=...`` -> items under
    ``tweets``; supports ``sinceTime``/``untilTime`` (unix seconds).
  * Replies: ``GET /twitter/tweet/replies?tweetId=<id>&cursor=...`` -> items under
    ``replies`` (we also accept ``tweets`` defensively). We additionally scan
    ``/twitter/tweet/replies/v2`` as a supplemental source because it returns
    some low-trust / low-follower replies that the standard endpoint omits, and
    it includes sub-replies in the source conversation.
  * Both: ``has_next_page`` / ``next_cursor`` for pagination, header ``X-API-Key``,
    ordered newest-first, ~20/page. Tweet fields: ``id``, ``text``, ``createdAt``
    ("Tue Dec 10 07:00:30 +0000 2024"), ``inReplyToId``, ``author.userName`` /
    ``author.name``.

We pass ``sinceTime`` server-side to reduce paging, AND always filter client-side
by ``createdAt`` as a safety net (the docs warn ``has_next_page`` can be
unreliable). No-backfill: at startup all currently-visible item IDs are marked
seen/ignored so only items discovered after startup are processed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import requests

from config import AppConfig, CommandSource, normalize_handle

logger = logging.getLogger("tweet_robot.twitter_reader")

_BASE_URL = "https://api.twitterapi.io"
_QUOTES_PATH = "/twitter/tweet/quotes"
_REPLIES_PATH = "/twitter/tweet/replies"
_REPLIES_V2_PATH = "/twitter/tweet/replies/v2"
_TWITTER_DATE_FMT = "%a %b %d %H:%M:%S %z %Y"

# Safety caps so a single poll can never loop forever (has_next_page can lie).
_MAX_PAGES_PER_POLL = 8
_MAX_PAGES_STARTUP = 15
_REQUEST_TIMEOUT_S = 15


@dataclass
class RawItem:
    tweet_id: str
    source: CommandSource
    author_handle: str
    author_name: str
    text: str
    created_at: float  # epoch seconds
    in_reply_to_tweet_id: str | None = None
    conversation_id: str | None = None


def _parse_created_at(value: str | None) -> float:
    if not value:
        return time.time()  # fail-open: treat unknown timestamps as "now"
    try:
        return datetime.strptime(value, _TWITTER_DATE_FMT).timestamp()
    except (ValueError, TypeError):
        logger.debug("Could not parse createdAt=%r; treating as now.", value)
        return time.time()


class TwitterReader:
    def __init__(self, cfg: AppConfig, source_tweet_id: str, start_time: float):
        self.cfg = cfg
        self.source_tweet_id = str(source_tweet_id)
        self.start_time = start_time
        self._headers = {"X-API-Key": cfg.twitterapi_io_api_key}
        self._last_replies_v2_poll_s = 0.0

    # ------------------------------------------------------------------ #
    # HTTP                                                               #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: dict) -> dict | None:
        try:
            resp = requests.get(
                _BASE_URL + path,
                headers=self._headers,
                params=params,
                timeout=_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("TwitterApi.io request to %s failed: %s", path, exc)
            return None
        except ValueError as exc:  # JSON decode
            logger.warning("TwitterApi.io returned non-JSON from %s: %s", path, exc)
            return None

    @staticmethod
    def _extract_list(payload: dict) -> list[dict]:
        items = payload.get("tweets")
        if items is None:
            items = payload.get("replies")
        return items or []

    @staticmethod
    def _extract_parent_id(raw: dict) -> str | None:
        """Return parent tweet ID for replies, handling known provider spellings."""
        for key in ("inReplyToId", "inReplyToTweetId", "inReplyToStatusId", "in_reply_to_status_id"):
            value = raw.get(key)
            if value:
                return str(value)
        return None

    def _to_item(self, raw: dict, source: CommandSource) -> RawItem | None:
        tweet_id = raw.get("id")
        if not tweet_id:
            return None
        author = raw.get("author") or {}
        return RawItem(
            tweet_id=str(tweet_id),
            source=source,
            author_handle=str(author.get("userName", "") or ""),
            author_name=str(author.get("name", "") or ""),
            text=str(raw.get("text", "") or ""),
            created_at=_parse_created_at(raw.get("createdAt")),
            in_reply_to_tweet_id=self._extract_parent_id(raw),
            conversation_id=str(raw.get("conversationId") or "") or None,
        )

    # ------------------------------------------------------------------ #
    # Paging                                                             #
    # ------------------------------------------------------------------ #

    def _page_source(
        self,
        path: str,
        source: CommandSource,
        *,
        seen_ids: set[str],
        stop_on_seen: bool,
        use_since_time: bool,
        max_pages: int,
    ) -> list[RawItem]:
        """Page through one source newest-first, collecting fresh items.

        Stops when it reaches an already-seen item or one older than
        ``start_time`` (everything beyond is older still), runs out of cursor, or
        hits ``max_pages``.
        """
        collected: list[RawItem] = []
        cursor = ""
        for _ in range(max_pages):
            params: dict[str, object] = {"tweetId": self.source_tweet_id}
            if cursor:
                params["cursor"] = cursor
            if use_since_time:
                params["sinceTime"] = int(self.start_time)

            payload = self._get(path, params)
            if payload is None:
                break

            rows = self._extract_list(payload)
            if not rows:
                break

            reached_old_or_seen = False
            for raw in rows:
                item = self._to_item(raw, source)
                if item is None:
                    continue
                if stop_on_seen and item.tweet_id in seen_ids:
                    reached_old_or_seen = True
                    continue
                if item.created_at < self.start_time:
                    reached_old_or_seen = True
                    continue
                if (
                    source == CommandSource.REPLY
                    and item.in_reply_to_tweet_id
                    and item.in_reply_to_tweet_id != self.source_tweet_id
                ):
                    # Only direct comments on the source tweet are commands. Nested
                    # replies include generated acknowledgements/status replies and
                    # can otherwise create loops in single-account mode.
                    continue
                collected.append(item)

            if reached_old_or_seen:
                break
            if not payload.get("has_next_page"):
                break
            cursor = payload.get("next_cursor") or ""
            if not cursor:
                break

        return collected

    def _page_replies_v2(self, *, seen_ids: set[str], max_pages: int) -> list[RawItem]:
        """Scan replies/v2 as a supplemental source for source-conversation replies.

        The v2 endpoint can include the source tweet and nested replies, and its
        ordering is tree-like rather than strictly newest-first. So do not stop
        early on seen/old rows; scan capped pages and filter to replies inside
        the source conversation.
        """
        collected: list[RawItem] = []
        cursor = ""
        for _ in range(max_pages):
            params: dict[str, object] = {"tweetId": self.source_tweet_id}
            if cursor:
                params["cursor"] = cursor

            payload = self._get(_REPLIES_V2_PATH, params)
            if payload is None:
                break

            rows = self._extract_list(payload)
            if not rows:
                break

            for raw in rows:
                item = self._to_item(raw, CommandSource.REPLY)
                if item is None:
                    continue
                if item.tweet_id == self.source_tweet_id:
                    continue
                if item.tweet_id in seen_ids:
                    continue
                if item.created_at < self.start_time:
                    continue
                if not item.in_reply_to_tweet_id:
                    continue
                if (
                    item.in_reply_to_tweet_id != self.source_tweet_id
                    and item.conversation_id != self.source_tweet_id
                ):
                    continue
                collected.append(item)

            if not payload.get("has_next_page"):
                break
            cursor = payload.get("next_cursor") or ""
            if not cursor:
                break

        return collected

    def _should_poll_replies_v2(self) -> bool:
        """Return True when the supplemental, expensive replies/v2 scan is due."""
        interval_s = max(0.0, float(self.cfg.replies_v2_poll_interval_s))
        max_pages = int(self.cfg.replies_v2_max_pages_per_poll)
        if interval_s <= 0.0 or max_pages <= 0:
            return False

        now = time.time()
        if self._last_replies_v2_poll_s == 0.0 or now - self._last_replies_v2_poll_s >= interval_s:
            self._last_replies_v2_poll_s = now
            return True
        return False

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fetch_new(self, seen_ids: set[str]) -> list[RawItem]:
        """Return new items from BOTH sources, deduped, oldest-first.

        Excludes already-seen IDs, the posting account's own posts in
        separate-account mode, and anything older than ``start_time``. Standard
        replies only contribute direct replies; replies/v2 supplements direct
        replies and sub-replies in the source conversation. The caller is
        responsible for marking returned IDs seen.
        """
        merged: dict[str, RawItem] = {}
        sources: tuple[tuple[str, CommandSource], ...] = (
            (_QUOTES_PATH, CommandSource.QUOTE),
            (_REPLIES_PATH, CommandSource.REPLY),
        )
        for path, source in sources:
            items = self._page_source(
                path,
                source,
                seen_ids=seen_ids,
                stop_on_seen=True,
                use_since_time=True,
                max_pages=_MAX_PAGES_PER_POLL,
            )
            for item in items:
                if item.tweet_id in seen_ids or item.tweet_id in merged:
                    continue
                single_account = self.cfg.bot_handle_norm == self.cfg.admin_handle_norm
                if normalize_handle(item.author_handle) == self.cfg.bot_handle_norm and not single_account:
                    continue  # never process the separate posting account's own posts
                merged[item.tweet_id] = item

        if self._should_poll_replies_v2():
            max_pages = max(1, int(self.cfg.replies_v2_max_pages_per_poll))
            for item in self._page_replies_v2(seen_ids=seen_ids, max_pages=max_pages):
                if item.tweet_id in seen_ids or item.tweet_id in merged:
                    continue
                single_account = self.cfg.bot_handle_norm == self.cfg.admin_handle_norm
                if normalize_handle(item.author_handle) == self.cfg.bot_handle_norm and not single_account:
                    continue
                merged[item.tweet_id] = item

        return sorted(merged.values(), key=lambda it: it.created_at)

    def fetch_current_ids(self) -> set[str]:
        """Collect all currently-visible item IDs (for no-backfill startup).

        Pages both sources without the start-time filter so existing quotes and
        replies can be marked seen/ignored before live processing begins.
        """
        ids: set[str] = set()
        for path in (_QUOTES_PATH, _REPLIES_PATH, _REPLIES_V2_PATH):
            cursor = ""
            for _ in range(_MAX_PAGES_STARTUP):
                params: dict[str, object] = {"tweetId": self.source_tweet_id}
                if cursor:
                    params["cursor"] = cursor
                payload = self._get(path, params)
                if payload is None:
                    break
                rows = self._extract_list(payload)
                if not rows:
                    break
                for raw in rows:
                    tid = raw.get("id")
                    if tid and str(tid) != self.source_tweet_id:
                        ids.add(str(tid))
                if not payload.get("has_next_page"):
                    break
                cursor = payload.get("next_cursor") or ""
                if not cursor:
                    break
        logger.info("Marked %d existing quote/reply IDs as seen (no-backfill).", len(ids))
        return ids
