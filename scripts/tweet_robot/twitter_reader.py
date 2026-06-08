"""Read-only TwitterApi.io ingestion: quote-tweets AND comments/replies.

Polls two endpoints for the same source tweet and merges them into one stream of
:class:`RawItem`. Confirmed against the TwitterApi.io docs:

  * Quotes:  ``GET /twitter/tweet/quotes?tweetId=<id>&cursor=...`` -> items under
    ``tweets``; supports ``sinceTime``/``untilTime`` (unix seconds).
  * Replies: ``GET /twitter/tweet/replies?tweetId=<id>&cursor=...`` -> items under
    ``replies`` (we also accept ``tweets`` defensively).
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

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fetch_new(self, seen_ids: set[str]) -> list[RawItem]:
        """Return new items from BOTH sources, deduped, oldest-first.

        Excludes already-seen IDs, the posting account's own posts in
        separate-account mode, nested replies, and anything older than
        ``start_time``. The caller is responsible for marking returned IDs seen.
        """
        merged: dict[str, RawItem] = {}
        for path, source in ((_QUOTES_PATH, CommandSource.QUOTE), (_REPLIES_PATH, CommandSource.REPLY)):
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

        return sorted(merged.values(), key=lambda it: it.created_at)

    def fetch_current_ids(self) -> set[str]:
        """Collect all currently-visible item IDs (for no-backfill startup).

        Pages both sources without the start-time filter so existing quotes and
        replies can be marked seen/ignored before live processing begins.
        """
        ids: set[str] = set()
        for path, source in ((_QUOTES_PATH, CommandSource.QUOTE), (_REPLIES_PATH, CommandSource.REPLY)):
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
                    if tid:
                        ids.add(str(tid))
                if not payload.get("has_next_page"):
                    break
                cursor = payload.get("next_cursor") or ""
                if not cursor:
                    break
        logger.info("Marked %d existing quote/reply IDs as seen (no-backfill).", len(ids))
        return ids
