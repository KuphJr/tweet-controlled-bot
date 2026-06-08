"""Write-side abstraction for posting replies via the official X API (Tweepy).

Posts as the configured account (single-account default: ``@KuphDev``) via
OAuth 1.0a user context. Replies always target the command's own tweet
(quote-tweet or comment). Posting failures are caught and logged — they never
crash the controller.

Two-tier rate limiting (timestamps persisted via the state store):
  * normal replies (ack / invalid / queue-full / duplicate / admin-status):
    capped at ``MAX_REPLIES_PER_HOUR``.
  * ALL posts, including critical ``@KuphDev`` error notifications: capped at
    ``ABSOLUTE_MAX_POSTS_PER_HOUR``.
Over the normal cap -> keep running, skip normal replies. Over the absolute cap
-> post nothing, log intended text.

``--dry-run`` (or missing X credentials) selects :class:`NoopXPostWriter`, which
logs intended text only.
"""

from __future__ import annotations

import logging
import time

from config import AppConfig

logger = logging.getLogger("tweet_robot.x_post_writer")

_ONE_HOUR_S = 3600.0
_MAX_TWEET_CHARS = 275


class XPostWriter:
    """Base writer: owns rate-limit accounting; subclasses implement ``_send``."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._normal_times: list[float] = []
        self._all_times: list[float] = []

    # -- rate-limit bookkeeping -------------------------------------------- #

    def _prune(self) -> None:
        cutoff = time.time() - _ONE_HOUR_S
        self._normal_times = [t for t in self._normal_times if t >= cutoff]
        self._all_times = [t for t in self._all_times if t >= cutoff]

    def _can_post(self, critical: bool) -> bool:
        if len(self._all_times) >= self.cfg.absolute_max_posts_per_hour:
            return False
        if not critical and len(self._normal_times) >= self.cfg.max_replies_per_hour:
            return False
        return True

    def load_timestamps(self, normal_times: list[float], all_times: list[float]) -> None:
        self._normal_times = list(normal_times or [])
        self._all_times = list(all_times or [])

    def export_timestamps(self) -> tuple[list[float], list[float]]:
        self._prune()
        return list(self._normal_times), list(self._all_times)

    # -- public API -------------------------------------------------------- #

    def send(
        self,
        text: str,
        in_reply_to_tweet_id: str | None = None,
        *,
        critical: bool = False,
    ) -> bool:
        """Post a reply (or standalone tweet). Returns True iff actually posted.

        ``critical=True`` marks error notifications, which bypass the normal-reply
        cap but still respect the absolute cap.
        """
        text = (text or "").strip()
        if not text:
            return False
        self._prune()
        if not self._can_post(critical):
            logger.warning(
                "Rate limit reached (critical=%s, normal=%d/%d, all=%d/%d); NOT posting: %s",
                critical,
                len(self._normal_times),
                self.cfg.max_replies_per_hour,
                len(self._all_times),
                self.cfg.absolute_max_posts_per_hour,
                text,
            )
            return False

        ok = self._send(text[:_MAX_TWEET_CHARS], in_reply_to_tweet_id)
        if ok:
            now = time.time()
            self._all_times.append(now)
            if not critical:
                self._normal_times.append(now)
        return ok

    def _send(self, text: str, in_reply_to_tweet_id: str | None) -> bool:
        raise NotImplementedError


class TweepyXPostWriter(XPostWriter):
    def __init__(self, cfg: AppConfig, client=None):
        super().__init__(cfg)
        if client is not None:
            self._client = client
        else:
            import tweepy

            self._client = tweepy.Client(
                consumer_key=cfg.x_api_key,
                consumer_secret=cfg.x_api_secret,
                access_token=cfg.x_access_token,
                access_token_secret=cfg.x_access_token_secret,
            )

    def _send(self, text: str, in_reply_to_tweet_id: str | None) -> bool:
        try:
            if in_reply_to_tweet_id:
                self._client.create_tweet(
                    text=text,
                    in_reply_to_tweet_id=str(in_reply_to_tweet_id),
                    user_auth=True,
                )
            else:
                self._client.create_tweet(text=text, user_auth=True)
            logger.info("Posted to X (reply_to=%s): %s", in_reply_to_tweet_id, text)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("X post failed (continuing): %s", exc)
            return False


class NoopXPostWriter(XPostWriter):
    """Logs intended text only; consumes no rate-limit budget (dry-run / no creds)."""

    def send(
        self,
        text: str,
        in_reply_to_tweet_id: str | None = None,
        *,
        critical: bool = False,
    ) -> bool:
        logger.info(
            "[log-only] Would post (reply_to=%s, critical=%s): %s",
            in_reply_to_tweet_id,
            critical,
            (text or "").strip(),
        )
        return False

    def _send(self, text: str, in_reply_to_tweet_id: str | None) -> bool:  # pragma: no cover
        return False


def build_x_post_writer(cfg: AppConfig) -> XPostWriter:
    """Factory: real Tweepy writer when configured (and not dry-run), else no-op.

    Tip: to run on real hardware WITHOUT posting, leave the ``X_*`` credentials
    blank — this returns the log-only writer.
    """
    if cfg.dry_run:
        logger.info("--dry-run set; using NoopXPostWriter (log-only, no posting).")
        return NoopXPostWriter(cfg)

    creds = (cfg.x_api_key, cfg.x_api_secret, cfg.x_access_token, cfg.x_access_token_secret)
    if not all(creds):
        logger.warning("X API credentials incomplete; using NoopXPostWriter (log-only).")
        return NoopXPostWriter(cfg)

    try:
        return TweepyXPostWriter(cfg)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to init Tweepy client; using NoopXPostWriter (log-only).")
        return NoopXPostWriter(cfg)
