"""Persistent local state for crash/restart safety (atomic JSON).

Persists tweet-ID bookkeeping (seen / ignored-at-startup / processed / failed),
paused/error flags, the last error, reply-rate-limit timestamps, and a
diagnostics-only snapshot of the queue.

Restart semantics (important): the in-memory command queue is NEVER restored from
disk. ``queued_snapshot`` exists only for debugging; the controller starts with an
empty queue. Combined with the reader's no-backfill startup, nothing from before a
restart is executed. The ID sets ARE restored so the same tweet never runs twice.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from threading import Lock

logger = logging.getLogger("tweet_robot.state_store")


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = Lock()

        self.source_tweet_id: str | None = None
        self.seen_tweet_ids: set[str] = set()
        self.ignored_at_startup_tweet_ids: set[str] = set()
        self.processed_tweet_ids: set[str] = set()
        self.failed_tweet_ids: set[str] = set()
        self.queued_snapshot: list[dict] = []
        self.paused: bool = False
        self.error: bool = False
        self.last_error: str | None = None
        self.reply_times_normal: list[float] = []
        self.reply_times_all: list[float] = []

    # ------------------------------------------------------------------ #
    # Load / save                                                        #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if not self.path.exists():
            logger.info("No existing state file at %s (fresh start).", self.path)
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read state file %s (%s); starting fresh.", self.path, exc)
            return

        self.source_tweet_id = data.get("source_tweet_id")
        self.seen_tweet_ids = set(data.get("seen_tweet_ids", []))
        self.ignored_at_startup_tweet_ids = set(data.get("ignored_at_startup_tweet_ids", []))
        self.processed_tweet_ids = set(data.get("processed_tweet_ids", []))
        self.failed_tweet_ids = set(data.get("failed_tweet_ids", []))
        # queued_snapshot is intentionally NOT used to repopulate the live queue.
        self.queued_snapshot = list(data.get("queued_snapshot", []))
        self.paused = bool(data.get("paused", False))
        self.error = bool(data.get("error", False))
        self.last_error = data.get("last_error")
        self.reply_times_normal = list(data.get("reply_times_normal", []))
        self.reply_times_all = list(data.get("reply_times_all", []))
        logger.info(
            "Loaded state: %d seen, %d processed, %d failed, paused=%s, error=%s",
            len(self.seen_tweet_ids),
            len(self.processed_tweet_ids),
            len(self.failed_tweet_ids),
            self.paused,
            self.error,
        )

    def _serialize(self) -> dict:
        return {
            "source_tweet_id": self.source_tweet_id,
            "seen_tweet_ids": sorted(self.seen_tweet_ids),
            "ignored_at_startup_tweet_ids": sorted(self.ignored_at_startup_tweet_ids),
            "processed_tweet_ids": sorted(self.processed_tweet_ids),
            "failed_tweet_ids": sorted(self.failed_tweet_ids),
            "queued_snapshot": self.queued_snapshot,
            "paused": self.paused,
            "error": self.error,
            "last_error": self.last_error,
            "reply_times_normal": self.reply_times_normal,
            "reply_times_all": self.reply_times_all,
        }

    def save(self) -> None:
        """Atomically persist state (tmpfile + os.replace). Never raises."""
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = self._serialize()
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(payload, f, indent=2)
                        f.write("\n")
                    os.replace(tmp_name, self.path)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
            except Exception:  # noqa: BLE001
                logger.exception("Failed to save state file %s (continuing).", self.path)

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #

    def is_seen(self, tweet_id: str) -> bool:
        return tweet_id in self.seen_tweet_ids

    def mark_seen(self, *tweet_ids: str) -> None:
        self.seen_tweet_ids.update(tweet_ids)

    def mark_processed(self, tweet_id: str) -> None:
        self.seen_tweet_ids.add(tweet_id)
        self.processed_tweet_ids.add(tweet_id)

    def mark_failed(self, tweet_id: str) -> None:
        self.seen_tweet_ids.add(tweet_id)
        self.failed_tweet_ids.add(tweet_id)
