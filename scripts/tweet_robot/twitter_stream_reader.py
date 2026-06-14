"""TwitterApi.io WebSocket ingestion for low-cost, low-latency commands."""

from __future__ import annotations

import json
import logging
import time
from threading import Event, Thread
from typing import Any, Callable

from config import AppConfig, CommandSource
from twitter_reader import RawItem, _parse_created_at

logger = logging.getLogger("tweet_robot.twitter_stream_reader")

_WS_URL = "wss://ws.twitterapi.io/twitter/tweet/websocket"


class TwitterStreamReader:
    """Background WebSocket reader that emits source-conversation ``RawItem``s."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        source_tweet_id: str,
        start_time: float,
        shutdown_event: Event,
        on_item: Callable[[RawItem], None],
    ):
        self.cfg = cfg
        self.source_tweet_id = str(source_tweet_id)
        self.start_time = start_time
        self.shutdown_event = shutdown_event
        self.on_item = on_item

        self._thread: Thread | None = None
        self._ws = None

    def start(self) -> None:
        if not self.cfg.twitterapi_stream_enabled:
            logger.info("TwitterApi.io WebSocket stream disabled.")
            return
        if not self.cfg.twitterapi_io_api_key:
            logger.warning("TWITTERAPI_IO_API_KEY not set; WebSocket stream disabled.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run_loop, name="twitterapi-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing TwitterApi.io WebSocket.", exc_info=True)

    def _run_loop(self) -> None:
        try:
            import websocket
        except Exception:  # noqa: BLE001
            logger.exception("websocket-client is not installed; TwitterApi.io stream disabled.")
            return

        reconnect_s = max(1.0, float(self.cfg.twitterapi_stream_reconnect_s))
        while not self.shutdown_event.is_set():
            ws = websocket.WebSocketApp(
                _WS_URL,
                header=[f"x-api-key: {self.cfg.twitterapi_io_api_key}"],
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._ws = ws
            logger.info("Connecting TwitterApi.io WebSocket stream...")
            try:
                ws.run_forever(ping_interval=40, ping_timeout=30)
            except Exception:  # noqa: BLE001
                logger.exception("TwitterApi.io WebSocket run_forever raised.")
            finally:
                self._ws = None

            if self.shutdown_event.is_set():
                break
            logger.warning("TwitterApi.io WebSocket disconnected; reconnecting in %.0fs.", reconnect_s)
            self.shutdown_event.wait(reconnect_s)

        logger.info("TwitterApi.io WebSocket stream exiting.")

    def _on_open(self, _ws) -> None:
        logger.info("TwitterApi.io WebSocket connected.")

    def _on_close(self, _ws, close_status_code, close_msg) -> None:
        logger.info("TwitterApi.io WebSocket closed: code=%s msg=%s", close_status_code, close_msg)

    def _on_error(self, _ws, error) -> None:
        logger.warning("TwitterApi.io WebSocket error: %s", error)

    def _on_message(self, _ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("TwitterApi.io WebSocket sent non-JSON message: %r", message[:200])
            return

        event_type = payload.get("event_type")
        if event_type in {"connected", "ping"}:
            logger.debug("TwitterApi.io WebSocket event: %s", event_type)
            return
        if event_type not in {"tweet", "fast_tweet"}:
            logger.debug("Ignoring TwitterApi.io WebSocket event_type=%r.", event_type)
            return

        for raw in self._extract_tweets(payload):
            item = self._to_item(raw)
            if item is None:
                continue
            try:
                self.on_item(item)
            except Exception:  # noqa: BLE001
                logger.exception("Error handling streamed item %s.", item.tweet_id)

    @staticmethod
    def _extract_tweets(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload.get("tweet"), dict):
            return [payload["tweet"]]
        tweets = payload.get("tweets")
        if isinstance(tweets, list):
            return [tw for tw in tweets if isinstance(tw, dict)]
        return []

    def _to_item(self, raw: dict[str, Any]) -> RawItem | None:
        tweet_id = raw.get("id")
        if not tweet_id:
            return None

        parent_id = self._parent_id(raw)
        conversation_id = str(raw.get("conversationId") or raw.get("conversation_id") or "") or None
        if str(tweet_id) == self.source_tweet_id:
            return None
        if not parent_id:
            return None
        if parent_id != self.source_tweet_id and conversation_id != self.source_tweet_id:
            return None

        created_at = self._created_at(raw)
        if created_at < self.start_time:
            return None

        author = raw.get("author") or {}
        author_handle = (
            author.get("userName")
            or author.get("username")
            or raw.get("screen_name")
            or raw.get("userName")
            or raw.get("username")
            or ""
        )
        author_name = author.get("name") or raw.get("display_name") or ""
        text = str(raw.get("text", "") or "")
        source = CommandSource.QUOTE if str(raw.get("type", "")).lower() == "quote" else CommandSource.REPLY

        return RawItem(
            tweet_id=str(tweet_id),
            source=source,
            author_handle=str(author_handle or ""),
            author_name=str(author_name or ""),
            text=text,
            created_at=created_at,
            in_reply_to_tweet_id=parent_id,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _parent_id(raw: dict[str, Any]) -> str | None:
        for key in ("inReplyToId", "inReplyToTweetId", "inReplyToStatusId", "in_reply_to_status_id"):
            value = raw.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _created_at(raw: dict[str, Any]) -> float:
        created_at = raw.get("createdAt")
        if created_at:
            return _parse_created_at(created_at)
        for key in ("created_ms", "snowflake_created_ms"):
            value = raw.get(key)
            if value:
                try:
                    return float(value) / 1000.0
                except (TypeError, ValueError):
                    pass
        return time.time()
