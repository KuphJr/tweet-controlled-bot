"""TweetRobotController: state machine, queue, and per-command orchestration.

Threads:
  * Poller thread: polls TwitterApi.io (quotes + replies), applies acceptance
    rules, enqueues valid user commands (with immediate acknowledgement), and
    dispatches admin commands.
  * Executor (main thread, via ``run()``): pops one command at a time and runs
    the remove->place pipeline. Never runs two policies concurrently.

Key behaviors (see the plan):
  * Per-author limit: one active-or-queued command per author.
  * Immediate accept/queued acknowledgement (execution may take ~a minute).
  * Empty target is recoverable: skip removal, place directly.
  * No success replies (the livestream shows the result).
  * pause finishes the current command then stops; shutdown / Ctrl+C interrupt.
  * dry-run / no-robot simulate the pipeline (vision + replies + logs) without
    moving the robot or failing on verification.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from threading import Event, Lock, Thread

from config import (
    AppConfig,
    Command,
    CommandSource,
    ControllerState,
    PolicyAction,
    WorkspaceState,
    color_on_target_state,
    get_policy_path,
    normalize_handle,
)
from command_parser import CommandParser
from reply_generator import ReplyCategory, ReplyGenerator
from robot_policy_runner import RobotPolicyRunner
from state_store import StateStore
from tts import TTS
from twitter_reader import RawItem, TwitterReader
from workspace_detector import WorkspaceDetector
from x_post_writer import XPostWriter

logger = logging.getLogger("tweet_robot.controller")

# Admin command keyword matching (checked in priority order).
# Checked in priority order. "restart" is checked before "reset error" /
# "resume" to avoid any ambiguity, and "shutdown" stays first.
_ADMIN_KEYWORDS = [
    ("shutdown", "shutdown"),
    ("restart", "restart"),
    ("resume", "resume"),
    ("pause", "pause"),
    ("clear queue", "clear_queue"),
    ("clearqueue", "clear_queue"),
    ("reset error", "reset_error"),
    ("clear error", "reset_error"),
    ("status", "status"),
]

# Sentinel returned by the pipeline when shutdown interrupts a run (not an error).
_ABORTED = "__aborted__"

_GENERATED_SELF_REPLY_FRAGMENTS = (
    "queued!",
    "you're #",
    "I'll place",
    "already have a command",
    "one request at a time",
    "queue is full",
    "couldn't figure out",
    "something is broken",
    "has been notified",
    "status:",
    "paused.",
    "resumed.",
    "queue cleared.",
    "error cleared.",
    "shutting down safely",
    "restarting now",
    "the robot hit an error",
)


class TweetRobotController:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        source_tweet_id: str,
        shutdown_event: Event,
        reader: TwitterReader,
        parser: CommandParser,
        detector: WorkspaceDetector,
        reply_gen: ReplyGenerator,
        writer: XPostWriter,
        tts: TTS,
        runner: RobotPolicyRunner,
        state_store: StateStore,
    ):
        self.cfg = cfg
        self.source_tweet_id = str(source_tweet_id)
        self.shutdown_event = shutdown_event
        self.reader = reader
        self.parser = parser
        self.detector = detector
        self.reply_gen = reply_gen
        self.writer = writer
        self.tts = tts
        self.runner = runner
        self.state_store = state_store

        self._lock = Lock()
        self._queue: deque[Command] = deque()
        self._active_authors: set[str] = set()
        self._current_author: str | None = None
        self._state = ControllerState.STARTING
        self._pause_requested = False
        self._restart_requested = False
        self._simulating = cfg.dry_run or cfg.no_robot

        self._poller_thread: Thread | None = None

    @property
    def restart_requested(self) -> bool:
        return self._restart_requested

    # ================================================================== #
    # Lifecycle                                                          #
    # ================================================================== #

    def start_polling(self) -> None:
        self._poller_thread = Thread(target=self._poll_loop, name="poller", daemon=True)
        self._poller_thread.start()

    def run(self) -> None:
        """Executor loop (main thread). Returns when shutdown is requested."""
        with self._lock:
            # Restarting/re-running the script always clears prior error/paused
            # state: re-running implies the operator has fixed the workspace, so
            # the robot should "just work" again without a manual reset step.
            self.state_store.error = False
            self.state_store.last_error = None
            self.state_store.paused = False
            self._pause_requested = False
            self._state = ControllerState.IDLE
        logger.info("Controller running. Initial state: %s", self._state.value)

        while not self.shutdown_event.is_set():
            cmd = self._next_command()
            if cmd is None:
                self.shutdown_event.wait(0.2)
                continue
            self._execute(cmd)

        with self._lock:
            self._state = ControllerState.SHUTTING_DOWN
        logger.info("Controller executor loop exited (shutting down).")

    def _next_command(self) -> Command | None:
        with self._lock:
            if self._state == ControllerState.ERROR:
                return None
            if self._pause_requested:
                self._state = ControllerState.PAUSED
                return None
            if not self._queue:
                if self._state != ControllerState.PAUSED:
                    self._state = ControllerState.IDLE
                return None
            cmd = self._queue.popleft()
            self._current_author = normalize_handle(cmd.author_handle)
            self._state = ControllerState.BUSY
            self._refresh_snapshot_locked()
            return cmd

    # ================================================================== #
    # Polling + acceptance                                               #
    # ================================================================== #

    def _poll_loop(self) -> None:
        logger.info("Poller thread started (interval=%.0fs).", self.cfg.poll_interval_s)
        while not self.shutdown_event.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001
                logger.exception("Poll cycle failed (continuing).")
            self.shutdown_event.wait(self.cfg.poll_interval_s)
        logger.info("Poller thread exiting.")

    def _poll_once(self) -> None:
        with self._lock:
            seen = set(self.state_store.seen_tweet_ids)
        items = self.reader.fetch_new(seen)
        if not items:
            return
        logger.info("Discovered %d new item(s).", len(items))
        for item in items:  # oldest-first
            if self.shutdown_event.is_set():
                break
            try:
                self._handle_item(item)
            except Exception:  # noqa: BLE001
                logger.exception("Failed handling item %s (marking seen).", item.tweet_id)
                with self._lock:
                    self.state_store.mark_seen(item.tweet_id)
        self.state_store.save()

    def _handle_item(self, item: RawItem) -> None:
        with self._lock:
            if self.state_store.is_seen(item.tweet_id):
                return
        author_norm = normalize_handle(item.author_handle)

        # In separate-account mode, never ingest the posting account's own replies.
        # In single-account mode (BOT_HANDLE == ADMIN_HANDLE), @KuphDev must still
        # be able to issue admin and normal commands, so only generated status/ack
        # text is ignored below.
        single_account = self.cfg.bot_handle_norm == self.cfg.admin_handle_norm
        if author_norm == self.cfg.bot_handle_norm and not single_account:
            with self._lock:
                self.state_store.mark_seen(item.tweet_id)
            return

        if single_account and author_norm == self.cfg.admin_handle_norm and self._looks_like_generated_self_reply(item.text):
            logger.info("Ignoring generated self-authored reply %s.", item.tweet_id)
            with self._lock:
                self.state_store.mark_seen(item.tweet_id)
            return

        # Admin commands take priority (only from the admin handle).
        if author_norm == self.cfg.admin_handle_norm:
            admin_action = self._parse_admin(item.text)
            if admin_action is not None:
                self._dispatch_admin(admin_action, item)
                with self._lock:
                    self.state_store.mark_seen(item.tweet_id)
                return
            # Otherwise the admin is issuing a normal duck command -> fall through.

        self._handle_user_command(item)

    def _handle_user_command(self, item: RawItem) -> None:
        author_norm = normalize_handle(item.author_handle)

        # ERROR state: refuse new commands with a clear message.
        with self._lock:
            in_error = self._state == ControllerState.ERROR
        if in_error:
            self._post_reply(self.reply_gen.error_user_reply(), item.tweet_id)
            self.tts.speak(f"Something is broken. But I have tagged @{self.cfg.admin_handle}.")
            with self._lock:
                self.state_store.mark_seen(item.tweet_id)
            return

        parsed = self.parser.parse(item.text)
        if not parsed.valid or parsed.requested_color is None:
            logger.info("Invalid command from @%s: %r (%s)", item.author_handle, item.text, parsed.reason)
            if author_norm == self.cfg.bot_handle_norm:
                logger.info("Ignoring invalid self-authored item %s without replying.", item.tweet_id)
                with self._lock:
                    self.state_store.mark_seen(item.tweet_id)
                return
            self._post_reply(
                self.reply_gen.generate(
                    ReplyCategory.INVALID, author_name=item.author_name, original_text=item.text
                ),
                item.tweet_id,
            )
            with self._lock:
                self.state_store.mark_seen(item.tweet_id)
            return

        decision = "accepted"
        position: int | None = None
        with self._lock:
            if author_norm in self._active_authors:
                decision = "duplicate"
            elif len(self._queue) >= self.cfg.max_queue_size:
                decision = "queue_full"
            else:
                position = len(self._queue) + 1
                cmd = Command(
                    tweet_id=item.tweet_id,
                    source=item.source,
                    author_handle=item.author_handle,
                    author_name=item.author_name,
                    raw_text=item.text,
                    requested_color=parsed.requested_color,
                    enqueued_at=time.time(),
                    replies_attempted=["accepted_ack"],
                )
                self._queue.append(cmd)
                self._active_authors.add(author_norm)
                self._refresh_snapshot_locked()
            self.state_store.mark_seen(item.tweet_id)

        # Replies happen outside the lock (network I/O).
        if decision == "duplicate":
            logger.info("@%s already has a command queued/active; not enqueuing.", item.author_handle)
            self._post_reply(
                self.reply_gen.generate(
                    ReplyCategory.DUPLICATE_AUTHOR, author_name=item.author_name, original_text=item.text
                ),
                item.tweet_id,
            )
        elif decision == "queue_full":
            logger.info("Queue full; rejecting command from @%s.", item.author_handle)
            self._post_reply(
                self.reply_gen.generate(
                    ReplyCategory.QUEUE_FULL, author_name=item.author_name, original_text=item.text
                ),
                item.tweet_id,
            )
        else:
            logger.info(
                "Queued %s command from @%s (#%s).",
                parsed.requested_color.value,
                item.author_handle,
                position,
            )
            self._post_reply(
                self.reply_gen.generate(
                    ReplyCategory.ACCEPTED,
                    author_name=item.author_name,
                    original_text=item.text,
                    color=parsed.requested_color,
                    position=position,
                ),
                item.tweet_id,
            )

    # ================================================================== #
    # Admin                                                              #
    # ================================================================== #

    @staticmethod
    def _parse_admin(text: str):
        from config import AdminCommand

        low = (text or "").lower()
        for keyword, action in _ADMIN_KEYWORDS:
            if keyword in low:
                return AdminCommand(action)
        return None

    @staticmethod
    def _looks_like_generated_self_reply(text: str) -> bool:
        """Best-effort loop guard for single-account mode.

        Direct parent-ID filtering in ``TwitterReader`` handles the common nested
        reply case. This fallback catches generated acknowledgement/status text if
        TwitterApi.io returns a self-authored reply without parent metadata.
        """
        low = (text or "").strip().lower()
        if any(fragment in low for fragment in _GENERATED_SELF_REPLY_FRAGMENTS):
            return True
        return "duck" in low and ("queued" in low or "queue position" in low or "soon" in low)

    def _dispatch_admin(self, action, item: RawItem) -> None:
        from config import AdminCommand

        logger.info("Admin command from @%s: %s", item.author_handle, action.value)
        reply: str | None = None

        if action == AdminCommand.PAUSE:
            with self._lock:
                self._pause_requested = True
                self.state_store.paused = True
                if self._state == ControllerState.IDLE:
                    self._state = ControllerState.PAUSED
            reply = "Paused. I'll finish the current command (if any) and stop taking new ones."
        elif action == AdminCommand.RESUME:
            with self._lock:
                self._pause_requested = False
                self.state_store.paused = False
                if self._state == ControllerState.PAUSED:
                    self._state = ControllerState.IDLE
            reply = "Resumed. Processing queued commands again."
        elif action == AdminCommand.SHUTDOWN:
            reply = "Shutting down safely. Thanks for watching!"
            self._post_reply(self.reply_gen.admin_status_reply(reply), item.tweet_id)
            self.shutdown_event.set()
            return
        elif action == AdminCommand.RESTART:
            reply = "Restarting now — back in a moment!"
            self._post_reply(self.reply_gen.admin_status_reply(reply), item.tweet_id)
            with self._lock:
                self._restart_requested = True
            self.shutdown_event.set()
            return
        elif action == AdminCommand.STATUS:
            reply = self._status_text()
        elif action == AdminCommand.CLEAR_QUEUE:
            with self._lock:
                self._queue.clear()
                self._active_authors = {self._current_author} if self._current_author else set()
                self._refresh_snapshot_locked()
            reply = "Queue cleared."
        elif action == AdminCommand.RESET_ERROR:
            with self._lock:
                self._state = (
                    ControllerState.PAUSED if self._pause_requested else ControllerState.IDLE
                )
                self.state_store.error = False
                self.state_store.last_error = None
            reply = "Error cleared. Back to normal."

        if reply is not None:
            self._post_reply(self.reply_gen.admin_status_reply(reply), item.tweet_id)
        self.state_store.save()

    def _status_text(self) -> str:
        with self._lock:
            state = self._state.value
            qlen = len(self._queue)
            err = self.state_store.last_error
        return f"Status: {state}. Queue: {qlen}/{self.cfg.max_queue_size}. Error: {err or 'none'}."

    # ================================================================== #
    # Per-command pipeline                                               #
    # ================================================================== #

    def _execute(self, cmd: Command) -> None:
        author_norm = normalize_handle(cmd.author_handle)
        log_entry = self._new_log_entry(cmd)
        failure_reason: str | None = None
        try:
            failure_reason = self._pipeline(cmd, log_entry)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during command execution.")
            failure_reason = f"unexpected error: {exc}"

        aborted = failure_reason == _ABORTED or self.shutdown_event.is_set()
        real_failure = bool(failure_reason) and not aborted and not self._simulating

        if real_failure:
            log_entry["tts_attempted"].append("Error encountered.")
            log_entry["replies_attempted"].append("admin_error_notification")
            self._announce_error(failure_reason or "unknown error")
            log_entry["success"] = False
            log_entry["error_message"] = failure_reason
        else:
            log_entry["success"] = not aborted
            if failure_reason and failure_reason != _ABORTED:
                log_entry["error_message"] = f"(simulated) {failure_reason}"
            if aborted:
                log_entry["error_message"] = "aborted (shutdown)"

        with self._lock:
            self._active_authors.discard(author_norm)
            self._current_author = None
            if real_failure:
                self._state = ControllerState.ERROR
                self.state_store.error = True
                self.state_store.last_error = failure_reason
                self.state_store.mark_failed(cmd.tweet_id)
            elif not aborted:
                self.state_store.mark_processed(cmd.tweet_id)
                self._state = ControllerState.PAUSED if self._pause_requested else ControllerState.IDLE
            self._refresh_snapshot_locked()
            log_entry["final_state"] = self._state.value

        self._write_log(log_entry)
        self.state_store.save()

    def _pipeline(self, cmd: Command, log_entry: dict) -> str | None:
        """Run remove->place. Returns a failure reason, ``_ABORTED``, or None."""
        req = cmd.requested_color

        request_tts = (
            f"Executing the request from {cmd.author_name} @{cmd.author_handle} "
            f"to place the {req.value} duck on the target."
        )
        log_entry["tts_attempted"].append(request_tts)
        self.tts.speak(request_tts)

        # --- 1. Initial workspace state ---
        before = self.detector.detect_state(self.runner.capture_workspace_image)
        log_entry["workspace_state_before"] = before.state.value
        logger.info("Initial workspace state: %s (conf=%.2f)", before.state.value, before.confidence)
        if before.is_error and not self._simulating:
            return f"initial vision error: {before.error_reason or before.description}"

        # --- 2/3. Removal (skip if target already clear) ---
        if before.state == WorkspaceState.EMPTY:
            logger.info("Target already empty; skipping removal.")
        elif before.state.on_target_color is not None:
            current = before.state.on_target_color
            log_entry["tts_attempted"].append(f"Removing the {current.value} duck from the target.")
            self.tts.speak(f"Removing the {current.value} duck from the target.")
            res = self.runner.run_policy(
                get_policy_path(current, PolicyAction.REMOVE),
                f"Remove {current.value} duck from target",
                self.cfg.policy_timeout_s,
            )
            log_entry["policies_run"].append(f"remove_{current.value}:{res.stop_reason}")
            if res.stop_reason == "shutdown":
                return _ABORTED
            if not res.success and not self._simulating:
                return f"remove policy failed ({res.stop_reason}: {res.error_message})"

            if self._interruptible_wait(self.cfg.post_policy_wait_s):
                return _ABORTED
            after_removal = self.detector.detect_state(self.runner.capture_workspace_image)
            log_entry["workspace_state_after_removal"] = after_removal.state.value
            logger.info("Post-removal state: %s", after_removal.state.value)
            # Primary condition: target is clear. Do not require the duck to be home.
            if not self._simulating and (
                after_removal.is_error or after_removal.state != WorkspaceState.EMPTY
            ):
                return "removal verification failed: target not clear"
        else:
            # before.state == ERROR while simulating: nothing to remove deterministically.
            logger.info("Simulating: initial state %s, skipping removal.", before.state.value)

        # --- 4. Placement ---
        placement_tts = f"Placing the {req.value} duck on the target."
        log_entry["tts_attempted"].append(placement_tts)
        self.tts.speak(placement_tts)
        res = self.runner.run_policy(
            get_policy_path(req, PolicyAction.PLACE),
            f"Place {req.value} duck on target",
            self.cfg.policy_timeout_s,
        )
        log_entry["policies_run"].append(f"place_{req.value}:{res.stop_reason}")
        if res.stop_reason == "shutdown":
            return _ABORTED
        if not res.success and not self._simulating:
            return f"place policy failed ({res.stop_reason}: {res.error_message})"

        if self._interruptible_wait(self.cfg.post_policy_wait_s):
            return _ABORTED
        after_place = self.detector.detect_state(self.runner.capture_workspace_image)
        log_entry["workspace_state_after_place"] = after_place.state.value
        logger.info("Post-placement state: %s", after_place.state.value)
        expected = color_on_target_state(req)
        if not self._simulating and (after_place.is_error or after_place.state != expected):
            return (
                f"placement verification failed: expected {expected.value}, "
                f"saw {after_place.state.value}"
            )
        return None

    def _interruptible_wait(self, seconds: float) -> bool:
        """Wait, returning True if shutdown was requested during the wait."""
        return self.shutdown_event.wait(seconds)

    def _announce_error(self, reason: str) -> None:
        logger.error("Entering ERROR state: %s", reason)
        self.tts.speak("Error encountered.")
        self._post_reply(self.reply_gen.error_admin_notify(reason), None, critical=True)

    # ================================================================== #
    # Helpers                                                            #
    # ================================================================== #

    def _post_reply(self, text: str, in_reply_to: str | None, *, critical: bool = False) -> bool:
        ok = self.writer.send(text, in_reply_to, critical=critical)
        # Persist updated rate-limit timestamps.
        normal_times, all_times = self.writer.export_timestamps()
        with self._lock:
            self.state_store.reply_times_normal = normal_times
            self.state_store.reply_times_all = all_times
        return ok

    def _new_log_entry(self, cmd: Command) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tweet_id": cmd.tweet_id,
            "source": cmd.source.value if isinstance(cmd.source, CommandSource) else str(cmd.source),
            "author_handle": cmd.author_handle,
            "author_name": cmd.author_name,
            "raw_text": cmd.raw_text,
            "parsed_command": cmd.requested_color.value,
            "workspace_state_before": None,
            "workspace_state_after_removal": None,
            "workspace_state_after_place": None,
            "final_state": None,
            "policies_run": [],
            "replies_attempted": list(cmd.replies_attempted),
            "tts_attempted": list(cmd.tts_attempted),
            "success": None,
            "error_message": None,
            "simulated": self._simulating,
        }

    def _write_log(self, entry: dict) -> None:
        try:
            self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_file.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write command log (continuing).")

    def _refresh_snapshot_locked(self) -> None:
        """Update the diagnostics-only queue snapshot. Caller must hold the lock."""
        self.state_store.queued_snapshot = [c.to_dict() for c in self._queue]
