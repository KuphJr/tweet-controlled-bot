#!/usr/bin/env python
"""Entry point for the tweet-controlled SO-101 duck robot.

People quote-tweet OR comment on a source tweet to ask the robot to place an
orange/green/yellow/pink duck on a target. The robot removes whatever duck is on
the target (if any), then places the requested one, narrating via TTS and
acknowledging via X replies. Deterministic, validated enums decide which policy
runs; the LLM only parses text and classifies images.

Run (safe ramp):

    # 1) No hardware movement, no posting — poll/parse/vision/log safely:
    uv run python scripts/tweet_robot/run_tweet_robot.py \\
        --source-tweet-id 1234567890 --no-robot --dry-run

    # 2) Full live run:
    uv run python scripts/tweet_robot/run_tweet_robot.py --source-tweet-id 1234567890

Ctrl+C (or an admin `shutdown`) stops polling, stops audio, disables robot torque,
and exits cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure this package dir (for flat sibling imports) and scripts/ (so the policy
# runner can import the example's StopAtNeutralStrategy) are importable, whether
# launched as a script or a module.
_PKG_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _PKG_DIR.parent
for _p in (str(_PKG_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.utils.process import ProcessSignalHandler  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from command_parser import CommandParser  # noqa: E402
from config import AppConfig  # noqa: E402
from controller import TweetRobotController  # noqa: E402
from reply_generator import ReplyGenerator  # noqa: E402
from robot_policy_runner import RobotPolicyRunner  # noqa: E402
from state_store import StateStore  # noqa: E402
from tts import TTS  # noqa: E402
from twitter_reader import TwitterReader  # noqa: E402
from workspace_detector import WorkspaceDetector  # noqa: E402
from x_post_writer import build_x_post_writer  # noqa: E402

logger = logging.getLogger("tweet_robot.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-tweet-id", required=True, help="Tweet ID users quote/comment to command the robot.")
    p.add_argument("--poll-interval-s", type=float, default=10.0, help="Seconds between TwitterApi.io polls.")
    p.add_argument("--dry-run", action="store_true", help="Do not move the robot or post; parse/vision/log only.")
    p.add_argument("--no-tts", action="store_true", help="Disable ElevenLabs narration.")
    p.add_argument("--no-robot", action="store_true", help="Skip policy execution (vision still runs if possible).")
    p.add_argument("--state-file", default=None, help="Path to JSON state file.")
    p.add_argument("--log-file", default=None, help="Path to JSONL command log.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.from_env()
    cfg.poll_interval_s = args.poll_interval_s
    cfg.dry_run = args.dry_run
    cfg.no_tts = args.no_tts
    cfg.no_robot = args.no_robot
    if args.state_file:
        cfg.state_file = Path(args.state_file)
    if args.log_file:
        cfg.log_file = Path(args.log_file)
    return cfg


def _log_banner(cfg: AppConfig, source_tweet_id: str) -> None:
    logger.info("=" * 72)
    logger.info("Tweet-controlled SO-101 duck robot")
    logger.info("=" * 72)
    logger.info("Source tweet : %s", source_tweet_id)
    logger.info("Bot account  : @%s   Admin: @%s", cfg.bot_handle, cfg.admin_handle)
    logger.info(
        "Modes        : dry_run=%s no_tts=%s no_robot=%s",
        cfg.dry_run,
        cfg.no_tts,
        cfg.no_robot,
    )
    logger.info("Poll interval: %.0fs   Queue size: %d   Policy timeout: %ds",
                cfg.poll_interval_s, cfg.max_queue_size, cfg.policy_timeout_s)
def _shutdown(
    cfg: AppConfig,
    shutdown_event,
    tts: TTS,
    runner: RobotPolicyRunner,
    store: StateStore,
    *,
    disable_torque: bool = True,
) -> None:
    action = "saving state" if not disable_torque else "disabling torque, saving state"
    logger.info("Graceful shutdown: stopping audio, %s...", action)
    shutdown_event.set()
    try:
        tts.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Error stopping TTS.")
    if disable_torque:
        try:
            runner.disable_torque_on_shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("Error disabling torque.")
    else:
        logger.info("Restart: leaving torque enabled so the arm holds across re-exec.")
    try:
        store.save()
    except Exception:  # noqa: BLE001
        logger.exception("Error saving state.")
    logger.info("Shutdown complete.")


def main() -> int:
    args = parse_args()
    init_logging()

    cfg = build_config(args)
    source_tweet_id = args.source_tweet_id
    _log_banner(cfg, source_tweet_id)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # State store
    store = StateStore(cfg.state_file)
    store.load()
    store.source_tweet_id = source_tweet_id

    # Components
    parser = CommandParser(cfg)
    detector = WorkspaceDetector(cfg)
    reply_gen = ReplyGenerator(cfg)
    tts = TTS(cfg)
    runner = RobotPolicyRunner(cfg, shutdown_event)
    writer = build_x_post_writer(cfg)
    writer.load_timestamps(store.reply_times_normal, store.reply_times_all)

    start_time = time.time()
    reader = TwitterReader(cfg, source_tweet_id, start_time)

    controller = TweetRobotController(
        cfg,
        source_tweet_id=source_tweet_id,
        shutdown_event=shutdown_event,
        reader=reader,
        parser=parser,
        detector=detector,
        reply_gen=reply_gen,
        writer=writer,
        tts=tts,
        runner=runner,
        state_store=store,
    )

    try:
        # No-backfill startup: mark all currently-visible quotes/replies as seen.
        if not cfg.twitterapi_io_api_key:
            logger.error("TWITTERAPI_IO_API_KEY not set; cannot poll.")
            return 2
        current_ids = reader.fetch_current_ids()
        store.ignored_at_startup_tweet_ids |= current_ids
        store.mark_seen(*current_ids)
        store.save()

        controller.start_polling()
        controller.run()  # blocks until shutdown_event is set
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received.")
    finally:
        # On restart, keep torque on so the arm holds across the brief re-exec.
        _shutdown(cfg, shutdown_event, tts, runner, store, disable_torque=not controller.restart_requested)

    # Admin "restart": re-exec this process with the same arguments. Startup will
    # clear error/paused state and re-apply no-backfill, giving a clean slate.
    if controller.restart_requested:
        logger.info("Restart requested by admin; re-executing: %s", " ".join(sys.argv))
        os.execv(sys.executable, [sys.executable, *sys.argv])

    return 0

if __name__ == "__main__":
    sys.exit(main())
