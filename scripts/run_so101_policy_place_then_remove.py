#!/usr/bin/env python
"""Run place-then-remove ACT policies on a physical SO-101 follower.

Executes two sequential rollouts, each with neutral-stop (leave startup pose,
complete the task, return to startup pose):

1. **Place** — ``place_green_duck`` checkpoint
2. **Remove** — ``act_remove_duck_from_target`` checkpoint

Each phase connects, runs ``StopAtNeutralStrategy`` (from
``run_so101_policy``), tears down (return to initial pose + disconnect), then
the next phase starts fresh.

Run with::

    uv run python scripts/run_so101_policy_place_then_remove.py

Override policy checkpoints (defaults are the module constants below)::

    uv run python scripts/run_so101_policy_place_then_remove.py \\
        --place-policy outputs/train/place_green_duck/checkpoints/last/pretrained_model \\
        --remove-policy outputs/train/remove_green_duck/checkpoints/120000/pretrained_model

Manual test (on hardware):

1. Start with the duck off the target and the arm at the training neutral pose.
2. Run the command above.
3. Phase 1 should place the duck, return to neutral, and disconnect.
4. After a short pause, phase 2 should remove the duck, return to neutral, and disconnect.
5. Ctrl-C during phase 1 should tear down and skip phase 2.
6. Ctrl-C during phase 2 should still run remove-phase teardown.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from threading import Event

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

from run_so101_policy import StopAtNeutralStrategy

# --------------------------------------------------------------------------- #
# Shared configuration                                                        #
# --------------------------------------------------------------------------- #

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower"

CAMERA_NAME = "top"
CAMERA_PATH = "/dev/video0"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

FPS = 30
DISPLAY_DATA = False
PAUSE_BETWEEN_PHASES_S = 0.5

# Phase 1: place green duck on target
PLACE_POLICY_PATH = "outputs/train/place_green_duck/checkpoints/last/pretrained_model"
PLACE_TASK = "Place green duck on target"
PLACE_DURATION_S = 30.0

# Phase 2: remove duck from target
REMOVE_POLICY_PATH = "outputs/train/remove_green_duck/checkpoints/120000/pretrained_model"
REMOVE_TASK = "Remove green duck from target"
REMOVE_DURATION_S = 30.0

logger = logging.getLogger("run_so101_policy_place_then_remove")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _robot_config() -> SO101FollowerConfig:
    camera_config = {
        CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=CAMERA_PATH,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        )
    }
    # Skip Torque_Enable=0 on disconnect: the rubber-band counterspring on the
    # elbow keeps motor id_=3 under residual load even at the neutral pose,
    # which intermittently makes the disable-torque write fail. The strategy
    # already returns the arm to neutral before disconnect, so leaving torque
    # enabled across the place→remove handoff (and through final teardown) is
    # safe; physically power off the arm when finished.
    return SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=camera_config,
        disable_torque_on_disconnect=False,
    )


def run_phase(
    *,
    phase_label: str,
    policy_path: str,
    task: str,
    duration_s: float,
    shutdown_event: Event,
) -> bool:
    """Run one neutral-stop rollout. Returns False if shutdown was requested."""
    logger.info("=" * 72)
    logger.info("%s", phase_label)
    logger.info("=" * 72)
    logger.info("Policy path : %s", policy_path)
    logger.info("Task        : %s", task)
    logger.info("Loop        : %d Hz for %.1f s (duration cap)", FPS, duration_s)

    logger.info("Loading policy config from '%s'...", policy_path)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    logger.info(
        "Policy config loaded (type=%s, device=%s, n_obs_steps=%d)",
        policy_config.type,
        policy_config.device,
        policy_config.n_obs_steps,
    )

    cfg = RolloutConfig(
        robot=_robot_config(),
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=float(FPS),
        duration=float(duration_s),
        task=task,
        display_data=DISPLAY_DATA,
    )

    logger.info("Building rollout context (loads policy weights + connects robot)...")
    ctx = build_rollout_context(cfg, shutdown_event)
    logger.info(
        "Context ready. Robot '%s' connected, %d action keys: %s",
        ctx.hardware.robot_wrapper.inner.name,
        len(ctx.data.ordered_action_keys),
        ctx.data.ordered_action_keys,
    )

    strategy = StopAtNeutralStrategy(cfg.strategy)
    interrupted = False
    try:
        strategy.setup(ctx)
        logger.info("Starting %.1fs rollout at %d Hz...", duration_s, FPS)
        strategy.run(ctx)
        logger.info("Rollout finished cleanly.")
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted by user (Ctrl-C).")
    except Exception:
        logger.exception("Rollout failed with an exception:")
        raise
    finally:
        logger.info("Tearing down (returning to initial pose + disconnecting)...")
        strategy.teardown(ctx)
        logger.info("%s teardown complete.", phase_label)

    if interrupted or shutdown_event.is_set():
        return False
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--place-policy",
        default=PLACE_POLICY_PATH,
        help=f"Path to the place-phase policy checkpoint (default: {PLACE_POLICY_PATH!r}).",
    )
    parser.add_argument(
        "--remove-policy",
        default=REMOVE_POLICY_PATH,
        help=f"Path to the remove-phase policy checkpoint (default: {REMOVE_POLICY_PATH!r}).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    init_logging()
    logger.info("=" * 72)
    logger.info("SO-101 ACT place-then-remove rollout")
    logger.info("=" * 72)
    logger.info("Robot port  : %s  (id=%s)", ROBOT_PORT, ROBOT_ID)
    logger.info(
        "Camera      : %s -> %s  %dx%d @ %d fps  fourcc=%s",
        CAMERA_NAME,
        CAMERA_PATH,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        CAMERA_FPS,
        CAMERA_FOURCC,
    )
    logger.info("Phase 1     : %s", args.place_policy)
    logger.info("Phase 2     : %s", args.remove_policy)
    logger.info("Pause between phases: %.1f s", PAUSE_BETWEEN_PHASES_S)

    if DISPLAY_DATA:
        from lerobot.utils.visualization_utils import init_rerun

        logger.info("Initialising Rerun viewer (session='run_so101_policy_place_then_remove')")
        init_rerun(session_name="run_so101_policy_place_then_remove")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    place_ok = run_phase(
        phase_label="Phase 1/2: PLACE",
        policy_path=args.place_policy,
        task=PLACE_TASK,
        duration_s=PLACE_DURATION_S,
        shutdown_event=shutdown_event,
    )
    if not place_ok:
        logger.info("Skipping remove phase (shutdown or interrupt after place).")
        return 0

    if PAUSE_BETWEEN_PHASES_S > 0:
        logger.info("Pausing %.1f s before remove phase...", PAUSE_BETWEEN_PHASES_S)
        time.sleep(PAUSE_BETWEEN_PHASES_S)

    if shutdown_event.is_set():
        logger.info("Skipping remove phase (shutdown requested during pause).")
        return 0

    run_phase(
        phase_label="Phase 2/2: REMOVE",
        policy_path=args.remove_policy,
        task=REMOVE_TASK,
        duration_s=REMOVE_DURATION_S,
        shutdown_event=shutdown_event,
    )
    logger.info("Place-then-remove rollout finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
