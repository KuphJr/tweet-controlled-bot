#!/usr/bin/env python
"""Run the trained ACT "remove duck from target" policy on a physical SO-101 follower.

Model checkpoint (extracted from cloud training):

    outputs/train/act_remove_duck_from_target/checkpoints/last/pretrained_model

Run with::

    uv run python scripts/run_so101_policy_remove_duck.py
"""

from __future__ import annotations

import logging
import sys

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.rollout import (
    BaseStrategy,
    BaseStrategyConfig,
    RolloutConfig,
    build_rollout_context,
)
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

POLICY_PATH = "outputs/train/act_remove_duck_from_target/checkpoints/last/pretrained_model"
TASK = "Remove the duck from the target"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower"

CAMERA_NAME = "top"
CAMERA_PATH = "/dev/video0"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

FPS = 30
DURATION_S = 60.0

DISPLAY_DATA = False

logger = logging.getLogger("run_so101_policy_remove_duck")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    init_logging()
    logger.info("=" * 72)
    logger.info("SO-101 ACT policy rollout (remove duck from target)")
    logger.info("=" * 72)
    logger.info("Policy path : %s", POLICY_PATH)
    logger.info("Task        : %s", TASK)
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
    logger.info("Loop        : %d Hz for %.1f s", FPS, DURATION_S)

    camera_config = {
        CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=CAMERA_PATH,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        )
    }
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=camera_config,
    )

    logger.info("Loading policy config from '%s'...", POLICY_PATH)
    policy_config = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_config.pretrained_path = POLICY_PATH
    logger.info(
        "Policy config loaded (type=%s, device=%s, n_obs_steps=%d)",
        policy_config.type,
        policy_config.device,
        policy_config.n_obs_steps,
    )

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=float(FPS),
        duration=float(DURATION_S),
        task=TASK,
        display_data=DISPLAY_DATA,
    )

    if DISPLAY_DATA:
        from lerobot.utils.visualization_utils import init_rerun

        logger.info("Initialising Rerun viewer (session='run_so101_policy_remove_duck')")
        init_rerun(session_name="run_so101_policy_remove_duck")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)

    logger.info("Building rollout context (loads policy weights + connects robot)...")
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    logger.info(
        "Context ready. Robot '%s' connected, %d action keys: %s",
        ctx.hardware.robot_wrapper.inner.name,
        len(ctx.data.ordered_action_keys),
        ctx.data.ordered_action_keys,
    )

    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        logger.info("Starting %.1fs rollout at %d Hz...", DURATION_S, FPS)
        strategy.run(ctx)
        logger.info("Rollout finished cleanly.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl-C).")
    except Exception:
        logger.exception("Rollout failed with an exception:")
        raise
    finally:
        logger.info("Tearing down (returning to initial pose + disconnecting)...")
        strategy.teardown(ctx)
        logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
