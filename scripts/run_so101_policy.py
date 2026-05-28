#!/usr/bin/env python
"""Run a trained ACT policy on a physical SO-101 follower for a fixed duration.

This script does **not** depend on ``lerobot-record --policy.path`` (this local
CLI does not support that flag) or on the ``lerobot-rollout`` console script
(its entry point may not be installed in this branch).  Instead it uses the
in-tree ``lerobot.rollout`` Python API directly:

    PreTrainedConfig.from_pretrained()  -> load policy config
    RolloutConfig(...)                  -> wire policy + robot + strategy
    build_rollout_context(...)          -> connects robot, loads weights,
                                           builds pre/post processors,
                                           creates SyncInferenceEngine
    BaseStrategy(...).run(ctx)          -> autonomous control loop @ FPS Hz
    BaseStrategy(...).teardown(ctx)     -> safely returns to start pose +
                                           disconnects hardware

Run with::

    uv run python scripts/run_so101_policy.py
"""

from __future__ import annotations

import logging
import sys
import time

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
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

POLICY_PATH = "outputs/train/place_green_duck/checkpoints/last/pretrained_model"
TASK = "Remove orange duck from target"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower"

CAMERA_NAME = "top"
CAMERA_PATH = "/dev/video0"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

FPS = 30
DURATION_S = 30.0

DISPLAY_DATA = False

# Stop early once the policy has moved away from the startup pose and then
# returned to it. Values are in degrees for the SO-101 default config.
MIN_RUNTIME_BEFORE_NEUTRAL_STOP_S = 10.0
LEAVE_NEUTRAL_TOLERANCE_DEG = 10.0
RETURN_NEUTRAL_TOLERANCE_DEG = 10.0
NEUTRAL_HOLD_S = 0.2

logger = logging.getLogger("run_so101_policy")


# --------------------------------------------------------------------------- #
# Neutral stop strategy                                                       #
# --------------------------------------------------------------------------- #


class StopAtNeutralStrategy(BaseStrategy):
    """Base rollout that stops after returning to the startup joint pose."""

    def run(self, ctx) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        neutral_pose = ctx.hardware.initial_position
        if not neutral_pose:
            logger.warning("No initial joint pose captured; falling back to duration-only rollout.")

        control_interval = interpolator.get_control_interval(cfg.fps)
        start_time = time.perf_counter()
        neutral_since: float | None = None
        left_neutral = False

        engine.resume()
        logger.info(
            "Neutral-stop rollout started (leave>%.1f deg, return<%.1f deg for %.1fs)",
            LEAVE_NEUTRAL_TOLERANCE_DEG,
            RETURN_NEUTRAL_TOLERANCE_DEG,
            NEUTRAL_HOLD_S,
        )

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            elapsed_s = loop_start - start_time

            if cfg.duration > 0 and elapsed_s >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if neutral_pose:
                neutral_error = _max_neutral_error(obs, neutral_pose)
                if not left_neutral and neutral_error > LEAVE_NEUTRAL_TOLERANCE_DEG:
                    left_neutral = True
                    logger.info("Robot left neutral pose (max joint error %.1f deg)", neutral_error)

                can_stop = left_neutral and elapsed_s >= MIN_RUNTIME_BEFORE_NEUTRAL_STOP_S
                if can_stop and neutral_error < RETURN_NEUTRAL_TOLERANCE_DEG:
                    neutral_since = neutral_since or loop_start
                    if loop_start - neutral_since >= NEUTRAL_HOLD_S:
                        logger.info(
                            "Robot returned to neutral pose for %.1fs (max joint error %.1f deg); stopping.",
                            NEUTRAL_HOLD_S,
                            neutral_error,
                        )
                        break
                else:
                    neutral_since = None

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "Rollout loop is running slower (%.1f Hz) than the target FPS (%s Hz).",
                    1 / dt,
                    cfg.fps,
                )


def _max_neutral_error(obs: dict, neutral_pose: dict) -> float:
    errors = [
        abs(float(obs[key]) - float(neutral_value))
        for key, neutral_value in neutral_pose.items()
        if key in obs
    ]
    return max(errors, default=0.0)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    init_logging()
    logger.info("=" * 72)
    logger.info("SO-101 ACT policy rollout")
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

        logger.info("Initialising Rerun viewer (session='run_so101_policy')")
        init_rerun(session_name="run_so101_policy")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)

    logger.info("Building rollout context (loads policy weights + connects robot)...")
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    logger.info(
        "Context ready. Robot '%s' connected, %d action keys: %s",
        ctx.hardware.robot_wrapper.inner.name,
        len(ctx.data.ordered_action_keys),
        ctx.data.ordered_action_keys,
    )

    strategy = StopAtNeutralStrategy(cfg.strategy)
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
