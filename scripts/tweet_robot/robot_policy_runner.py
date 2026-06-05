"""Isolated, reliability-first policy execution for the tweet robot.

Each policy run is a self-contained rollout that mirrors ``run_phase()`` from
``scripts/run_so101_policy_place_then_remove.py``: build a fresh rollout context
(connects the robot + loads weights), run the neutral-stop strategy, then tear
down (return to neutral + disconnect). NO core LeRobot changes and NO persistent
rollout context / inference engine reuse — reliability is prioritized over speed.

Torque policy (servo 3 has a counter-spring / rubber band):
  * Between policies: the robot config uses ``disable_torque_on_disconnect=False``
    so teardown disconnects WITHOUT cutting torque; the arm holds neutral.
  * On shutdown: ``disable_torque_on_shutdown()`` is the single place torque is cut.

Vision frames come from a short-lived standalone ``OpenCVCamera`` (same config as
the robot's ``top`` camera, so the deterministic crop in ``camera_crop.json`` is
applied identically). The robot is disconnected between isolated runs, so the
camera device is free for these grabs.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
from lerobot.rollout.inference import SyncInferenceConfig

from config import AppConfig, PolicyResult

# Make the sibling example script importable (it lives in scripts/, one level up)
# so we can reuse the proven StopAtNeutralStrategy without modifying it.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_so101_policy import StopAtNeutralStrategy  # noqa: E402

logger = logging.getLogger("tweet_robot.policy_runner")

# Classification margin: if the rollout control-loop runtime reaches within this
# many seconds of the duration cap, we treat it as a duration timeout (the policy
# did not return to neutral early). Importantly, this excludes policy loading,
# robot connection, setup, teardown, and disconnect, so slow I/O cannot create
# false "duration" errors when the actual policy returned home successfully.
_DURATION_MARGIN_S = 1.0

# Number of frames to read/flush from a freshly opened camera before trusting one.
_CAMERA_FLUSH_READS = 5


class RobotPolicyRunner:
    """Owns no persistent connection; each method connects/disconnects as needed."""

    def __init__(self, cfg: AppConfig, shutdown_event: Event):
        self.cfg = cfg
        self.shutdown_event = shutdown_event

    # ------------------------------------------------------------------ #
    # Config builders                                                    #
    # ------------------------------------------------------------------ #

    def _camera_config(self) -> dict[str, OpenCVCameraConfig]:
        return {
            self.cfg.camera_name: OpenCVCameraConfig(
                index_or_path=self.cfg.camera_path,
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
                fps=self.cfg.camera_fps,
                fourcc=self.cfg.camera_fourcc,
            )
        }

    def _robot_config(
        self, *, disable_torque_on_disconnect: bool, include_cameras: bool
    ) -> SO101FollowerConfig:
        return SO101FollowerConfig(
            port=self.cfg.robot_port,
            id=self.cfg.robot_id,
            cameras=self._camera_config() if include_cameras else {},
            disable_torque_on_disconnect=disable_torque_on_disconnect,
        )

    # ------------------------------------------------------------------ #
    # Policy execution                                                   #
    # ------------------------------------------------------------------ #

    def run_policy(self, policy_path: str, task: str, timeout_s: int | None = None) -> PolicyResult:
        """Run one isolated neutral-stop rollout.

        Returns a :class:`PolicyResult`. Never raises for hardware/inference
        problems — those are captured into ``stop_reason="error"`` so the
        controller can transition to ERROR cleanly.
        """
        timeout_s = int(timeout_s if timeout_s is not None else self.cfg.policy_timeout_s)

        if self.cfg.no_robot or self.cfg.dry_run:
            mode = "no-robot" if self.cfg.no_robot else "dry-run"
            logger.info("[%s] Skipping policy execution: %s (%s)", mode, task, policy_path)
            return PolicyResult(success=True, stop_reason="skipped", elapsed_s=0.0)

        logger.info("Running policy: task=%r path=%s timeout=%ds", task, policy_path, timeout_s)

        try:
            policy_config = PreTrainedConfig.from_pretrained(policy_path)
            policy_config.pretrained_path = policy_path
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load policy config from %s", policy_path)
            return PolicyResult(
                success=False, stop_reason="error", elapsed_s=0.0, error_message=f"load_policy: {exc}"
            )

        rollout_cfg = RolloutConfig(
            robot=self._robot_config(disable_torque_on_disconnect=False, include_cameras=True),
            policy=policy_config,
            strategy=BaseStrategyConfig(),
            inference=SyncInferenceConfig(),
            fps=float(self.cfg.control_fps),
            duration=float(timeout_s),
            task=task,
            display_data=False,
        )

        error_message: str | None = None
        interrupted = False
        ctx = None
        strategy = None
        rollout_elapsed = 0.0
        try:
            ctx = build_rollout_context(rollout_cfg, self.shutdown_event)
            strategy = StopAtNeutralStrategy(rollout_cfg.strategy)
            strategy.setup(ctx)
            rollout_start = time.perf_counter()
            strategy.run(ctx)
            rollout_elapsed = time.perf_counter() - rollout_start
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Policy interrupted by user (Ctrl-C).")
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            logger.exception("Policy run raised an exception.")
        finally:
            if strategy is not None and ctx is not None:
                try:
                    strategy.teardown(ctx)
                except Exception:  # noqa: BLE001
                    logger.exception("Policy teardown raised an exception (continuing).")

        return self._classify_result(rollout_elapsed, timeout_s, interrupted, error_message)

    def _classify_result(
        self, elapsed: float, timeout_s: int, interrupted: bool, error_message: str | None
    ) -> PolicyResult:
        if error_message is not None:
            return PolicyResult(False, "error", elapsed, error_message)
        if interrupted or self.shutdown_event.is_set():
            return PolicyResult(False, "shutdown", elapsed, "shutdown requested during policy")
        if elapsed >= (timeout_s - _DURATION_MARGIN_S):
            # Hit the duration cap without returning to neutral -> treat as timeout.
            return PolicyResult(
                False, "duration", elapsed, f"policy hit {timeout_s}s duration cap without reaching neutral"
            )
        return PolicyResult(True, "neutral", elapsed, None)

    # ------------------------------------------------------------------ #
    # Vision capture                                                     #
    # ------------------------------------------------------------------ #

    def capture_workspace_image(self) -> np.ndarray:
        """Grab one cropped, RGB top-down frame via a short-lived camera open.

        Raises on failure so the caller (workspace detector) can map it to a
        capture error.
        """
        cam = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=self.cfg.camera_path,
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
                fps=self.cfg.camera_fps,
                fourcc=self.cfg.camera_fourcc,
            )
        )
        cam.connect(warmup=True)
        try:
            frame: np.ndarray | None = None
            for _ in range(_CAMERA_FLUSH_READS):
                frame = cam.read()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            return frame
        finally:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("Camera disconnect raised (continuing).")

    # ------------------------------------------------------------------ #
    # Shutdown                                                           #
    # ------------------------------------------------------------------ #

    def disable_torque_on_shutdown(self) -> None:
        """Cut torque as graceful cleanup. The single place torque is disabled.

        Connects a robot configured with ``disable_torque_on_disconnect=True``
        (no cameras needed) and immediately disconnects, which disables torque
        via the motor bus. Tolerant of the intermittent servo-3 write failure.
        """
        if self.cfg.no_robot or self.cfg.dry_run:
            logger.info("Skipping torque disable (no-robot/dry-run).")
            return

        logger.info("Disabling robot torque for graceful shutdown...")
        try:
            robot = make_robot_from_config(
                self._robot_config(disable_torque_on_disconnect=True, include_cameras=False)
            )
            robot.connect(calibrate=False)
            robot.disconnect()
            logger.info("Robot torque disabled and disconnected.")
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to cleanly disable torque (rubber-band on servo 3 can cause this). "
                "Physically power off the arm to be safe."
            )
