#!/usr/bin/env python
"""Compare cropped workspace frames after different OpenCVCamera warmups.

This uses the same patched ``OpenCVCamera`` path as the tweet robot, so
``camera_crop.json`` is applied by LeRobot's OpenCV camera implementation.
Each capture also mirrors ``RobotPolicyRunner.capture_workspace_image()``:
``cam.connect(warmup=True)`` followed by 5 synchronous flush reads.

Examples:

    python scripts/compare_camera_warmup.py
    python scripts/compare_camera_warmup.py --warmups 0.25 0.5 1.0 --show

For each warmup value, press Enter to open the camera fresh. The script configures
``OpenCVCameraConfig(warmup_s=<value>)``, calls ``connect(warmup=True)``, performs
the same flush reads as the tweet robot, writes the final frame to
``runtime/camera_warmup_tests``, and optionally displays it until you press a key.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig  # noqa: E402

_ROBOT_FLUSH_READS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="OpenCV camera path/index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--fourcc", default="MJPG", help="Requested FOURCC.")
    parser.add_argument(
        "--warmups",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="OpenCVCameraConfig warmup_s values to compare.",
    )
    parser.add_argument(
        "--flush-reads",
        type=int,
        default=_ROBOT_FLUSH_READS,
        help="Number of synchronous reads after connect; default mirrors tweet_robot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "runtime" / "camera_warmup_tests",
        help="Directory for saved comparison images.",
    )
    parser.add_argument("--show", action="store_true", help="Display each captured frame.")
    return parser.parse_args()


def capture_with_warmup(args: argparse.Namespace, warmup_s: float, index: int) -> Path:
    cam = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
            warmup_s=warmup_s,
        )
    )

    print(
        f"[info] Opening {args.device}; connect(warmup=True, warmup_s={warmup_s:.2f}), "
        f"then {args.flush_reads} flush read(s)..."
    )
    started = time.perf_counter()
    cam.connect(warmup=True)
    try:
        frame = None
        for _ in range(max(1, args.flush_reads)):
            frame = cam.read()
        elapsed = time.perf_counter() - started
    finally:
        cam.disconnect()

    if frame is None:
        raise RuntimeError("camera returned no frame")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"warmup_{index:02d}_{warmup_s:.2f}s_{stamp}.png"
    # OpenCVCamera returns RGB by default; cv2.imwrite expects BGR.
    cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[ok] Saved {out_path} (elapsed {elapsed:.2f}s)")

    if args.show:
        cv2.imshow(f"warmup {warmup_s:.2f}s", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("[info] Press any key in the image window to continue.")
        cv2.waitKey(0)
        cv2.destroyWindow(f"warmup {warmup_s:.2f}s")

    return out_path


def main() -> int:
    args = parse_args()
    print("This test opens the camera fresh for each warmup value.")
    print("Capture path mirrors tweet_robot: connect(warmup=True) + flush reads.")
    print("Close other camera users before running (preview scripts, robot process, etc.).")
    print(f"Output directory: {args.output_dir}")

    saved: list[Path] = []
    for idx, warmup_s in enumerate(args.warmups, start=1):
        input(f"\nPress Enter to capture with warmup_s={warmup_s:.2f}...")
        saved.append(capture_with_warmup(args, warmup_s, idx))

    print("\nSaved comparison frames:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
