#!/usr/bin/env python
"""Interactive 16:9-locked camera crop authoring tool.

Opens ``/dev/video0`` via V4L2 / MJPG at ``1280x720 @ 30 fps`` and lets you
position a 16:9 crop rectangle over the workspace. The cropped region is
resized back to ``1280x720`` and shown in a second window so you can see
exactly what the policy will receive at both data-collection and inference
time.

Press ``q`` to atomically write the resulting crop to::

    /home/kuphdev/lerobot/camera_crop.json

The JSON shape is consumed by ``OpenCVCamera`` in
``src/lerobot/cameras/opencv/camera_opencv.py`` so the same crop is applied
deterministically across ``lerobot-record``, ``lerobot-train`` (any policy),
``lerobot-eval``, and local rollout scripts like
``scripts/run_so101_policy.py``.

Run with::

    uv run python scripts/preview_camera_crop.py
    uv run python scripts/preview_camera_crop.py --device /dev/video2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


import cv2


def _configure_opencv_qt_fonts() -> None:
    """Point Qt at system TrueType fonts when the OpenCV wheel lacks cv2/qt/fonts (4.13+)."""
    current = os.environ.get("QT_QPA_FONTDIR", "")
    if current and os.path.isdir(current):
        if any(name.endswith((".ttf", ".otf")) for name in os.listdir(current)):
            return

    for path in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts/TTF",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
    ):
        if os.path.isdir(path) and any(name.endswith((".ttf", ".otf")) for name in os.listdir(path)):
            os.environ["QT_QPA_FONTDIR"] = path
            return


_configure_opencv_qt_fonts()

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
CAPTURE_FOURCC = "MJPG"
ASPECT_W = 16
ASPECT_H = 9

CROP_CONFIG_PATH = Path("/home/kuphdev/lerobot/camera_crop.json")

WINDOW_FULL = "camera (full)"
WINDOW_MODEL = "model view (1280x720)"

MOVE_STEP_FINE = 10
MOVE_STEP_COARSE = 50
ZOOM_STEP_COARSE = 32
ZOOM_STEP_FINE = 8
MIN_CROP_WIDTH = 160


def height_for_width(w: int) -> int:
    """Return the 16:9 height for a given width, rounded to int."""
    return int(round(w * ASPECT_H / ASPECT_W))


def clamp_crop(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Clamp (x, y, w, h) inside (frame_w, frame_h) while preserving 16:9 by trimming size first."""
    w = max(MIN_CROP_WIDTH, min(int(w), frame_w))
    h = height_for_width(w)
    if h > frame_h:
        h = frame_h
        w = int(round(h * ASPECT_W / ASPECT_H))
    x = max(0, min(int(x), frame_w - w))
    y = max(0, min(int(y), frame_h - h))
    return x, y, w, h


def recenter_zoom(
    old_x: int, old_y: int, old_w: int, old_h: int, new_w: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Resize crop to ``new_w`` (16:9) keeping the previous center, then clamp."""
    new_h = height_for_width(new_w)
    cx = old_x + old_w // 2
    cy = old_y + old_h // 2
    nx = cx - new_w // 2
    ny = cy - new_h // 2
    return clamp_crop(nx, ny, new_w, new_h, frame_w, frame_h)


def load_initial_crop(frame_w: int, frame_h: int) -> tuple[tuple[int, int, int, int], bool]:
    """Load existing crop from CROP_CONFIG_PATH, defaulting to full frame.

    Returns ((x, y, w, h), enabled).
    """
    full = (0, 0, frame_w, frame_h)
    if not CROP_CONFIG_PATH.exists():
        return full, False
    try:
        data = json.loads(CROP_CONFIG_PATH.read_text())
        x = int(data["x"])
        y = int(data["y"])
        w = int(data["w"])
        h = int(data["h"])
        enabled = bool(data.get("enabled", False))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
        print(f"[warn] Could not parse existing {CROP_CONFIG_PATH}: {err}. Defaulting to full frame.")
        return full, False

    if h != height_for_width(w):
        print(f"[warn] Existing crop {w}x{h} is not 16:9. Snapping height to {height_for_width(w)}.")
    return clamp_crop(x, y, w, h, frame_w, frame_h), enabled


def open_camera(device: str) -> cv2.VideoCapture:
    """Open the camera with V4L2 / MJPG at the canonical capture mode."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {device} with cv2.CAP_V4L2.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    print(
        f"[info] {device}: requested {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}@{CAPTURE_FPS} "
        f"{CAPTURE_FOURCC} -> actual {actual_w}x{actual_h}@{actual_fps:.1f} {actual_fourcc!r}"
    )
    if (actual_w, actual_h) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        print(
            "[warn] V4L2 driver snapped to a different resolution. "
            "The crop coordinates will be authored against this actual frame size."
        )
    return cap


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` atomically via tmpfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def draw_hud(
    full_frame,
    crop: tuple[int, int, int, int],
    enabled: bool,
    dirty: bool,
    save_path: Path,
) -> None:
    """Draw the crop rectangle and status text onto ``full_frame`` in place."""
    x, y, w, h = crop
    color = (0, 255, 0) if enabled else (0, 165, 255)
    cv2.rectangle(full_frame, (x, y), (x + w, y + h), color, 2)

    lines = [
        f"crop x={x} y={y} w={w} h={h}  (output {CAPTURE_WIDTH}x{CAPTURE_HEIGHT})",
        f"enabled={enabled}  dirty={'yes' if dirty else 'no'}",
        f"save -> {save_path}",
        "[wasd] move 10  [WASD] move 50  [+/-] zoom 32  [=/_] zoom 8",
        "[r] reset  [p] toggle enabled  [q] save+quit  [ESC] quit",
    ]
    for i, line in enumerate(lines):
        org = (10, 24 + i * 22)
        # Hershey vector fonts (LINE_8) avoid Qt's missing cv2/qt/fonts directory on OpenCV 4.13+.
        cv2.putText(full_frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(full_frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/video0", help="V4L2 device path (default: /dev/video0)")
    args = parser.parse_args()

    print(f"[info] Save path: {CROP_CONFIG_PATH}")

    cap = open_camera(args.device)
    try:
        ok, probe = cap.read()
        if not ok or probe is None:
            raise RuntimeError(f"Initial read from {args.device} failed.")
        frame_h, frame_w = probe.shape[:2]

        crop, enabled = load_initial_crop(frame_w, frame_h)
        original_state = (crop, enabled)

        cv2.namedWindow(WINDOW_FULL, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WINDOW_MODEL, cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] Frame read failed; retrying...")
                continue

            x, y, w, h = crop
            x, y, w, h = clamp_crop(x, y, w, h, frame_w, frame_h)
            crop = (x, y, w, h)

            cropped = frame[y : y + h, x : x + w]
            if enabled:
                model_view = cv2.resize(
                    cropped, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_LINEAR
                )
            else:
                model_view = cv2.resize(
                    frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_LINEAR
                )

            display = frame.copy()
            dirty = (crop, enabled) != original_state
            draw_hud(display, crop, enabled, dirty, CROP_CONFIG_PATH)

            cv2.imshow(WINDOW_FULL, display)
            cv2.imshow(WINDOW_MODEL, model_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue

            if key == ord("q"):
                payload = {"enabled": bool(enabled), "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                atomic_write_json(CROP_CONFIG_PATH, payload)
                print(f"[info] Wrote {CROP_CONFIG_PATH}: {payload}")
                return 0

            if key == 27:
                if dirty:
                    print("[info] Unsaved changes; press ESC again to quit without saving, any other key to cancel.")
                    confirm = cv2.waitKey(0) & 0xFF
                    if confirm != 27:
                        continue
                print("[info] Quit without saving.")
                return 0

            if key == ord("r"):
                crop = (0, 0, frame_w, frame_h)
                continue

            if key == ord("p"):
                enabled = not enabled
                continue

            if key in (ord("a"), ord("d"), ord("w"), ord("s"), ord("A"), ord("D"), ord("W"), ord("S")):
                step = MOVE_STEP_COARSE if key < ord("a") else MOVE_STEP_FINE
                kc = chr(key).lower()
                if kc == "a":
                    x -= step
                elif kc == "d":
                    x += step
                elif kc == "w":
                    y -= step
                elif kc == "s":
                    y += step
                crop = clamp_crop(x, y, w, h, frame_w, frame_h)
                continue

            if key in (ord("+"), ord("=")):
                step = ZOOM_STEP_COARSE if key == ord("+") else ZOOM_STEP_FINE
                crop = recenter_zoom(x, y, w, h, w - step, frame_w, frame_h)
                continue

            if key in (ord("-"), ord("_")):
                step = ZOOM_STEP_COARSE if key == ord("-") else ZOOM_STEP_FINE
                crop = recenter_zoom(x, y, w, h, w + step, frame_w, frame_h)
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
