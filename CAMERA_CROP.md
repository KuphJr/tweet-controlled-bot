# Camera Crop — Personal Reference

Deterministic 16:9 camera crop shared across data collection and policy inference for the SO-101 setup.

The crop is authored once with an interactive script, saved to a JSON file, and applied automatically by the OpenCV camera at runtime — so `lerobot-record`, `lerobot-train`, `lerobot-eval`, and `scripts/run_so101_policy.py` all see the exact same cropped image distribution.

---

## Why

To focus the policy on the workspace and ignore noisy edges of the camera frame, while:
- Preserving 16:9 aspect ratio (no stretching/warping).
- Keeping the dataset/output shape at the configured `1280x720` so no other LeRobot config has to change.
- Guaranteeing record-time and rollout-time camera frames match byte-for-byte modulo sensor noise.

---

## Files

- [`scripts/preview_camera_crop.py`](scripts/preview_camera_crop.py) — interactive crop authoring tool.
- [`src/lerobot/cameras/opencv/camera_opencv.py`](src/lerobot/cameras/opencv/camera_opencv.py) — applies the crop on every frame from the camera read thread.
- `/home/kuphdev/lerobot/camera_crop.json` — the saved crop. **Hard-coded path.**

---

## Quick start

```bash
# 1. Author / re-author the crop
uv run python scripts/preview_camera_crop.py
# (camera defaults to /dev/video0 — pass --device /dev/videoN to override)

# 2. Confirm the file
cat /home/kuphdev/lerobot/camera_crop.json
# -> {"enabled": true, "x": 96, "y": 54, "w": 1088, "h": 612}

# 3. Use it (no changes to your existing commands)
lerobot-record --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30, fourcc: 'MJPG'}}" ...
uv run python scripts/run_so101_policy.py
```

On startup you'll see one info log line per camera connection:

```
camera_crop.json loaded from /home/kuphdev/lerobot/camera_crop.json: x=96 y=54 w=1088 h=612 (crop will be resized back to camera capture size).
```

If you ever don't see this line, the crop is **not** being applied — see Troubleshooting.

---

## JSON format

```json
{
  "enabled": true,
  "x": 96,
  "y": 54,
  "w": 1088,
  "h": 612
}
```

- All values are **integer pixels in the raw 1280x720 sensor frame**.
- `(w, h)` is expected to be 16:9 — the preview script enforces this; the camera trusts whatever's in the file.
- `enabled: false` makes the camera behave as if no crop file existed.

---

## Preview script controls

Two windows open: the full 1280x720 frame with a green/orange crop rectangle and HUD, and a "model view" showing exactly what the policy will receive (1280x720 after crop+resize).

| Key       | Action                                        |
| --------- | --------------------------------------------- |
| `w/a/s/d` | Move crop by 10 px                            |
| `W/A/S/D` | Move crop by 50 px (Shift)                    |
| `+`       | Zoom in by 32 px width (16:9 + recenter)      |
| `=`       | Zoom in by 8 px width (fine)                  |
| `-`       | Zoom out by 32 px width                       |
| `_`       | Zoom out by 8 px width (fine)                 |
| `r`       | Reset to full frame (1280x720)                |
| `p`       | Toggle `enabled` in memory (preview only)     |
| `q`       | **Save** (atomic write) and quit              |
| `ESC`     | Quit without saving (asks to confirm if dirty)|

Notes:
- Aspect ratio is locked 16:9; `w` is the source of truth and `h` is computed.
- Zoom keeps the crop's center stable and clamps to the frame.
- Save is atomic (`tmpfile + os.replace`), so a Ctrl-C mid-save won't corrupt the file.

---

## How the crop reaches both record and rollout

```
V4L2 1280x720 MJPG
  -> OpenCVCamera._read_from_hardware()
       crop[y:y+h, x:x+w]
       resize back to 1280x720
  -> OpenCVCamera._postprocess_image() (BGR->RGB, dim assertion)
  -> latest_frame
  -> SOFollower.get_observation()  <-- consumed identically by:
       - lerobot-record  (writes 1280x720 frames into the dataset)
       - lerobot-train   (any policy: ACT, SmolVLA, ...) on that dataset
       - lerobot-eval / scripts/run_so101_policy.py
```

The crop is loaded once per `OpenCVCamera` instance (in `__init__`), so the info log line appears exactly once on startup. The robot's declared image feature shape stays `(720, 1280, 3)` because it's derived from the camera config, not the actual frame data.

---

## Disable / revert

| What                                | How                                              |
| ----------------------------------- | ------------------------------------------------ |
| Pause crop temporarily              | Set `"enabled": false` in `camera_crop.json`     |
| Permanently revert to no crop       | `rm /home/kuphdev/lerobot/camera_crop.json`      |
| Re-author from scratch              | `uv run python scripts/preview_camera_crop.py`   |

In the disabled / missing case, the camera logs `... no crop applied.` once at startup and behaves exactly as upstream LeRobot.

---

## Troubleshooting

- **"camera_crop.json not found ..." log line, but file exists.** The path is hard-coded to `/home/kuphdev/lerobot/camera_crop.json`. Make sure that's where the file is, not next to the scripts.
- **"... is malformed (...)" log line.** JSON parse error or missing key. Re-author with the preview script (it always emits a complete, valid file).
- **Recorded frames look uncropped.** Check the startup logs of `lerobot-record` for the `camera_crop.json loaded ...` line. If you see `enabled=false` instead, flip it. If you see `not found`, the file path is wrong.
- **Policy runs at inference look different from training data.** The same JSON must exist at training-data-collection time *and* at rollout time. If you change/delete it between record and rollout you'll have a distribution mismatch. The single source of truth is one JSON file — keep it stable.
- **Preview script shows "V4L2 driver snapped to a different resolution" warning.** Your camera doesn't actually support 1280x720 MJPG @ 30 fps. Check `v4l2-ctl --device=/dev/video0 --list-formats-ext` and adjust the constants at the top of [`scripts/preview_camera_crop.py`](scripts/preview_camera_crop.py).
- **Image looks stretched/warped in the model view.** The crop in the JSON isn't 16:9. Re-author with the preview script.

---

## When to re-author

- Camera physically moves on the mount.
- You change camera mode/resolution (e.g. switch to 1080p capture — would also need code changes; see chat history for trade-offs).
- You add a wrist camera (currently only the `top` camera is cropped — wrist would be uncropped).
- You change workspace layout enough that the existing crop now misses important pixels.
