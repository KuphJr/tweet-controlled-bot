"""OpenAI-vision workspace-state detection.

Classifies a top-down camera frame of the target zone into one of:
``orange_on_target``, ``green_on_target``, ``yellow_on_target``,
``pink_on_target``, ``empty``, or ``error``. Returns strict, validated enum
values (:class:`WorkspaceState` / :class:`RequestedColor`). The LLM classifies;
deterministic controller code decides what to do with each state.

Optional few-shot examples live in ``vision_examples/<descriptive folder>/*.png``.
Folder names are human-readable and mapped to a canonical ``WorkspaceState`` by a
small resolver (e.g. ``flipped over duck error`` -> ``error``). The descriptive
name is also handed to the model as the example's natural-language label.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from config import (
    AppConfig,
    RequestedColor,
    WorkspaceDetection,
    WorkspaceState,
    color_on_target_state,
)

logger = logging.getLogger("tweet_robot.workspace_detector")

_ALLOWED_STATES = [s.value for s in WorkspaceState]
_ALLOWED_COLORS = [c.value for c in RequestedColor]

# Cap example count and downscale to keep per-call token cost/latency reasonable.
# Cap is per FOLDER (not per state) so multiple error folders (e.g. "flipped over"
# and "misplaced") each contribute variety.
_MAX_EXAMPLES_PER_FOLDER = 3
_EXAMPLE_MAX_WIDTH = 512
_QUERY_MAX_WIDTH = 768
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

_SYSTEM_PROMPT = (
    "You are a precise visual inspector for a robot arm workspace, viewed from a "
    "top-down camera. In the center is a round red-and-white bullseye "
    "target. Rubber ducks come in four colors: orange, green, yellow, and pink. "
    "Determine the state of the TARGET (the bullseye):\n"
    "- 'orange_on_target' / 'green_on_target' / 'yellow_on_target' / "
    "'pink_on_target': exactly one duck of that color sits near the center of the target.\n"
    "- 'empty': the target/bullseye is clear with no duck on it (ducks elsewhere "
    "in the scene are fine and should be ignored).\n"
    "- 'error': anything abnormal — a duck is very significantly off-center / only partly on the "
    "target, dropped or misplaced, flipped over, multiple ducks on the target, or "
    "the scene is otherwise unreadable/unknown.\n"
    "Note: it is easy to confuse the yellow and orange ducks because they are similar colors, "
    "so be sure to carefully review the image and the example images. "
    "Another clue to distinguish the color that is on the target is that one "
    "color will be on the target and the other color will be off of the target.\n"
    "Set current_color to the duck color on the target, or null if empty/error. "
    "Set is_error=true only for the 'error' state. Provide confidence in [0,1] and "
    "a short description. Respond ONLY with the structured JSON."
)

_JSON_SCHEMA = {
    "name": "workspace_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": _ALLOWED_STATES},
            "current_color": {"type": ["string", "null"], "enum": [*_ALLOWED_COLORS, None]},
            "confidence": {"type": "number"},
            "is_error": {"type": "boolean"},
            "error_reason": {"type": ["string", "null"]},
            "description": {"type": "string"},
        },
        "required": ["state", "current_color", "confidence", "is_error", "error_reason", "description"],
        "additionalProperties": False,
    },
}


def _folder_to_state(folder_name: str) -> WorkspaceState | None:
    """Map a descriptive example folder name to a canonical WorkspaceState."""
    name = folder_name.lower()
    if "error" in name:
        return WorkspaceState.ERROR
    if "empty" in name:
        return WorkspaceState.EMPTY
    for color in RequestedColor:
        if color.value in name:
            return color_on_target_state(color)
    return None


def _encode_image_array(image_rgb: np.ndarray, max_width: int) -> str:
    """Downscale (if needed) and JPEG-encode an RGB array to a data URL."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("failed to JPEG-encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_image_file(path: Path, max_width: int) -> str | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        logger.warning("Could not read example image %s; skipping.", path)
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _encode_image_array(rgb, max_width)


class _Example:
    __slots__ = ("state", "label", "data_url")

    def __init__(self, state: WorkspaceState, label: str, data_url: str):
        self.state = state
        self.label = label
        self.data_url = data_url


class WorkspaceDetector:
    def __init__(self, cfg: AppConfig, client: OpenAI | None = None):
        self.cfg = cfg
        if client is not None:
            self._client: OpenAI | None = client
        elif cfg.openai_api_key:
            self._client = OpenAI(api_key=cfg.openai_api_key)
        else:
            self._client = None
            logger.warning("OPENAI_API_KEY not set; workspace detection will report errors.")
        self._examples: list[_Example] = self._load_examples()

    # ------------------------------------------------------------------ #
    # Few-shot examples                                                  #
    # ------------------------------------------------------------------ #

    def _load_examples(self) -> list[_Example]:
        examples_dir = self.cfg.vision_examples_dir
        if not examples_dir.exists():
            logger.info("No vision_examples directory at %s; running without few-shot.", examples_dir)
            return []

        examples: list[_Example] = []
        states_seen: set[WorkspaceState] = set()
        for folder in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
            state = _folder_to_state(folder.name)
            if state is None:
                logger.warning("Vision example folder '%s' did not map to a state; skipping.", folder.name)
                continue
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            taken = 0
            for img_path in images:
                if taken >= _MAX_EXAMPLES_PER_FOLDER:
                    break
                data_url = _encode_image_file(img_path, _EXAMPLE_MAX_WIDTH)
                if data_url is None:
                    continue
                label = (
                    f"Reference example — this image shows: '{folder.name}'. "
                    f"Correct state = '{state.value}'."
                )
                examples.append(_Example(state, label, data_url))
                states_seen.add(state)
                taken += 1

        if examples:
            logger.info(
                "Loaded %d vision few-shot examples across %d states.",
                len(examples),
                len(states_seen),
            )
        return examples

    # ------------------------------------------------------------------ #
    # Detection                                                          #
    # ------------------------------------------------------------------ #

    def detect_state(self, capture_fn) -> WorkspaceDetection:
        """Capture + classify, retrying ONCE on capture failure / error / low confidence.

        ``capture_fn`` is a zero-arg callable returning an RGB ``np.ndarray``
        (e.g. ``RobotPolicyRunner.capture_workspace_image``); it may raise.
        """
        first = self._capture_and_detect(capture_fn)
        if not first.is_error and first.confidence >= self.cfg.vision_confidence_threshold:
            return first

        logger.info(
            "Detection inconclusive (state=%s, is_error=%s, conf=%.2f); retrying once...",
            first.state.value,
            first.is_error,
            first.confidence,
        )
        time.sleep(0.4)
        second = self._capture_and_detect(capture_fn)
        if not second.is_error and second.confidence >= self.cfg.vision_confidence_threshold:
            return second

        # Both inconclusive: prefer a non-error reading, otherwise the more confident one.
        if not second.is_error and first.is_error:
            return second
        if not first.is_error and second.is_error:
            return first
        return second if second.confidence >= first.confidence else first

    def _capture_and_detect(self, capture_fn) -> WorkspaceDetection:
        try:
            image = capture_fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workspace image capture failed.")
            return WorkspaceDetection(
                state=WorkspaceState.ERROR,
                current_color=None,
                confidence=0.0,
                is_error=True,
                error_reason=f"camera capture failed: {exc}",
                description="camera capture failed",
            )
        return self.detect_image(image)

    def detect_image(self, image_rgb: np.ndarray) -> WorkspaceDetection:
        """Classify a single already-captured RGB frame (never raises)."""
        if self._client is None:
            return WorkspaceDetection(
                WorkspaceState.ERROR, None, 0.0, True, "no OpenAI key", "detector unavailable"
            )
        try:
            query_url = _encode_image_array(image_rgb, _QUERY_MAX_WIDTH)
            raw = self._call_llm(query_url)
            return self._validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vision detection failed.")
            return WorkspaceDetection(
                WorkspaceState.ERROR, None, 0.0, True, f"vision_error: {exc}", "vision call failed"
            )

    def _call_llm(self, query_url: str) -> dict:
        content: list[dict] = []
        if self._examples:
            content.append({"type": "text", "text": "Here are labeled reference examples:"})
            for ex in self._examples:
                content.append({"type": "text", "text": ex.label})
                # "low" detail keeps example token cost flat; the labels carry the signal.
                content.append({"type": "image_url", "image_url": {"url": ex.data_url, "detail": "low"}})
        content.append({"type": "text", "text": "Now classify THIS image (the current workspace):"})
        content.append({"type": "image_url", "image_url": {"url": query_url, "detail": "high"}})

        resp = self._client.chat.completions.create(
            model=self.cfg.openai_vision_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
        )
        return json.loads(resp.choices[0].message.content or "{}")

    def _validate(self, raw: dict) -> WorkspaceDetection:
        try:
            state = WorkspaceState(str(raw.get("state", "error")))
        except ValueError:
            state = WorkspaceState.ERROR

        color_raw = raw.get("current_color")
        current_color: RequestedColor | None = None
        if isinstance(color_raw, str):
            try:
                current_color = RequestedColor(color_raw.strip().lower())
            except ValueError:
                current_color = None

        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        is_error = bool(raw.get("is_error", False)) or state is WorkspaceState.ERROR
        error_reason = raw.get("error_reason")
        if error_reason is not None:
            error_reason = str(error_reason)
        description = str(raw.get("description", ""))

        # Reconcile state/color so downstream code can trust the on_target_color helper.
        expected_color = state.on_target_color
        if expected_color is not None:
            current_color = expected_color
        elif state in (WorkspaceState.EMPTY, WorkspaceState.ERROR):
            current_color = None

        return WorkspaceDetection(
            state=state,
            current_color=current_color,
            confidence=confidence,
            is_error=is_error,
            error_reason=error_reason,
            description=description,
        )
