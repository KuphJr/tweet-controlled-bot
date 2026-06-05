"""LLM command parsing into strict, validated enum values.

Turns free-form tweet/comment text like "put the green duck on the target",
"orange please", or "do pink" into a :class:`ParsedCommand` whose
``requested_color`` is one of the validated :class:`RequestedColor` enum values
(or ``None``). The LLM never selects a policy or code path — it only proposes a
color + confidence, which deterministic code here validates and gates.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from config import AppConfig, ParsedCommand, RequestedColor

logger = logging.getLogger("tweet_robot.command_parser")

_ALLOWED_COLORS = [c.value for c in RequestedColor]

_SYSTEM_PROMPT = (
    "You parse short social-media messages directing a robot arm to place a "
    "rubber duck of a specific color onto a target. The only valid colors are "
    "orange, green, yellow, and pink. The user is asking for ONE color to be "
    "placed on the target. Extract the requested color if the message clearly "
    "asks for one of the four colors; otherwise mark it invalid. Be tolerant of "
    "casual phrasing (e.g. 'green please', 'do orange', 'pink duck on target', "
    "'move yellow'). If multiple colors or no clear color is requested, or the "
    "message is off-topic/ambiguous, set valid=false and requested_color=null. "
    "Set confidence in [0,1] reflecting how sure you are."
)

_JSON_SCHEMA = {
    "name": "duck_command",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "requested_color": {
                "type": ["string", "null"],
                "enum": [*_ALLOWED_COLORS, None],
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["valid", "requested_color", "confidence", "reason"],
        "additionalProperties": False,
    },
}


class CommandParser:
    def __init__(self, cfg: AppConfig, client: OpenAI | None = None):
        self.cfg = cfg
        if client is not None:
            self._client: OpenAI | None = client
        elif cfg.openai_api_key:
            self._client = OpenAI(api_key=cfg.openai_api_key)
        else:
            self._client = None
            logger.warning("OPENAI_API_KEY not set; command parsing will always return invalid.")

    def parse(self, raw_text: str) -> ParsedCommand:
        """Parse text into a validated :class:`ParsedCommand` (never raises)."""
        text = (raw_text or "").strip()
        if not text:
            return ParsedCommand(False, None, 0.0, "empty message")
        if self._client is None:
            return ParsedCommand(False, None, 0.0, "parser unavailable (no OpenAI key)")

        try:
            raw = self._call_llm(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Command parse LLM call failed.")
            return ParsedCommand(False, None, 0.0, f"parser_error: {exc}")

        return self._validate(raw)

    def _call_llm(self, text: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.cfg.openai_command_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def _validate(self, raw: dict) -> ParsedCommand:
        """Apply deterministic gating on top of the raw LLM output."""
        reason = str(raw.get("reason", ""))
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        color_raw = raw.get("requested_color")
        color: RequestedColor | None = None
        if isinstance(color_raw, str):
            try:
                color = RequestedColor(color_raw.strip().lower())
            except ValueError:
                color = None

        llm_valid = bool(raw.get("valid", False))

        # Deterministic validity: requires a recognized color, LLM-claimed validity,
        # AND confidence above the configured threshold. The enum (not LLM prose)
        # is the source of truth for which policy can later run.
        valid = llm_valid and color is not None and confidence >= self.cfg.command_confidence_threshold

        if not valid and not reason:
            if color is None:
                reason = "no recognized duck color (orange/green/yellow/pink)"
            elif confidence < self.cfg.command_confidence_threshold:
                reason = f"low confidence ({confidence:.2f})"

        # Keep the detected color for logging; the controller only acts on it when valid.
        return ParsedCommand(valid=valid, requested_color=color, confidence=confidence, reason=reason)
