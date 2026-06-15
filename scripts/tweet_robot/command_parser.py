"""LLM command parsing into strict, validated enum values.

Turns free-form tweet/comment text like "put the green duck on the target",
"orange please", "do pink", or "remove the duck" into a
:class:`ParsedCommand` whose ``command_kind`` is a validated enum and whose
``requested_color`` is only present for place commands. The LLM never selects a
policy or code path — it only proposes intent/color + confidence, which
deterministic code here validates and gates.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from config import AppConfig, CommandKind, ParsedCommand, RequestedColor

logger = logging.getLogger("tweet_robot.command_parser")

_ALLOWED_COLORS = [c.value for c in RequestedColor]
_ALLOWED_COMMAND_KINDS = [k.value for k in CommandKind]

_SYSTEM_PROMPT = (
    "You parse short social-media messages directing a robot arm that manages "
    "rubber ducks on a target. There are two valid command kinds:\n"
    "1. place: place one duck color onto the target. Valid colors are orange, "
    "green, yellow, and pink. For place commands, command_kind='place' and "
    "requested_color must be that color.\n"
    "2. remove: remove/clear/take off whatever duck is currently on the target. "
    "For remove commands, command_kind='remove' and requested_color must be null; "
    "the vision system will determine the actual color later.\n"
    "Be tolerant of casual place phrasing (e.g. 'green please', 'do orange', "
    "'pink duck on target', 'move yellow'). Be tolerant of clear remove phrasing "
    "(e.g. 'remove the duck', 'clear the target', 'take the duck off', "
    "'move the duck off the target'). If the user asks to remove/replace one duck "
    "and then place one clear final color (e.g. 'remove yellow and place pink' or "
    "'replace the pink duck with green'), treat it as a place command for that final "
    "color; deterministic robot code will handle removing the current duck first. "
    "If a message is ambiguous, asks for multiple final place colors, asks for an "
    "unsupported arrangement, or is off-topic, set valid=false. Set confidence in [0,1] "
    "reflecting how sure you are."
)

_JSON_SCHEMA = {
    "name": "duck_command",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "command_kind": {
                "type": ["string", "null"],
                "enum": [*_ALLOWED_COMMAND_KINDS, None],
            },
            "requested_color": {
                "type": ["string", "null"],
                "enum": [*_ALLOWED_COLORS, None],
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["valid", "command_kind", "requested_color", "confidence", "reason"],
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
            return ParsedCommand(False, None, None, 0.0, "empty message")
        if self._client is None:
            return ParsedCommand(False, None, None, 0.0, "parser unavailable (no OpenAI key)")

        try:
            raw = self._call_llm(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Command parse LLM call failed.")
            return ParsedCommand(False, None, None, 0.0, f"parser_error: {exc}")

        return self._validate(raw)

    def _call_llm(self, text: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.cfg.openai_command_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
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

        kind_raw = raw.get("command_kind")
        command_kind: CommandKind | None = None
        if isinstance(kind_raw, str):
            try:
                command_kind = CommandKind(kind_raw.strip().lower())
            except ValueError:
                command_kind = None

        color_raw = raw.get("requested_color")
        color: RequestedColor | None = None
        if isinstance(color_raw, str):
            try:
                color = RequestedColor(color_raw.strip().lower())
            except ValueError:
                color = None

        llm_valid = bool(raw.get("valid", False))

        # Deterministic validity: requires a recognized command kind, LLM-claimed
        # validity, confidence above threshold, and the required enum fields for
        # that command. The enum values, not LLM prose, drive policy selection.
        if command_kind is CommandKind.PLACE:
            valid = llm_valid and color is not None and confidence >= self.cfg.command_confidence_threshold
        elif command_kind is CommandKind.REMOVE:
            valid = llm_valid and confidence >= self.cfg.command_confidence_threshold
            color = None
        else:
            valid = False

        if not valid and not reason:
            if command_kind is None:
                reason = "no recognized command kind (place/remove)"
            elif command_kind is CommandKind.PLACE and color is None:
                reason = "no recognized duck color (orange/green/yellow/pink)"
            elif confidence < self.cfg.command_confidence_threshold:
                reason = f"low confidence ({confidence:.2f})"

        # Keep the detected color for logging; the controller only acts on it when valid.
        return ParsedCommand(
            valid=valid,
            command_kind=command_kind,
            requested_color=color,
            confidence=confidence,
            reason=reason,
        )
