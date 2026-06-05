"""Bounded, context-aware public reply text with deterministic fallbacks.

The robot posts a few categories of short public replies (acknowledge a queued
command, explain an invalid command, etc.). This module optionally uses the LLM
to make them short/fun/varied, but EVERY category has a deterministic fallback
used when the LLM is disabled, errors, times out, or returns something unusable.

Important: generated text is purely cosmetic. It NEVER influences policy
selection, validation, or control flow — those come only from validated enums.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from openai import OpenAI

from config import AppConfig, RequestedColor

logger = logging.getLogger("tweet_robot.reply_generator")

# Keep replies comfortably under the 280-char limit, accounting for the implicit
# @handle prefix X adds on replies.
_MAX_REPLY_CHARS = 220


class ReplyCategory(str, Enum):
    ACCEPTED = "accepted"
    INVALID = "invalid"
    DUPLICATE_AUTHOR = "duplicate_author"
    QUEUE_FULL = "queue_full"


# Categories that may be LLM-enhanced (the playful ones). Error/admin replies are
# always deterministic and precise, so they are handled by their own helpers.
_LLM_ENHANCED = {
    ReplyCategory.ACCEPTED,
    ReplyCategory.INVALID,
    ReplyCategory.DUPLICATE_AUTHOR,
    ReplyCategory.QUEUE_FULL,
}

_INTENT = {
    ReplyCategory.ACCEPTED: (
        "Acknowledge that the user's request was accepted and queued. Mention the "
        "duck color and their queue position. Upbeat and brief."
    ),
    ReplyCategory.INVALID: (
        "Politely say you couldn't understand the command and that they should ask "
        "for one of: orange, green, yellow, or pink."
    ),
    ReplyCategory.DUPLICATE_AUTHOR: (
        "Kindly tell the user they already have a command in the queue and to wait "
        "for it to finish before sending another. One at a time."
    ),
    ReplyCategory.QUEUE_FULL: (
        "Apologize that the queue is full right now and ask them to try again shortly."
    ),
}


def _clean(text: str) -> str:
    """Collapse whitespace and hard-cap length."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) > _MAX_REPLY_CHARS:
        text = text[: _MAX_REPLY_CHARS - 1].rstrip() + "\u2026"
    return text


class ReplyGenerator:
    def __init__(self, cfg: AppConfig, client: OpenAI | None = None):
        self.cfg = cfg
        if client is not None:
            self._client: OpenAI | None = client
        elif cfg.openai_api_key:
            self._client = OpenAI(api_key=cfg.openai_api_key)
        else:
            self._client = None

    # ------------------------------------------------------------------ #
    # Deterministic fallbacks                                            #
    # ------------------------------------------------------------------ #

    def _fallback(
        self,
        category: ReplyCategory,
        *,
        color: RequestedColor | None,
        position: int | None,
    ) -> str:
        color_str = color.value if color is not None else "requested"
        if category is ReplyCategory.ACCEPTED:
            pos = f"#{position}" if position is not None else "in line"
            return f"Queued! You're {pos}. I'll place the {color_str} duck soon."
        if category is ReplyCategory.INVALID:
            return (
                "Sorry, I couldn't figure out the command. "
                "Please ask for orange, green, yellow, or pink."
            )
        if category is ReplyCategory.DUPLICATE_AUTHOR:
            return (
                "You already have a command in the queue — hang tight and I'll get "
                "to it. One request at a time, please!"
            )
        if category is ReplyCategory.QUEUE_FULL:
            return "I'm backed up and the queue is full right now. Please try again in a minute!"
        return "Got it!"

    # ------------------------------------------------------------------ #
    # Public reply generation                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        category: ReplyCategory,
        *,
        author_name: str = "",
        original_text: str = "",
        color: RequestedColor | None = None,
        position: int | None = None,
    ) -> str:
        """Return reply text for ``category`` (never raises)."""
        fallback = self._fallback(category, color=color, position=position)
        if category not in _LLM_ENHANCED or self._client is None:
            return fallback

        try:
            enhanced = self._enhance(category, author_name, original_text, color, position)
        except Exception:  # noqa: BLE001
            logger.exception("Reply LLM enhancement failed; using fallback.")
            return fallback

        enhanced = _clean(enhanced)
        return enhanced or fallback

    def _enhance(
        self,
        category: ReplyCategory,
        author_name: str,
        original_text: str,
        color: RequestedColor | None,
        position: int | None,
    ) -> str:
        ctx_lines = [f"Intent: {_INTENT[category]}"]
        if author_name:
            ctx_lines.append(f"Requester name: {author_name}")
        if original_text:
            ctx_lines.append(f"Their message: {original_text[:200]}")
        if color is not None:
            ctx_lines.append(f"Requested duck color: {color.value}")
        if position is not None:
            ctx_lines.append(f"Queue position: #{position}")

        system = (
            "You write very short, fun, friendly public replies for a livestreamed "
            "robot arm that places rubber ducks on a target. One sentence, under "
            f"{_MAX_REPLY_CHARS} characters. No hashtags. Do not invent facts or "
            "promise specific timing beyond 'soon'. Do not include @mentions. "
            "Return only the reply text."
        )
        resp = self._client.chat.completions.create(
            model=self.cfg.openai_command_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(ctx_lines)},
            ],
            temperature=0.9,
            max_tokens=80,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    # Deterministic, precise replies (no LLM)                           #
    # ------------------------------------------------------------------ #

    def error_user_reply(self) -> str:
        """Reply shown to a user who sends a command while the system is broken."""
        return f"Something is broken. Please tag @{self.cfg.admin_handle}"

    def error_admin_notify(self, detail: str) -> str:
        """Critical error notification tagging the admin."""
        return _clean(f"@{self.cfg.admin_handle} the robot hit an error: {detail}")

    def admin_status_reply(self, status_text: str) -> str:
        return _clean(status_text)
