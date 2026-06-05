"""Optional ElevenLabs narration with a configurable, failure-tolerant player.

Synthesizes speech via ElevenLabs to a temp file, then plays it with a
configurable external command (``AUDIO_PLAYER_CMD``, e.g. ``ffplay``/``aplay``/
``paplay``) so it works on Ubuntu without extra Python audio deps. Synthesis +
playback run in a worker thread joined with a timeout, so TTS NEVER blocks policy
execution indefinitely. Any failure (missing key, synth error, missing player,
timeout) is logged and ignored. ``--no-tts`` disables it entirely.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
from threading import Lock, Thread

from config import AppConfig

logger = logging.getLogger("tweet_robot.tts")

_OUTPUT_FORMAT = "mp3_44100_128"


class TTS:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._lock = Lock()
        self._proc: subprocess.Popen | None = None
        self._client = None

        self.enabled = (
            not cfg.no_tts and bool(cfg.elevenlabs_api_key) and bool(cfg.elevenlabs_voice_id)
        )
        if cfg.no_tts:
            logger.info("--no-tts set; TTS disabled (intended speech will be logged).")
        elif not self.enabled:
            logger.warning("ElevenLabs not fully configured; TTS disabled (speech will be logged).")
        else:
            try:
                from elevenlabs.client import ElevenLabs

                self._client = ElevenLabs(api_key=cfg.elevenlabs_api_key)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to init ElevenLabs client; TTS disabled.")
                self.enabled = False

    # ------------------------------------------------------------------ #
    # Pronunciation                                                      #
    # ------------------------------------------------------------------ #

    def _pronounce(self, text: str) -> str:
        """Fix tricky handle pronunciations for the synthesizer."""
        # @KuphDev -> "at KoofDev" (and the configured admin handle generically).
        text = re.sub(r"@KuphDev\b", "at KoofDev", text, flags=re.IGNORECASE)
        admin = self.cfg.admin_handle.lstrip("@")
        if admin and admin.lower() != "kuphdev":
            text = re.sub(rf"@{re.escape(admin)}\b", f"at {admin}", text, flags=re.IGNORECASE)
        # Any remaining @handle -> "at handle" so it isn't read as "at sign".
        text = re.sub(r"@(\w+)", r"at \1", text)
        return text

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def speak(self, text: str, *, timeout_s: float | None = None) -> None:
        """Synthesize + play ``text``, bounded by ``timeout_s`` (never raises)."""
        text = (text or "").strip()
        if not text:
            return
        logger.info("TTS say: %s", text)
        if not self.enabled or self._client is None:
            return

        timeout_s = timeout_s if timeout_s is not None else self.cfg.tts_timeout_s
        spoken = self._pronounce(text)
        worker = Thread(target=self._synth_and_play, args=(spoken,), daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            logger.warning("TTS exceeded %.1fs; continuing (audio may finish in background).", timeout_s)

    def stop(self) -> None:
        """Terminate any in-progress playback (used on shutdown)."""
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to terminate audio player.", exc_info=True)

    # ------------------------------------------------------------------ #
    # Worker                                                             #
    # ------------------------------------------------------------------ #

    def _synth_and_play(self, text: str) -> None:
        path: str | None = None
        try:
            audio = b"".join(
                self._client.text_to_speech.convert(
                    voice_id=self.cfg.elevenlabs_voice_id,
                    model_id=self.cfg.elevenlabs_model_id,
                    text=text,
                    output_format=_OUTPUT_FORMAT,
                )
            )
            if not audio:
                logger.warning("ElevenLabs returned empty audio; skipping playback.")
                return
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio)
                path = f.name
            self._play_file(path)
        except Exception:  # noqa: BLE001
            logger.exception("TTS synth/playback failed (continuing).")
        finally:
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _play_file(self, path: str) -> None:
        cmd = shlex.split(self.cfg.audio_player_cmd) + [path]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.warning("Audio player not found: %r. Set AUDIO_PLAYER_CMD.", self.cfg.audio_player_cmd)
            return
        with self._lock:
            self._proc = proc
        try:
            proc.wait()
        finally:
            with self._lock:
                self._proc = None
