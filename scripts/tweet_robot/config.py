"""Shared configuration, enums, and dataclasses for the tweet-controlled robot.

This module centralizes every tunable value so the rest of the package never
hard-codes secrets, model names, policy paths, or thresholds. Secrets/config are
loaded from ``scripts/tweet_robot/.env`` (see ``.env.example``).

Design principle: deterministic, validated enums (never raw LLM prose) decide
which policy runs. The LLM only fills in ``RequestedColor`` / ``WorkspaceState``
values that are validated against these enums here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

PACKAGE_DIR = Path(__file__).resolve().parent
# scripts/tweet_robot -> scripts -> repo root
REPO_ROOT = PACKAGE_DIR.parent.parent
ENV_PATH = PACKAGE_DIR / ".env"

# Load .env from the package directory (no-op if it does not exist).
load_dotenv(ENV_PATH)

DEFAULT_STATE_FILE = REPO_ROOT / "runtime" / "tweet_robot_state.json"
DEFAULT_LOG_FILE = REPO_ROOT / "runtime" / "tweet_robot_commands.jsonl"
VISION_EXAMPLES_DIR = PACKAGE_DIR / "vision_examples"


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #


class RequestedColor(str, Enum):
    """The four duck colors the robot can place/remove."""

    ORANGE = "orange"
    GREEN = "green"
    YELLOW = "yellow"
    PINK = "pink"


class PolicyAction(str, Enum):
    PLACE = "place"
    REMOVE = "remove"


class WorkspaceState(str, Enum):
    """Vision-detected state of the target zone."""

    ORANGE_ON_TARGET = "orange_on_target"
    GREEN_ON_TARGET = "green_on_target"
    YELLOW_ON_TARGET = "yellow_on_target"
    PINK_ON_TARGET = "pink_on_target"
    EMPTY = "empty"
    ERROR = "error"

    @property
    def on_target_color(self) -> "RequestedColor | None":
        """Return the color currently on the target, or None for empty/error."""
        return _STATE_TO_COLOR.get(self)


_STATE_TO_COLOR: dict[WorkspaceState, RequestedColor] = {
    WorkspaceState.ORANGE_ON_TARGET: RequestedColor.ORANGE,
    WorkspaceState.GREEN_ON_TARGET: RequestedColor.GREEN,
    WorkspaceState.YELLOW_ON_TARGET: RequestedColor.YELLOW,
    WorkspaceState.PINK_ON_TARGET: RequestedColor.PINK,
}

_COLOR_TO_STATE: dict[RequestedColor, WorkspaceState] = {v: k for k, v in _STATE_TO_COLOR.items()}


def color_on_target_state(color: RequestedColor) -> WorkspaceState:
    """Return the ``<color>_on_target`` state for a given color."""
    return _COLOR_TO_STATE[color]


class ControllerState(str, Enum):
    STARTING = "STARTING"
    IDLE = "IDLE"
    PAUSED = "PAUSED"
    BUSY = "BUSY"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class CommandSource(str, Enum):
    QUOTE = "quote"
    REPLY = "reply"


class AdminCommand(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    STATUS = "status"
    CLEAR_QUEUE = "clear_queue"
    RESET_ERROR = "reset_error"


# --------------------------------------------------------------------------- #
# Shared dataclasses                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class ParsedCommand:
    """Result of LLM command parsing, validated against ``RequestedColor``."""

    valid: bool
    requested_color: RequestedColor | None
    confidence: float
    reason: str


@dataclass
class WorkspaceDetection:
    """Result of LLM vision workspace-state detection."""

    state: WorkspaceState
    current_color: RequestedColor | None
    confidence: float
    is_error: bool
    error_reason: str | None
    description: str


@dataclass
class Command:
    """A validated, queued user command."""

    tweet_id: str
    source: CommandSource
    author_handle: str
    author_name: str
    raw_text: str
    requested_color: RequestedColor
    enqueued_at: float
    replies_attempted: list[str] = field(default_factory=list)
    tts_attempted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tweet_id": self.tweet_id,
            "source": self.source.value,
            "author_handle": self.author_handle,
            "author_name": self.author_name,
            "raw_text": self.raw_text,
            "requested_color": self.requested_color.value,
            "enqueued_at": self.enqueued_at,
            "replies_attempted": list(self.replies_attempted),
            "tts_attempted": list(self.tts_attempted),
        }


@dataclass
class PolicyResult:
    """Outcome of a single isolated policy rollout."""

    success: bool
    stop_reason: str  # "neutral" | "duration" | "shutdown" | "error"
    elapsed_s: float
    error_message: str | None = None


# --------------------------------------------------------------------------- #
# Policy paths (overridable via env, swappable without code changes)           #
# --------------------------------------------------------------------------- #

_DEFAULT_POLICY_PATHS: dict[tuple[RequestedColor, PolicyAction], str] = {
    (RequestedColor.ORANGE, PolicyAction.PLACE): "outputs/train/place_orange_duck_pruned_2nd/checkpoints/last/pretrained_model",
    (RequestedColor.ORANGE, PolicyAction.REMOVE): "outputs/train/remove_orange_duck/checkpoints/last/pretrained_model",
    (RequestedColor.GREEN, PolicyAction.PLACE): "outputs/train/place_green_duck/checkpoints/last/pretrained_model",
    (RequestedColor.GREEN, PolicyAction.REMOVE): "outputs/train/remove_green_duck/checkpoints/140000/pretrained_model",
    (RequestedColor.YELLOW, PolicyAction.PLACE): "outputs/train/place_yellow_duck_pruned_2/checkpoints/200000/pretrained_model",
    (RequestedColor.YELLOW, PolicyAction.REMOVE): "outputs/train/remove_yellow_duck_pruned/checkpoints/400000/pretrained_model",
    (RequestedColor.PINK, PolicyAction.PLACE): "outputs/train/place_pink_duck_pruned/checkpoints/last/pretrained_model",
    (RequestedColor.PINK, PolicyAction.REMOVE): "outputs/train/remove_pink_duck_pruned/checkpoints/last/pretrained_model",
}


def _policy_env_key(color: RequestedColor, action: PolicyAction) -> str:
    return f"POLICY_{action.value.upper()}_{color.value.upper()}"


def get_policy_path(color: RequestedColor, action: PolicyAction) -> str:
    """Resolve a policy checkpoint path, honoring env overrides.

    Relative paths are resolved against the repository root so the script can
    be launched from any working directory.
    """
    raw = os.getenv(_policy_env_key(color, action), _DEFAULT_POLICY_PATHS[(color, action)])
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def normalize_handle(handle: str | None) -> str:
    """Lower-case and strip a leading '@' for case-insensitive handle compares."""
    if not handle:
        return ""
    return handle.lstrip("@").strip().lower()


# --------------------------------------------------------------------------- #
# Application config                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class AppConfig:
    """Runtime configuration assembled from environment variables.

    CLI flags in ``run_tweet_robot.py`` override a few fields (state/log paths,
    poll interval, and the ``--no-*`` / ``--dry-run`` toggles) after construction.
    """

    # OpenAI
    # Command parsing + reply text use a fast, cheap model; vision uses the
    # state-of-the-art model so success/error detection is reliable.
    openai_api_key: str = ""
    openai_command_model: str = "gpt-5.4-mini"
    openai_vision_model: str = "gpt-5.5"

    # TwitterApi.io (read side)
    twitterapi_io_api_key: str = ""

    # Official X API (write side, OAuth 1.0a user context)
    x_api_key: str = ""
    x_api_secret: str = ""
    x_access_token: str = ""
    x_access_token_secret: str = ""

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    audio_player_cmd: str = "ffplay -nodisp -autoexit -loglevel quiet"

    # Handles
    admin_handle: str = "KuphDev"
    bot_handle: str = "KuphDev"

    # Rate limits / queue / timeouts
    max_replies_per_hour: int = 30
    absolute_max_posts_per_hour: int = 50
    max_queue_size: int = 5
    policy_timeout_s: int = 30
    policy_error_retries: int = 2
    replies_v2_poll_interval_s: float = 180.0
    replies_v2_max_pages_per_poll: int = 1

    # Confidence gating
    command_confidence_threshold: float = 0.5
    vision_confidence_threshold: float = 0.55

    # TTS
    tts_timeout_s: float = 20.0

    # Robot / camera (mirrors scripts/run_so101_policy.py)
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so101_follower"
    camera_name: str = "top"
    camera_path: str = "/dev/video0"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_fourcc: str = "MJPG"
    camera_warmup_s: float = 0.5
    control_fps: int = 30

    # Verification pause after each policy before re-capturing
    post_policy_wait_s: float = 0.5

    # Paths
    state_file: Path = field(default_factory=lambda: DEFAULT_STATE_FILE)
    log_file: Path = field(default_factory=lambda: DEFAULT_LOG_FILE)
    vision_examples_dir: Path = field(default_factory=lambda: VISION_EXAMPLES_DIR)

    # Runtime toggles (set by CLI)
    poll_interval_s: float = 15.0
    dry_run: bool = False
    no_tts: bool = False
    no_robot: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_command_model=_env_str("OPENAI_COMMAND_MODEL", "gpt-5.4-mini"),
            openai_vision_model=_env_str("OPENAI_VISION_MODEL", "gpt-5.5"),
            twitterapi_io_api_key=_env_str("TWITTERAPI_IO_API_KEY"),
            x_api_key=_env_str("X_API_KEY"),
            x_api_secret=_env_str("X_API_SECRET"),
            x_access_token=_env_str("X_ACCESS_TOKEN"),
            x_access_token_secret=_env_str("X_ACCESS_TOKEN_SECRET"),
            elevenlabs_api_key=_env_str("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=_env_str("ELEVENLABS_VOICE_ID"),
            elevenlabs_model_id=_env_str("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
            audio_player_cmd=_env_str("AUDIO_PLAYER_CMD", "ffplay -nodisp -autoexit -loglevel quiet"),
            admin_handle=_env_str("ADMIN_HANDLE", "KuphDev"),
            bot_handle=_env_str("BOT_HANDLE", "KuphDev"),
            max_replies_per_hour=_env_int("MAX_REPLIES_PER_HOUR", 30),
            absolute_max_posts_per_hour=_env_int("ABSOLUTE_MAX_POSTS_PER_HOUR", 50),
            max_queue_size=_env_int("MAX_QUEUE_SIZE", 5),
            policy_timeout_s=_env_int("POLICY_TIMEOUT_S", 30),
            policy_error_retries=_env_int("POLICY_ERROR_RETRIES", 2),
            replies_v2_poll_interval_s=_env_float("REPLIES_V2_POLL_INTERVAL_S", 180.0),
            replies_v2_max_pages_per_poll=_env_int("REPLIES_V2_MAX_PAGES_PER_POLL", 1),
            command_confidence_threshold=_env_float("COMMAND_CONFIDENCE_THRESHOLD", 0.5),
            vision_confidence_threshold=_env_float("VISION_CONFIDENCE_THRESHOLD", 0.55),
            tts_timeout_s=_env_float("TTS_TIMEOUT_S", 20.0),
            robot_port=_env_str("ROBOT_PORT", "/dev/ttyACM0"),
            robot_id=_env_str("ROBOT_ID", "so101_follower"),
            camera_path=_env_str("CAMERA_PATH", "/dev/video0"),
            camera_warmup_s=_env_float("CAMERA_WARMUP_S", 0.5),
            post_policy_wait_s=_env_float("POST_POLICY_WAIT_S", 0.5),
        )

    @property
    def admin_handle_norm(self) -> str:
        return normalize_handle(self.admin_handle)

    @property
    def bot_handle_norm(self) -> str:
        return normalize_handle(self.bot_handle)
