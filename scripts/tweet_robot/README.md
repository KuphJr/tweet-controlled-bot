# Tweet-Controlled SO-101 Duck Robot

Let people on X control a physical SO-101 arm during a livestream. Viewers
**quote-tweet** or **comment (reply)** on a source tweet asking the robot to put
one of four rubber ducks — **orange, green, yellow, pink** — on the target. The
robot removes whatever duck is currently on the target (if any), places the
requested one, narrates what it's doing with ElevenLabs TTS, and replies from
the same `@KuphDev` account that posted the stream tweet.

> **Design principle:** LLMs only *parse text* and *classify images*.
> Deterministic, validated enums decide which policy runs. Raw model output never
> selects a code path, shell command, or policy.

---

## End-to-end flow

```
X quote/comment ──> TwitterReader (TwitterApi.io) ──> CommandParser (OpenAI)
                                                          │ valid? (enum + confidence)
                                                          ▼
                              FIFO queue (per-author limit, immediate ack reply)
                                                          │
                                                          ▼
   Executor: capture ─> WorkspaceDetector (OpenAI vision) ─> remove_<color> policy
             ─> verify target clear ─> place_<requested> policy ─> verify on target
                          │                         │
                       TTS narration            XPostWriter (official X API)
```

Per command:

1. Acknowledge immediately on accept ("Queued! You're #N…").
2. Capture a top-down image, detect the current target state.
3. If a duck is on the target, run `remove_<color>`, wait 0.5s, verify the target is clear.
   (If the target is already empty, skip removal.)
4. Run `place_<requested_color>`, wait 0.5s, verify the requested color is on the target.
5. On any policy/vision/verification failure → ERROR state, narrate "Error
   encountered", and notify `@KuphDev`. Success posts **no** reply (the stream shows it).

---

## Architecture (one file per concern)

| Module | Responsibility |
| --- | --- |
| `config.py` | Enums, dataclasses (`Command`, `PolicyResult`, …), `.env` loading, the 8 policy paths (env-overridable), all tunables. |
| `robot_policy_runner.py` | Isolated `run_policy()` (build context → `StopAtNeutralStrategy` → teardown), `capture_workspace_image()`, `disable_torque_on_shutdown()`. Reuses the proven `scripts/run_so101_policy.py` strategy unchanged. |
| `command_parser.py` | OpenAI strict-JSON parse of text → `RequestedColor` + confidence gating. |
| `workspace_detector.py` | OpenAI vision strict-JSON → `WorkspaceState`, with optional folder-mapped few-shot examples and retry-once. |
| `reply_generator.py` | Short, fun, context-aware public replies — every category has a deterministic fallback. Cosmetic only. |
| `twitter_reader.py` | Read-only TwitterApi.io polling of **quotes + direct replies**, no-backfill startup, cross-source dedup, nested-reply loop guard. |
| `x_post_writer.py` | Official X API (Tweepy) posting from `@KuphDev` with reply targeting, two-tier rate limiting, and failure tolerance; `--dry-run` or blank creds → log-only. |
| `tts.py` | Non-blocking ElevenLabs synth + configurable external player (`ffplay`/`aplay`/`paplay`); serialized so narrations do not overlap. |
| `state_store.py` | Atomic JSON persistence (seen/processed/failed IDs, flags, rate-limit timestamps). Queue is **not** replayed on restart. |
| `controller.py` | State machine, FIFO queue + per-author limit, acceptance rules, the remove→place pipeline, admin handling, JSONL logging. |
| `run_tweet_robot.py` | CLI entry point: wiring, signal handling, graceful shutdown, dry-run / no-robot modes. |

---

## Setup

1. **Install the extra dependencies** (into the existing lerobot `uv` environment):

   ```bash
   uv pip install -r scripts/tweet_robot/requirements.txt
   ```

2. **Configure secrets.** Copy the example and fill it in:

   ```bash
   cp scripts/tweet_robot/.env.example scripts/tweet_robot/.env
   ```

   | Var(s) | Where to get it |
   | --- | --- |
   | `OPENAI_API_KEY`, `OPENAI_COMMAND_MODEL`, `OPENAI_VISION_MODEL` | platform.openai.com. Defaults: `gpt-5.4-mini` for command parsing + reply text (fast/cheap), `gpt-5.5` for vision (SOTA, for reliable success/error detection). |
   | `TWITTERAPI_IO_API_KEY` | twitterapi.io — **read** side (quotes + replies). |
   | `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | developer.x.com — OAuth 1.0a **user-context** creds for `@KuphDev`, the same account that posts the stream tweet and replies. No bearer token needed. |
   | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID` | elevenlabs.io — optional TTS. |
   | `AUDIO_PLAYER_CMD` | Ubuntu playback command; the audio file path is appended as the last arg. `ffplay -nodisp -autoexit -loglevel quiet` (default), `paplay`, or `aplay`. ElevenLabs returns MP3 — `ffplay` handles it directly; `aplay`/`paplay` expect WAV. |
   | `ADMIN_HANDLE`, `BOT_HANDLE` | Single-account mode defaults both to `KuphDev`. |

   `.env` is git-ignored. Never commit real secrets.

3. **Camera crop.** The top camera applies a deterministic crop from
   `camera_crop.json` (see [`CAMERA_CROP.md`](../../CAMERA_CROP.md)). Re-author it
   with `scripts/preview_camera_crop.py` if the rig moves. The vision detector
   uses the same cropped view as the policies.

---

## Vision examples (recommended before going live)

Optional but strongly recommended: drop a few **cropped** example images (matching
the camera view) into `scripts/tweet_robot/vision_examples/<folder>/`. Folder
names are human-readable; they're mapped to a canonical state automatically:

- name containing `error` → `error` (e.g. `flipped over duck error`, `misplaced duck error`)
- name containing `empty` → `empty`
- otherwise the leading color + "on target" → `<color>_on_target`

So the existing layout works as-is:

```
vision_examples/
├── orange duck on target/   ├── empty target/
├── green duck on target/    ├── flipped over duck error/
├── yellow duck on target/   └── misplaced duck error/
└── pink duck on target/
```

Up to 3 images per folder are sent as labeled few-shot references (examples at
"low" detail to keep token cost flat). The detector works fine with no examples
at all. Add more `error` variants to improve robustness.

---

## Running

Ramp up safely:

```bash
# 1) No hardware movement and no posting — poll, parse, check vision, log only.
uv run python scripts/tweet_robot/run_tweet_robot.py \
    --source-tweet-id 1234567890123456789 --no-robot --dry-run

# 2) Full live run.
uv run python scripts/tweet_robot/run_tweet_robot.py --source-tweet-id 1234567890123456789
```

### CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source-tweet-id <id>` | — | Required for live runs. The tweet users quote/comment on. |
| `--poll-interval-s <s>` | `10` | Seconds between TwitterApi.io polls. |
| `--dry-run` | off | Do not move the robot and **do not post** (log-only writer). Still parses, captures/checks images, generates intended replies, and logs. Verification mismatches are logged, not failed. |
| `--no-tts` | off | Disable ElevenLabs narration (log intended speech). |
| `--no-robot` | off | Skip policy execution; vision still runs via a standalone camera open if available. |
| `--state-file <path>` | `runtime/tweet_robot_state.json` | JSON state file. |
| `--log-file <path>` | `runtime/tweet_robot_commands.jsonl` | JSONL command log. |

---

## Admin controls (`@KuphDev` only)

Admins issue commands as quote-tweets or comments on the same source tweet:

| Command (substring match) | Effect |
| --- | --- |
| `pause` | Finish the current command, then stop processing the queue (state → PAUSED). Admin commands still work. |
| `resume` | Resume processing queued commands. |
| `shutdown` | Graceful shutdown (stops polling/audio, disables torque, exits). |
| `restart` | Graceful shutdown, then re-launch the same process. Clears error/paused state on the way back up — handy for clearing false-flag errors AFK. |
| `status` | Reply with state, queue length, and error status. |
| `clear queue` | Drop queued user commands (a running command finishes). |
| `reset error` | Clear ERROR state immediately without restarting (→ IDLE or PAUSED). |

`pause` never interrupts a running policy. `shutdown`, `restart`, and **Ctrl+C**
are the interrupt paths (they set the shared shutdown event, which a running
policy checks each tick).

If an admin's message is a normal duck request (no admin keyword), it's treated as
a normal command.

---

## Queue, replies, and rate limits

- FIFO queue, max `MAX_QUEUE_SIZE` (5). Only valid commands are queued.
- **One active-or-queued command per author.** A second request while one is
  pending gets a "you already have one" reply, not a second slot.
- Duplicate tweet IDs are ignored. Invalid commands get a helpful reply but aren't queued.
- Single-account mode lets `@KuphDev` issue admin and normal commands. To avoid
  feedback loops, nested replies are ignored when TwitterApi.io provides parent
  metadata, and generated self-authored acknowledgement/status text is ignored.
- Posting is limited by default to: queued acknowledgements, invalid-command
  replies, queue-full / duplicate-author replies, error notifications, and
  admin/status replies. **No success replies.**
- Rate limits: `MAX_REPLIES_PER_HOUR` (30) for normal replies;
  `ABSOLUTE_MAX_POSTS_PER_HOUR` (50) for all posts including critical `@KuphDev`
  notifications. Over the normal cap → keep running, skip normal replies. Over the
  absolute cap → post nothing, log intended text.

---

## Updating

- **Swap a policy checkpoint** without code changes: set the matching env var in
  `.env` (`POLICY_PLACE_ORANGE`, `POLICY_REMOVE_GREEN`, …) to a new path (absolute
  or relative to the repo root). Defaults live in `config.py` (`_DEFAULT_POLICY_PATHS`).
- **Tune behavior** via env: `MAX_REPLIES_PER_HOUR`, `ABSOLUTE_MAX_POSTS_PER_HOUR`,
  `MAX_QUEUE_SIZE`, `POLICY_TIMEOUT_S`, `COMMAND_CONFIDENCE_THRESHOLD`,
  `VISION_CONFIDENCE_THRESHOLD`, `CAMERA_WARMUP_S`, `POST_POLICY_WAIT_S`,
  `TTS_TIMEOUT_S`.
- **Change voice/models**: `ELEVENLABS_VOICE_ID` / `ELEVENLABS_MODEL_ID`,
  `OPENAI_COMMAND_MODEL` / `OPENAI_VISION_MODEL`.

---

## Safety & shutdown

- **Torque:** the SO-101 has a counter-spring/rubber band on servo 3, so torque
  must stay enabled while running. Each policy run uses
  `disable_torque_on_disconnect=False`, so the arm holds its neutral pose between
  the remove and place policies. Torque is disabled **only** on graceful shutdown.
- **Ctrl+C / admin `shutdown`:** stop polling → stop audio → (interrupt a running
  policy via the shutdown event) → disable torque → save state → exit. If the
  torque-disable write fails (the servo-3 rubber band can cause this), physically
  power off the arm to be safe.
- Every external call (TwitterApi.io, X API, OpenAI, ElevenLabs, camera, policy
  execution, robot connect/disconnect) is wrapped so a failure logs and degrades
  gracefully instead of crashing the controller.

---

## Logs & state

- `runtime/tweet_robot_commands.jsonl` — one JSON object per executed command
  (timestamp, tweet_id, source `quote`/`reply`, author, raw text, parsed color,
  workspace states before/after removal/after place, policies run, final state,
  success/error, replies attempted, TTS attempted).
- `runtime/tweet_robot_state.json` — atomic snapshot of seen/processed/failed IDs,
  paused/error flags, and reply rate-limit timestamps.
- **Restart behavior:** old queued commands are **never** replayed. On startup the
  queue is empty and all currently-visible quotes/replies are marked seen
  (no-backfill), so only items posted after startup are processed. Seen/processed
  IDs persist so the same tweet never runs twice. **Re-running the script (or the
  admin `restart` command) clears any ERROR/PAUSED state** so it just works again —
  re-running implies you've fixed the workspace.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `camera ... failed to open` / busy | Another process holds `/dev/video0`. The robot disconnects between policies so vision can grab frames; make sure nothing else (e.g. a preview script) is using the camera. |
| No audio | Wrong `AUDIO_PLAYER_CMD` for your system, or MP3 vs WAV mismatch (`aplay`/`paplay` need WAV; prefer `ffplay`). Failures are logged and ignored. |
| Replies not posting | Check the X OAuth 1.0a creds and that you're not in `--dry-run`; watch logs for rate-limit messages. X self-serve API replies may 403 unless the replying account has been summoned by the target tweet's author; using `@KuphDev` for both the source tweet and replies makes comment/quote acknowledgements more reliable. Posting failures never crash the robot. |
| Vision misclassifies | Add/curate `vision_examples/` (especially `error` variants); re-author the camera crop so the target is centered; raise `VISION_CONFIDENCE_THRESHOLD`. |
| Policy timeout / ERROR | The policy didn't return to neutral within `POLICY_TIMEOUT_S`. Inspect the arm, then `reset error` (admin) to resume. |
| Want real hardware but no X posting | Leave the `X_*` credentials blank in `.env` — the writer falls back to log-only. (`--dry-run` also disables posting but skips the robot.) |
| `--source-tweet-id is required` | Provide the source tweet ID users should quote/comment on. |
