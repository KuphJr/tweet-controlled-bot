#!/usr/bin/env python
"""Repair a LeRobotDataset v3.0 dataset.

Handles two independent failure modes that both surface during training as
"can't find frame N in MP4 with M < N frames":

A) Streaming encoder dropped frames during recording
---------------------------------------------------
``lerobot-record --dataset.streaming_encoding=true`` runs the video encoder
in a background thread with a bounded queue (default ``encoder_queue_maxsize=30``).
When the recording loop briefly bursts faster than the encoder can drain the
queue, ``put(image, timeout=0.1)`` raises ``queue.Full`` and the frame is
**silently dropped from the MP4** while the parquet row is still added (see
``src/lerobot/datasets/video_utils.py::StreamingVideoEncoder.feed_frame`` and the
"Encoder queue full ... dropped N frame(s)" warning).

B) Stale leftover files from a prior ``delete_episodes`` + ``push_to_hub``
---------------------------------------------------------------------------
``delete_episodes`` writes a brand-new dataset (renumbered ``episode_index``,
possibly different ``(chunk_index, file_index)`` file layout) into a separate
local directory. When that pruned dataset is then pushed back to the **same**
Hub repo, ``LeRobotDataset.push_to_hub`` calls ``hub_api.upload_folder(...)``
which is purely additive: files with colliding paths get overwritten, but
orphan files at the old layout stay on the Hub. ``snapshot_download`` then
faithfully delivers both, and the loaded ``meta/episodes`` ends up with
duplicate ``episode_index`` values from the original (pre-prune) numbering
mixed in with the new canonical numbering. The reader's
``self._meta.episodes[ep_idx]`` (position-based) then returns the wrong
episode's video offsets for half the rows -> spurious IndexError.

This shows up at training time as one of two flavors of the same disease --
the parquet asks for a frame that doesn't exist in the MP4:

    IndexError: Invalid frame index=1358 for streamIndex=0; must be less than 1341
    FrameTimestampError: queried timestamp 53.6333s, but loaded video timestamps only reach 44.6667s

No timestamp rewrite alone fixes this: the per-row ``timestamp`` is already
``frame_index / fps`` (set by ``DatasetWriter.add_frame``), and the MP4
genuinely has fewer frames than the parquet expects.

The good news
-------------
``DatasetWriter._save_episode_video`` records each episode's actual encoded
MP4 duration into ``videos/<key>/from_timestamp`` and ``to_timestamp`` (via
``get_video_duration_in_s`` on the per-episode temp MP4, **before** the file
is concatenated into the multi-episode video file). That gives us a
ground-truth signal of how many frames actually made it into each episode's
MP4 slice::

    actual_encoded_frames[E] = round((to_timestamp[E] - from_timestamp[E]) * fps)

If ``actual_encoded_frames[E] < length[E]``, exactly
``length[E] - actual_encoded_frames[E]`` rows of that episode's parquet have
no matching MP4 frame.

What this script does
---------------------
Three phases, all writing only into a fresh ``--out-root`` (the source root /
Hub snapshot is never modified):

**Phase 1 - Stale-leftover cleanup (only if needed).** Detects case (B) by
comparing ``info.json:total_episodes`` to the sum of row counts across all
``meta/episodes/*/*.parquet`` files. If they don't match, it picks the one
metadata parquet whose ``episode_index`` is exactly ``[0..total-1]`` as the
canonical one, derives the set of ``(chunk, file)`` pairs that the canonical
metadata actually references for ``data/`` and ``videos/<key>/``, and
deletes every orphan parquet/MP4 from the ``--out-root`` mirror.

**Phase 2 - Streaming-drop truncation.** For each episode, computes
``actual = round((to_timestamp - from_timestamp) * fps)`` (the writer records
each episode's true encoded MP4 duration via ``get_video_duration_in_s`` on
the per-episode temp MP4). For any episode where ``actual < length``:

1.  Drops the last ``deficit`` rows of that episode from the data parquet.
2.  Rewrites the per-row ``timestamp`` to a clean ``frame_index / fps`` grid.
3.  Renumbers the global ``index`` column across **all** data parquet files.
4.  Updates ``length``, ``dataset_from_index``, ``dataset_to_index`` and
    ``videos/<key>/from_timestamp`` / ``to_timestamp`` in every
    ``meta/episodes/*.parquet``, recomputing from/to in seconds from the
    truncated lengths so the per-episode MP4 slice boundaries stay aligned
    with the new parquet timeline.
5.  Updates ``meta/info.json:total_frames``.
6.  **Never** touches MP4 files. The dataset reader continues to pull from
    them at the same byte offsets as before; we just stop asking for frames
    past the end of each slice.

**Phase 3 - Validation.** Instantiates ``LeRobotDataset`` from ``--out-root``
and iterates the first ``--validate-n`` items to confirm decoding works
end-to-end without ``IndexError`` / ``FrameTimestampError``.

Trade-off
---------
Truncation drops rows **at the end of each episode**. The encoder doesn't
always drop frames at the end (drops happen whenever its queue fills up,
often during high-motion middle sections). That means some obs/action pairs
in the middle of each affected episode can be slightly misaligned with the
actual MP4 frame they end up paired with. For the SO-101 ACT workflow where
the last ~5-10 s of each episode is the robot sitting idle waiting for the
next reset, the trailing data is genuinely throwaway, so this trade-off is
in practice harmless.

Usage
-----
The script is meant to run on the cloud pod that does the training (it
imports ``lerobot.datasets.LeRobotDataset`` for validation, which needs
``torchcodec`` / ``pyav``). Typical invocation::

    uv run python scripts/repair_video_timestamps.py \\
        --repo-id ${HF_USER}/remove_duck_from_target_20260517_094243 \\
        --out-root /workspace/datasets/remove_duck_from_target_repaired \\
        --validate-n 5000

Pass ``--dry-run`` to print the per-episode diagnostic and stop without
modifying anything. Pass ``--src-root /path/to/local/copy`` to skip the
Hub download if the dataset is already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("repair_video_timestamps")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub dataset repo id, e.g. 'KuphDev/remove_duck_from_target_20260517_094243'.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Local directory to write the repaired dataset into. Created if missing.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help=(
            "Optional path to an already-downloaded local copy of the dataset. "
            "If omitted, the dataset is snapshot-downloaded from the Hub."
        ),
    )
    parser.add_argument(
        "--video-keys",
        nargs="*",
        default=None,
        help=(
            "Restrict repair to a subset of video keys (default: all video features "
            "in meta/info.json). Multi-key datasets truncate each episode to the "
            "minimum encoded length across the chosen keys so every camera decodes."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS (defaults to value in meta/info.json).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF Hub revision (branch / tag / sha) to download. Defaults to LeRobot CODEBASE_VERSION.",
    )
    parser.add_argument(
        "--link-videos",
        action="store_true",
        help=(
            "Symlink the videos/ directory into --out-root instead of copying. "
            "Saves a lot of disk space; the script never writes to videos/."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-episode diagnostic and stop without modifying anything.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Skip the stale-leftover cleanup pass. Use only if you are sure the dataset "
            "does NOT have leftover files from a prior delete_episodes + push_to_hub round trip."
        ),
    )
    parser.add_argument(
        "--validate-n",
        type=int,
        default=5000,
        help="Number of items to iterate through after repair to confirm decoding works (0 disables).",
    )
    parser.add_argument(
        "--validate-step",
        type=int,
        default=1,
        help="Sample every N-th item during validation (use >1 to spot-check huge datasets faster).",
    )
    parser.add_argument(
        "--overwrite-out",
        action="store_true",
        help="Delete --out-root if it already exists before starting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity to WARNING.",
    )
    return parser.parse_args()


def setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Source materialization                                                      #
# --------------------------------------------------------------------------- #


def materialize_source(repo_id: str, src_root: Path | None, revision: str | None) -> Path:
    """Return a local path that contains the source dataset, downloading if needed."""
    if src_root is not None:
        src_root = src_root.expanduser().resolve()
        if not src_root.is_dir():
            raise FileNotFoundError(f"--src-root does not exist or is not a directory: {src_root}")
        if not (src_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"--src-root {src_root} is missing meta/info.json -- not a LeRobotDataset."
            )
        logger.info("Using existing local source root: %s", src_root)
        return src_root

    from huggingface_hub import snapshot_download

    if revision is None:
        try:
            from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

            revision = CODEBASE_VERSION
        except Exception:  # pragma: no cover
            revision = None

    logger.info("Snapshot-downloading %s (revision=%s) ...", repo_id, revision)
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    )
    logger.info("Downloaded to %s", local_dir)
    return local_dir


def populate_out_root(src_root: Path, out_root: Path, link_videos: bool, overwrite: bool) -> None:
    """Mirror ``src_root`` into ``out_root``, optionally symlinking the videos dir."""
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"--out-root {out_root} already exists. Pass --overwrite-out to delete and recreate it."
            )
        logger.warning("Removing existing --out-root %s", out_root)
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True)

    for entry in sorted(src_root.iterdir()):
        if entry.name.startswith(".") and entry.is_dir():
            continue

        target = out_root / entry.name

        if entry.is_dir() and entry.name == "videos" and link_videos:
            target.symlink_to(entry.resolve(), target_is_directory=True)
            logger.info("Symlinked videos/ -> %s", entry.resolve())
            continue

        if entry.is_dir():
            shutil.copytree(entry, target, symlinks=False)
        else:
            shutil.copy2(entry, target)

    logger.info("Populated out root at %s", out_root)


# --------------------------------------------------------------------------- #
# Episode plan                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class EpisodePlan:
    """Repair plan for a single episode."""

    episode_index: int
    old_length: int
    new_length: int
    old_dataset_from_index: int
    old_dataset_to_index: int
    new_dataset_from_index: int
    new_dataset_to_index: int
    # per video_key: (chunk_idx, file_idx, new_from_ts, new_to_ts)
    new_video_offsets: dict[str, tuple[int, int, float, float]]
    # which (chunk_idx, file_idx) of meta/episodes this row lives in
    meta_chunk_index: int
    meta_file_index: int
    # original from/to timestamps per key, useful for diagnostics
    actual_encoded_frames: dict[str, int]

    @property
    def deficit(self) -> int:
        return self.old_length - self.new_length


def _round_frames(seconds: float, fps: int) -> int:
    """Convert a duration in seconds to an integer frame count. Avoids FP off-by-one."""
    return int(round(seconds * fps))


# --------------------------------------------------------------------------- #
# Stale-leftover cleanup                                                      #
# --------------------------------------------------------------------------- #
#
# When `lerobot.datasets.dataset_tools.delete_episodes` is run on an existing
# dataset and the **result** is pushed back to the same Hub repo with
# `LeRobotDataset.push_to_hub`, the upload is purely additive -- it overwrites
# files at colliding paths but never deletes orphans. If the post-prune dataset
# uses a different `(chunk_index, file_index)` layout than the original (which
# `delete_episodes` is free to do, since it writes from a fresh
# `LeRobotDatasetMetadata.create(...)`), the Hub repo ends up with a mix of
# new files at new paths AND stale originals at the old paths.
#
# `snapshot_download` then faithfully delivers both, which surfaces during
# training as duplicate `episode_index` values (the stale meta/episodes parquets
# use the original episode numbering, the new consolidated metadata uses the
# renumbered 0..total_episodes-1).
#
# This phase trusts `meta/info.json:total_episodes` and finds the single
# "canonical" metadata parquet whose `episode_index` column is exactly
# `[0..total_episodes-1]`. Then it derives which `data/` and `videos/` files
# that canonical metadata actually references, and deletes everything else
# under those subtrees in the local out_root mirror.


def _read_episode_index_sequence(path: Path) -> list[int]:
    return pq.read_table(path, columns=["episode_index"]).column("episode_index").to_pylist()


def _is_canonical_metadata_file(path: Path, total_episodes: int) -> bool:
    """A file is canonical iff its episode_index column is exactly [0..total-1] (in any order)."""
    eps = _read_episode_index_sequence(path)
    return len(eps) == total_episodes and sorted(eps) == list(range(total_episodes))


def find_canonical_metadata_file(out_root: Path, total_episodes: int) -> Path:
    """Return the meta/episodes parquet that exactly matches info.json's total_episodes."""
    ep_files = sorted((out_root / "meta" / "episodes").glob("*/*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No meta/episodes/*/*.parquet under {out_root}")

    matches = [p for p in ep_files if _is_canonical_metadata_file(p, total_episodes)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one canonical metadata file found with rows=={total_episodes} and ep_idx==[0..N-1]: "
            f"{[str(p) for p in matches]}. Don't know which one is authoritative."
        )

    summary = "\n".join(
        f"    {p.name}: rows={len(_read_episode_index_sequence(p))}" for p in ep_files
    )
    raise RuntimeError(
        f"No meta/episodes file matches info.json:total_episodes={total_episodes}. "
        f"Cannot determine the canonical metadata.\n  Found:\n{summary}"
    )


def referenced_data_and_video_files(
    canonical_meta_path: Path, video_keys: list[str]
) -> tuple[set[tuple[int, int]], dict[str, set[tuple[int, int]]]]:
    """Return the set of (chunk, file) for data/ and per video_key, as referenced by the canonical metadata."""
    cols = ["data/chunk_index", "data/file_index"] + [
        f"videos/{k}/{x}" for k in video_keys for x in ("chunk_index", "file_index")
    ]
    table = pq.read_table(canonical_meta_path, columns=cols)

    data_chunks = table.column("data/chunk_index").to_pylist()
    data_files = table.column("data/file_index").to_pylist()
    data_refs = {(int(c), int(f)) for c, f in zip(data_chunks, data_files, strict=True)}

    video_refs: dict[str, set[tuple[int, int]]] = {}
    for key in video_keys:
        chunks = table.column(f"videos/{key}/chunk_index").to_pylist()
        files = table.column(f"videos/{key}/file_index").to_pylist()
        video_refs[key] = {(int(c), int(f)) for c, f in zip(chunks, files, strict=True)}

    return data_refs, video_refs


def _parse_chunk_file_from_path(path: Path) -> tuple[int, int] | None:
    """Parse a path like '.../chunk-000/file-012.<ext>' into (0, 12). Returns None on mismatch."""
    parent = path.parent.name
    name = path.stem
    if not parent.startswith("chunk-") or not name.startswith("file-"):
        return None
    try:
        return int(parent.removeprefix("chunk-")), int(name.removeprefix("file-"))
    except ValueError:
        return None


def cleanup_stale_files(
    out_root: Path,
    canonical_meta_path: Path,
    data_refs: set[tuple[int, int]],
    video_refs: dict[str, set[tuple[int, int]]],
) -> dict:
    """Delete stale leftover files from a prior `delete_episodes` + `push_to_hub` mistake.

    Operates only inside `out_root` -- the source snapshot is never touched.
    Returns a small report dict.
    """
    report = {
        "deleted_meta": [],  # type: ignore[var-annotated]
        "deleted_data": [],
        "deleted_videos": defaultdict(list),
    }

    # ── meta/episodes ─────────────────────────────────────────────────
    for p in sorted((out_root / "meta" / "episodes").glob("*/*.parquet")):
        if p.resolve() == canonical_meta_path.resolve():
            continue
        if p.is_symlink() or p.is_file():
            p.unlink()
            report["deleted_meta"].append(str(p.relative_to(out_root)))

    # ── data ──────────────────────────────────────────────────────────
    for p in sorted((out_root / "data").glob("*/*.parquet")):
        key = _parse_chunk_file_from_path(p)
        if key is None:
            continue
        if key in data_refs:
            continue
        if p.is_symlink() or p.is_file():
            p.unlink()
            report["deleted_data"].append(str(p.relative_to(out_root)))

    # ── videos/<key> ──────────────────────────────────────────────────
    videos_root = out_root / "videos"
    if videos_root.is_dir():
        for key, refs in video_refs.items():
            for p in sorted((videos_root / key).glob("*/*.mp4")):
                k = _parse_chunk_file_from_path(p)
                if k is None or k in refs:
                    continue
                if p.is_symlink() or p.is_file():
                    p.unlink()
                    report["deleted_videos"][key].append(str(p.relative_to(out_root)))

    return report


def load_episode_metadata(out_root: Path) -> list[dict]:
    """Load all meta/episodes/*.parquet files into a list of row dicts.

    Returns the rows in episode_index order. Each dict carries every column that
    was in the source parquet so we can write it back unchanged except for the
    fields we explicitly modify.
    """
    episode_files = sorted((out_root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No meta/episodes/*/*.parquet under {out_root}")

    rows: list[dict] = []
    for path in episode_files:
        table = pq.read_table(path)
        for i in range(table.num_rows):
            row = {name: table.column(name)[i].as_py() for name in table.schema.names}
            rows.append(row)

    rows.sort(key=lambda r: r["episode_index"])
    return rows


def build_plans(
    episode_rows: list[dict],
    video_keys: list[str],
    fps: int,
) -> list[EpisodePlan]:
    """Build one EpisodePlan per episode using ``(to - from) * fps`` as ground truth."""
    # Guard: episode_index must be globally unique. If it isn't, the dataset has
    # leftover stale metadata from a prior prune + push (or some other merge that
    # didn't renumber); the cleanup phase should have already removed those.
    seen: dict[int, int] = {}
    for row in episode_rows:
        ep = int(row["episode_index"])
        seen[ep] = seen.get(ep, 0) + 1
    dups = sorted(ep for ep, c in seen.items() if c > 1)
    if dups:
        sample = ", ".join(str(e) for e in dups[:20])
        raise RuntimeError(
            f"Metadata contains {len(dups)} duplicate episode_index value(s): {sample}"
            f"{' ...' if len(dups) > 20 else ''}. "
            "This usually means the dataset has stale leftover files from a previous "
            "`delete_episodes` + `push_to_hub` round trip; cleanup should have removed "
            "them. Re-run the script and check the cleanup phase logs."
        )

    plans: list[EpisodePlan] = []
    cumulative = 0

    for row in episode_rows:
        ep = int(row["episode_index"])
        old_length = int(row["length"])
        old_from = int(row["dataset_from_index"])
        old_to = int(row["dataset_to_index"])
        meta_chunk = int(row["meta/episodes/chunk_index"])
        meta_file = int(row["meta/episodes/file_index"])

        actual_per_key: dict[str, int] = {}
        chunk_file_per_key: dict[str, tuple[int, int]] = {}
        for key in video_keys:
            from_col = f"videos/{key}/from_timestamp"
            to_col = f"videos/{key}/to_timestamp"
            chunk_col = f"videos/{key}/chunk_index"
            file_col = f"videos/{key}/file_index"
            if from_col not in row or to_col not in row:
                raise KeyError(
                    f"Episode {ep} metadata is missing required column "
                    f"'{from_col}' or '{to_col}'. This dataset doesn't have per-key "
                    f"video timestamps; the script can't trust-recompute encoded frames."
                )
            seconds = float(row[to_col]) - float(row[from_col])
            actual_per_key[key] = max(0, _round_frames(seconds, fps))
            chunk_file_per_key[key] = (int(row[chunk_col]), int(row[file_col]))

        # Take the min across keys so every camera decodes within range.
        capped = min(actual_per_key.values()) if actual_per_key else old_length
        new_length = min(old_length, capped)

        plans.append(
            EpisodePlan(
                episode_index=ep,
                old_length=old_length,
                new_length=new_length,
                old_dataset_from_index=old_from,
                old_dataset_to_index=old_to,
                new_dataset_from_index=cumulative,
                new_dataset_to_index=cumulative + new_length,
                # from/to timestamps are filled in a second pass below so they can
                # be cumulative inside each (chunk, file) video file group.
                new_video_offsets={k: (*chunk_file_per_key[k], 0.0, 0.0) for k in video_keys},
                meta_chunk_index=meta_chunk,
                meta_file_index=meta_file,
                actual_encoded_frames=actual_per_key,
            )
        )
        cumulative += new_length

    # Recompute from/to_timestamp per video_key per (chunk_index, file_index) so
    # every episode's slice in the (possibly concatenated) MP4 is the cumulative
    # sum of new_length / fps. This stays consistent with how the writer assigns
    # offsets, just based on the truncated lengths.
    for key in video_keys:
        running: dict[tuple[int, int], float] = defaultdict(float)
        for plan in plans:
            chunk_idx, file_idx, _, _ = plan.new_video_offsets[key]
            offset = running[(chunk_idx, file_idx)]
            duration = plan.new_length / fps
            plan.new_video_offsets[key] = (chunk_idx, file_idx, offset, offset + duration)
            running[(chunk_idx, file_idx)] = offset + duration

    return plans


# --------------------------------------------------------------------------- #
# Diagnostic                                                                  #
# --------------------------------------------------------------------------- #


def print_diagnostic(plans: list[EpisodePlan], video_keys: list[str], fps: int) -> None:
    affected = [p for p in plans if p.deficit > 0]
    total_old = sum(p.old_length for p in plans)
    total_new = sum(p.new_length for p in plans)
    total_deficit = total_old - total_new

    logger.info("=" * 72)
    logger.info("Per-episode diagnostic (fps=%d, video_keys=%s)", fps, video_keys)
    logger.info("=" * 72)
    header = f"{'ep':>4} {'old_len':>8} {'new_len':>8} {'deficit':>8} {'drop %':>7}  per-key encoded frames"
    logger.info(header)
    logger.info("-" * len(header))
    for plan in plans:
        per_key = " ".join(f"{k}={plan.actual_encoded_frames[k]}" for k in video_keys)
        pct = (plan.deficit / plan.old_length * 100) if plan.old_length else 0.0
        marker = " *" if plan.deficit > 0 else ""
        logger.info(
            "%4d %8d %8d %8d %6.2f%%  %s%s",
            plan.episode_index,
            plan.old_length,
            plan.new_length,
            plan.deficit,
            pct,
            per_key,
            marker,
        )
    logger.info("-" * len(header))
    logger.info(
        "Summary: %d/%d episodes affected, total deficit %d / %d rows (%.2f%% of dataset)",
        len(affected),
        len(plans),
        total_deficit,
        total_old,
        (total_deficit / total_old * 100) if total_old else 0.0,
    )
    if affected:
        worst = max(affected, key=lambda p: p.deficit)
        logger.info(
            "Worst episode: ep=%d  deficit=%d/%d (%.2f%%)",
            worst.episode_index,
            worst.deficit,
            worst.old_length,
            worst.deficit / worst.old_length * 100,
        )
    logger.info("=" * 72)


# --------------------------------------------------------------------------- #
# Data parquet rewriting                                                      #
# --------------------------------------------------------------------------- #


def repair_data_parquets(out_root: Path, plans: list[EpisodePlan], fps: int) -> int:
    """Truncate over-long episodes in every data parquet, renumber ``index`` and
    rewrite ``timestamp``. Returns the new total frame count."""
    plan_by_ep = {p.episode_index: p for p in plans}

    data_files = sorted((out_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data/*/*.parquet under {out_root}")

    grand_total = 0
    for data_path in data_files:
        rel = data_path.relative_to(out_root)
        table = pq.read_table(data_path)
        schema = table.schema
        names = set(schema.names)

        for required in ("episode_index", "frame_index", "index", "timestamp"):
            if required not in names:
                raise RuntimeError(
                    f"{data_path} is missing required column '{required}'; "
                    f"available: {sorted(names)}"
                )

        episode_indices = table.column("episode_index").to_numpy(zero_copy_only=False)
        frame_indices = table.column("frame_index").to_numpy(zero_copy_only=False)

        # Build the keep-mask: frame_index < new_length[episode_index]
        # An unknown episode_index here means the data parquet contains rows for an episode
        # that is NOT in the canonical metadata. After the stale-cleanup phase this should
        # be impossible; if it ever happens, it's a real data-integrity problem worth surfacing.
        keep_mask = []
        orphan_eps: set[int] = set()
        for ep, fi in zip(episode_indices, frame_indices, strict=True):
            ep_i = int(ep)
            fi_i = int(fi)
            if ep_i not in plan_by_ep:
                orphan_eps.add(ep_i)
                keep_mask.append(False)
                continue
            keep_mask.append(fi_i < plan_by_ep[ep_i].new_length)

        if orphan_eps:
            raise RuntimeError(
                f"{data_path} contains rows for episode_index value(s) not present in the canonical "
                f"meta/episodes: {sorted(orphan_eps)}. The stale-cleanup pass should have removed this "
                f"file. Re-run from a fresh out_root with --overwrite-out, or report the bug."
            )

        kept_table = table.filter(pa.array(keep_mask, type=pa.bool_()))

        # Recompute index and timestamp columns on the kept rows.
        kept_eps = kept_table.column("episode_index").to_numpy(zero_copy_only=False)
        kept_fis = kept_table.column("frame_index").to_numpy(zero_copy_only=False)

        new_index = [
            plan_by_ep[int(ep)].new_dataset_from_index + int(fi)
            for ep, fi in zip(kept_eps, kept_fis, strict=True)
        ]
        new_timestamp = [int(fi) / fps for fi in kept_fis]

        idx_field = schema.field("index")
        ts_field = schema.field("timestamp")

        kept_table = kept_table.set_column(
            schema.get_field_index("index"),
            idx_field,
            pa.array(new_index, type=idx_field.type),
        )
        kept_table = kept_table.set_column(
            schema.get_field_index("timestamp"),
            ts_field,
            pa.array(new_timestamp, type=ts_field.type),
        )

        pq.write_table(kept_table, data_path, compression="snappy", use_dictionary=True)

        dropped = table.num_rows - kept_table.num_rows
        grand_total += kept_table.num_rows
        logger.info(
            "  data: %s  kept=%d  dropped=%d", rel, kept_table.num_rows, dropped
        )

    return grand_total


# --------------------------------------------------------------------------- #
# Episode metadata rewriting                                                  #
# --------------------------------------------------------------------------- #


def repair_episode_metadata(out_root: Path, plans: list[EpisodePlan], video_keys: list[str]) -> None:
    """Rewrite length, dataset_from/to_index and per-key from/to_timestamp in every
    meta/episodes/*.parquet, preserving every other column bit-for-bit."""
    plan_by_ep = {p.episode_index: p for p in plans}

    files_to_groups: dict[tuple[int, int], Path] = {}
    for ep_path in sorted((out_root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(ep_path, columns=["meta/episodes/chunk_index", "meta/episodes/file_index"])
        if table.num_rows == 0:
            continue
        chunk_idx = int(table.column("meta/episodes/chunk_index")[0].as_py())
        file_idx = int(table.column("meta/episodes/file_index")[0].as_py())
        files_to_groups[(chunk_idx, file_idx)] = ep_path

    for (chunk_idx, file_idx), ep_path in files_to_groups.items():
        rel = ep_path.relative_to(out_root)
        table = pq.read_table(ep_path)
        schema = table.schema

        # Pull the episode_index column so we know how to align with plans.
        episode_idx_col = table.column("episode_index").to_pylist()

        # Build new arrays for the columns we modify, preserving row order.
        new_length = []
        new_from_idx = []
        new_to_idx = []
        new_from_ts: dict[str, list[float]] = {k: [] for k in video_keys}
        new_to_ts: dict[str, list[float]] = {k: [] for k in video_keys}

        for ep in episode_idx_col:
            p = plan_by_ep[int(ep)]
            new_length.append(p.new_length)
            new_from_idx.append(p.new_dataset_from_index)
            new_to_idx.append(p.new_dataset_to_index)
            for key in video_keys:
                _, _, frm, to = p.new_video_offsets[key]
                new_from_ts[key].append(frm)
                new_to_ts[key].append(to)

        def _replace(table_: pa.Table, name: str, values: list) -> pa.Table:
            field = schema.field(name)
            return table_.set_column(
                schema.get_field_index(name),
                field,
                pa.array(values, type=field.type),
            )

        table = _replace(table, "length", new_length)
        table = _replace(table, "dataset_from_index", new_from_idx)
        table = _replace(table, "dataset_to_index", new_to_idx)
        for key in video_keys:
            table = _replace(table, f"videos/{key}/from_timestamp", new_from_ts[key])
            table = _replace(table, f"videos/{key}/to_timestamp", new_to_ts[key])

        pq.write_table(table, ep_path, compression="snappy", use_dictionary=True)
        logger.info("  meta: %s  episodes=%d  updated=yes", rel, table.num_rows)


def update_info_json(out_root: Path, new_total_frames: int) -> None:
    info_path = out_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    old = info.get("total_frames")
    info["total_frames"] = int(new_total_frames)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("  info: total_frames %s -> %d", old, new_total_frames)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def run_validation(out_root: Path, repo_id: str, n: int, step: int, video_keys: list[str]) -> None:
    if n <= 0:
        logger.info("Validation skipped (--validate-n=0)")
        return

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as e:  # pragma: no cover
        logger.warning("Could not import lerobot.datasets.LeRobotDataset, skipping validation: %s", e)
        return

    logger.info("Validating repaired dataset at %s ...", out_root)
    ds = LeRobotDataset(repo_id=repo_id, root=out_root)
    total = len(ds)
    n = min(n, total)
    step = max(1, int(step))

    missing = [k for k in video_keys if k not in ds.meta.video_keys]
    if missing:
        logger.warning("Repaired keys not in dataset video_keys=%s: %s", list(ds.meta.video_keys), missing)

    logger.info(
        "Loaded dataset: %d frames, %d episodes, fps=%d, video_keys=%s",
        total,
        ds.num_episodes,
        ds.fps,
        list(ds.meta.video_keys),
    )

    last_report = -10
    checked = 0
    failures: list[tuple[int, str]] = []
    for i in range(0, n, step):
        try:
            _ = ds[i]
        except Exception as e:
            failures.append((i, repr(e)))
            if len(failures) <= 3:
                logger.error("Failure at index %d: %s", i, e)
            if len(failures) >= 10:
                logger.error("Aborting validation after 10 failures.")
                break
        checked += 1
        pct = int((i + 1) / max(n, 1) * 100)
        if pct >= last_report + 10:
            logger.info("  validated %d / %d (%d%%)", i + 1, n, pct)
            last_report = pct

    if failures:
        raise SystemExit(
            f"Validation FAILED: {len(failures)} failure(s) over {checked} checks. "
            f"First failure: index={failures[0][0]} error={failures[0][1]}"
        )

    logger.info("Validation OK: %d items iterated through without errors.", checked)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def _load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def _detect_stale_metadata(src_root: Path, total_episodes: int) -> tuple[Path | None, int, list[Path]]:
    """Read-only scan of src_root's meta/episodes. Returns (canonical_path, total_rows, all_paths).

    canonical_path is None when nothing matches info.json:total_episodes. Stale leftovers
    are diagnosed by ``total_rows > total_episodes``.
    """
    ep_files = sorted((src_root / "meta" / "episodes").glob("*/*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No meta/episodes/*/*.parquet under {src_root}")

    total_rows = 0
    canonical: Path | None = None
    for p in ep_files:
        eps = _read_episode_index_sequence(p)
        total_rows += len(eps)
        if (
            canonical is None
            and len(eps) == total_episodes
            and sorted(eps) == list(range(total_episodes))
        ):
            canonical = p

    return canonical, total_rows, ep_files


def main() -> int:
    args = parse_args()
    setup_logging(args.quiet)

    out_root: Path = args.out_root.expanduser().resolve()

    src_root = materialize_source(args.repo_id, args.src_root, args.revision)
    info = _load_info(src_root)
    fps = args.fps if args.fps is not None else int(info["fps"])
    total_episodes = int(info["total_episodes"])

    declared_video_keys = sorted(
        k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"
    )
    if not declared_video_keys:
        raise SystemExit("Dataset has no video features in info.json; nothing to repair.")
    video_keys = list(args.video_keys) if args.video_keys else declared_video_keys
    unknown = [k for k in video_keys if k not in declared_video_keys]
    if unknown:
        raise SystemExit(
            f"--video-keys references keys not in dataset: {unknown}. "
            f"Available video keys: {declared_video_keys}"
        )
    logger.info(
        "Operating on video_keys=%s, fps=%d, info.total_episodes=%d, info.total_frames=%d",
        video_keys,
        fps,
        total_episodes,
        int(info.get("total_frames", -1)),
    )

    # ── Stale-leftover detection (read-only on src_root) ──────────────
    canonical_meta, total_meta_rows, all_meta_files = _detect_stale_metadata(src_root, total_episodes)
    has_stale_leftovers = total_meta_rows != total_episodes

    if has_stale_leftovers:
        logger.warning(
            "Stale leftover detected: info.json:total_episodes=%d but meta/episodes contains "
            "%d rows across %d file(s). The source dataset has orphan files from a previous "
            "`delete_episodes` + `push_to_hub` round trip.",
            total_episodes,
            total_meta_rows,
            len(all_meta_files),
        )
        if canonical_meta is None:
            raise SystemExit(
                f"No meta/episodes file has exactly {total_episodes} rows with episode_index=="
                f"[0..N-1]. Cannot determine canonical metadata. Aborting -- inspect the dataset "
                f"by hand or re-push the pruned dataset to a fresh repo_id."
            )
        logger.warning(
            "Canonical metadata file is %s. The cleanup phase will delete the other "
            "%d meta/episodes file(s) plus orphan data/ and videos/ files in --out-root.",
            canonical_meta.relative_to(src_root),
            len(all_meta_files) - 1,
        )

    # ── Diagnostic (uses canonical metadata when present) ─────────────
    if has_stale_leftovers:
        # Load only the canonical metadata so the diagnostic represents the post-cleanup state.
        table = pq.read_table(canonical_meta)
        episode_rows = [
            {name: table.column(name)[i].as_py() for name in table.schema.names}
            for i in range(table.num_rows)
        ]
        episode_rows.sort(key=lambda r: r["episode_index"])
    else:
        episode_rows = load_episode_metadata(src_root)

    plans = build_plans(episode_rows, video_keys, fps)
    print_diagnostic(plans, video_keys, fps)

    if args.dry_run:
        if has_stale_leftovers:
            logger.info(
                "Dry run: would also delete %d stale meta/episodes file(s) and any orphan "
                "data/videos files in --out-root if you re-run without --dry-run.",
                len(all_meta_files) - 1,
            )
        logger.info("Dry run requested -- not modifying anything. Exiting.")
        return 0

    populate_out_root(src_root, out_root, link_videos=args.link_videos, overwrite=args.overwrite_out)

    # ── Stale-leftover cleanup (writes only inside out_root) ──────────
    if has_stale_leftovers and not args.no_cleanup:
        out_canonical = out_root / canonical_meta.relative_to(src_root)
        data_refs, video_refs = referenced_data_and_video_files(out_canonical, video_keys)
        if args.link_videos:
            # Symlinked videos can't be safely modified (would mutate the source snapshot).
            logger.warning(
                "--link-videos was used; stale video files under %s will be left in place "
                "(can't delete through the symlink without touching the source snapshot). "
                "Re-run without --link-videos if you need a fully clean local copy.",
                src_root / "videos",
            )
            video_refs_for_cleanup: dict[str, set[tuple[int, int]]] = {}
        else:
            video_refs_for_cleanup = video_refs
        logger.info("Cleaning up stale leftover files in %s ...", out_root)
        report = cleanup_stale_files(out_root, out_canonical, data_refs, video_refs_for_cleanup)
        logger.info(
            "Cleanup: deleted %d meta file(s), %d data file(s), %d video file(s)",
            len(report["deleted_meta"]),
            len(report["deleted_data"]),
            sum(len(v) for v in report["deleted_videos"].values()),
        )
        for rel in report["deleted_meta"]:
            logger.info("  deleted meta: %s", rel)
        for rel in report["deleted_data"]:
            logger.info("  deleted data: %s", rel)
        for key, rels in report["deleted_videos"].items():
            for rel in rels:
                logger.info("  deleted video[%s]: %s", key, rel)
    elif has_stale_leftovers and args.no_cleanup:
        logger.warning(
            "--no-cleanup set; skipping stale-leftover cleanup. The dataset will likely fail "
            "validation because duplicate episode_index values will trip the uniqueness guard."
        )

    logger.info("Truncating data parquet files ...")
    new_total = repair_data_parquets(out_root, plans, fps)

    logger.info("Rewriting episode metadata ...")
    repair_episode_metadata(out_root, plans, video_keys)

    update_info_json(out_root, new_total)

    logger.info(
        "Repair complete: total_frames %d -> %d (dropped %d rows across %d episodes)",
        sum(p.old_length for p in plans),
        new_total,
        sum(p.deficit for p in plans),
        sum(1 for p in plans if p.deficit > 0),
    )

    run_validation(
        out_root=out_root,
        repo_id=args.repo_id,
        n=args.validate_n,
        step=args.validate_step,
        video_keys=video_keys,
    )

    logger.info("Done. Repaired dataset written to %s", out_root)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Repair failed: %s", e)
        sys.exit(1)
