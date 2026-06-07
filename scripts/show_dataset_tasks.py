#!/usr/bin/env python
"""Show the task name(s) for one or more LeRobot datasets.

Uses ``LeRobotDatasetMetadata`` (lightweight: reads ``meta/`` only, no episode
data) to load each dataset's tasks and print them, flagging any dataset that
does not have exactly one task.

Examples:
    # Inspect a single dataset by local root
    python scripts/show_dataset_tasks.py --root /home/kuphdev/lerobot_datasets/place_yellow_duck_pruned

    # Inspect a single dataset by Hub repo id (downloads meta/ if needed)
    python scripts/show_dataset_tasks.py --repo-id KuphDev/place_yellow_duck_pruned

    # Scan every dataset under a directory (default shown)
    python scripts/show_dataset_tasks.py --datasets-dir /home/kuphdev/lerobot_datasets
"""

import argparse
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

DEFAULT_DATASETS_DIR = Path.home() / "lerobot_datasets"


def get_tasks(repo_id: str, root: Path | None) -> list[str]:
    """Return the list of task strings for a dataset."""
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(root) if root is not None else None)
    # meta.tasks is a DataFrame indexed by the task string, with a `task_index` column.
    return meta.tasks.index.tolist()


def report(name: str, repo_id: str, root: Path | None) -> bool:
    """Print the tasks for one dataset. Returns True if it has exactly one task."""
    try:
        tasks = get_tasks(repo_id, root)
    except Exception as e:  # noqa: BLE001 - we want to keep scanning other datasets
        print(f"ERR  {name}: {e}")
        return False

    single = len(tasks) == 1
    flag = "OK " if single else ">>> MULTIPLE"
    print(f"{flag} [{len(tasks)} task(s)] {name}")
    for t in tasks:
        print(f"        - {t!r}")
    return single


def find_local_datasets(datasets_dir: Path) -> list[Path]:
    """Return sub-directories of `datasets_dir` that look like LeRobot datasets."""
    if not datasets_dir.is_dir():
        return []
    return sorted(p for p in datasets_dir.iterdir() if (p / "meta" / "tasks.parquet").is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="Local root of a single dataset to inspect.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hub repo id of a single dataset (used for the lookup and as a label).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Directory whose sub-folders are scanned for datasets (default: {DEFAULT_DATASETS_DIR}).",
    )
    args = parser.parse_args()

    # Single-dataset mode: triggered when --root and/or --repo-id is given.
    if args.root is not None or args.repo_id is not None:
        root = args.root
        # LeRobotDatasetMetadata requires a repo_id; fall back to the folder name for local datasets.
        repo_id = args.repo_id or (root.name if root is not None else None)
        if repo_id is None:
            parser.error("Provide --repo-id when --root is omitted.")
        ok = report(repo_id, repo_id, root)
        return 0 if ok else 1

    # Scan mode.
    datasets = find_local_datasets(args.datasets_dir)
    if not datasets:
        print(f"No datasets (with meta/tasks.parquet) found under {args.datasets_dir}")
        return 1

    print(f"Scanning {len(datasets)} dataset(s) under {args.datasets_dir}\n")
    all_single = True
    for ds_root in datasets:
        # For local datasets the repo_id is only used as a label/version-check string.
        all_single &= report(ds_root.name, ds_root.name, ds_root)

    print()
    if all_single:
        print("All datasets have exactly one task.")
    else:
        print("Some datasets do NOT have exactly one task (see lines marked '>>> MULTIPLE').")
    return 0 if all_single else 1


if __name__ == "__main__":
    sys.exit(main())
