"""Derive a FineART-VLA training copy of the Astribot serving-brewed-coffee dataset.

The source dataset is left byte-identical (OpenTau trains on it too). The derived copy at ``--dst``
holds its own ``meta/`` and ``data/`` (~50 MB) and a symlink to the source's ``videos/``:

1. ``language_persistent`` / ``language_events`` columns, built from the per-frame ``response``
   captions: one ``style=subtask`` row per caption segment, stamped with the segment's first frame
   timestamp, broadcast to every frame of the episode (the layout ``lerobot-annotate`` writes).
   Captions get light normalization (whitespace, capitalized first letter, one trailing period);
   adjacent segments that become identical are merged.
2. Flat ``observation.state`` / ``action`` names. The source stores them as a nested list, which
   LeRobot's relative-action mask reads as one name, so ``relative_exclude_joints=["gripper"]`` would
   silently make the grippers relative too.
3. ``meta/stats.json``: the ``action`` entry is replaced by stats in relative space (actions minus the
   chunk-start state for the 14 arm joints, grippers absolute) over every valid ``--chunk_size`` chunk.
   All other entries are kept from the source. Training reads normalization stats from this file.

Usage (once per machine, CPU only, a few minutes):
    uv run python custom_astribot/make_fineart_dataset.py --src <dataset> --dst <dataset>_fineart
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter, _normalize_persistent_row
from lerobot.datasets.compute_stats import compute_relative_action_stats
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import language_feature_info
from lerobot.processor import RelativeActionsProcessorStep

CAPTION_COLUMN = "response"
EXCLUDE_JOINTS = ["gripper"]


def normalize_caption(text: str) -> str:
    """Collapse whitespace, capitalize the first letter and end with exactly one period."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        raise ValueError("empty subtask caption")
    return (text[0].upper() + text[1:]).rstrip(". ") + "."


def flat_names(names: object) -> list[str]:
    """Unwrap a nested ``[[...]]`` feature-names list into a flat list of strings."""
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
        raise ValueError(f"unexpected feature names layout: {names!r}")
    return list(names)


def subtask_rows(episodes: np.ndarray, timestamps: np.ndarray, captions: list[str]) -> dict[int, list[dict]]:
    """One persistent subtask row per caption segment, keyed by episode."""
    rows: dict[int, list[dict]] = {}
    prev_ep, prev_text = None, None
    for ep, ts, raw in zip(episodes.tolist(), timestamps.tolist(), captions, strict=True):
        text = normalize_caption(raw)
        if ep != prev_ep:
            rows.setdefault(ep, [])
            prev_text = None
        if text != prev_text:
            row = {"role": "assistant", "content": text, "style": "subtask", "timestamp": float(ts)}
            rows[ep].append(_normalize_persistent_row(row))
        prev_ep, prev_text = ep, text
    return rows


def main() -> None:
    """Build the derived dataset at ``--dst`` from ``--src``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=50, help="must match policy.chunk_size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=">1 is faster but makes the quantiles depend on thread order",
    )
    args = parser.parse_args()
    src, dst = args.src.resolve(), args.dst.absolute()

    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"{src} is not a LeRobot dataset")
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"{dst} already exists; remove it first to rebuild")

    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    shutil.copytree(src / "data", dst / "data")
    (dst / "videos").symlink_to(src / "videos", target_is_directory=True)

    # --- info.json: flat names + language features -------------------------------------------
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = flat_names(info["features"][key]["names"])
    names = info["features"]["action"]["names"]
    mask = RelativeActionsProcessorStep(enabled=True, exclude_joints=EXCLUDE_JOINTS, action_names=names)
    mask = mask._build_mask(len(names))
    if mask != [True] * 14 + [False] * 2:
        raise SystemExit(f"unexpected relative-action mask {mask} for names {names}")
    info["features"].update(language_feature_info())
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    # --- language columns + gather arrays for stats ------------------------------------------
    writer = LanguageColumnsWriter()
    all_ep, all_state, all_action = [], [], []
    n_rows = n_segments = 0
    distinct: set[str] = set()
    for path in sorted((dst / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        ep = np.asarray(table.column("episode_index").to_numpy())
        ts = np.asarray(table.column("timestamp").to_numpy())
        frame = np.asarray(table.column("frame_index").to_numpy())
        order_ok = all(
            np.all(np.diff(frame[ep == e]) == 1) and np.all(np.diff(ts[ep == e]) > 0) for e in np.unique(ep)
        )
        if not order_ok:
            raise SystemExit(f"{path}: frames are not contiguous and time-ordered within episodes")
        rows_by_ep = subtask_rows(ep, ts, table.column(CAPTION_COLUMN).to_pylist())
        persistent = [rows_by_ep[e] for e in ep.tolist()]
        events: list[list[dict]] = [[] for _ in range(len(ep))]
        new_table = writer._materialize_table(table, persistent, events, drop_old=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        write_table_one_row_group_per_episode(new_table, tmp)
        tmp.replace(path)

        n_rows += len(ep)
        n_segments += sum(len(r) for r in rows_by_ep.values())
        distinct.update(r["content"] for rs in rows_by_ep.values() for r in rs)
        all_ep.append(ep)
        all_state.append(np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)))
        all_action.append(np.stack(table.column("action").to_numpy(zero_copy_only=False)))

    episodes = np.concatenate(all_ep)
    state = np.concatenate(all_state).astype(np.float32)
    action = np.concatenate(all_action).astype(np.float32)
    if n_rows != info["total_frames"]:
        raise SystemExit(f"wrote {n_rows} rows but info.json says {info['total_frames']} frames")

    # --- stats.json: relative-space action stats, everything else kept ----------------------
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    rel = compute_relative_action_stats(
        hf_dataset={"action": action, "observation.state": state, "episode_index": episodes},
        features=info["features"],
        chunk_size=args.chunk_size,
        exclude_joints=EXCLUDE_JOINTS,
        num_workers=args.num_workers,
    )
    stats["action"] = {k: np.asarray(v).tolist() for k, v in rel.items()}
    # Sanity-check the kept state quantiles against the data (pi05 normalizes state with q01/q99).
    for q, p in (("q01", 1), ("q99", 99)):
        err = np.abs(np.percentile(state, p, axis=0) - np.asarray(stats["observation.state"][q])).max()
        if err > 1e-2:
            raise SystemExit(
                f"source observation.state {q} is off by {err:.3f} from the data; fix stats first"
            )
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    (dst / "meta" / "fineart_conversion.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "caption_column": CAPTION_COLUMN,
                "caption_normalization": "collapse whitespace, capitalize first letter, one trailing period",
                "subtask_segments": n_segments,
                "distinct_subtasks": len(distinct),
                "relative_action_stats": {"chunk_size": args.chunk_size, "exclude_joints": EXCLUDE_JOINTS},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {dst}: {n_rows} frames, {n_segments} subtask segments, {len(distinct)} distinct subtasks")


if __name__ == "__main__":
    main()
