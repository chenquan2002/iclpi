#!/usr/bin/env python3
"""Build a support manifest for human-video in-context training.

The manifest is generated at data-processing time. Training only does lookup by:
  (global_episode_index, support_round_id)

This solves two issues:
  1. LeRobot mixed repos contain many tasks, so global episode_index must be mapped
     through meta/episode_origin.jsonl.
  2. If training uses fewer rounds than available videos, each trajectory still sees
     a deterministic shuffled subset rather than always demo_000, demo_001, ...

Supported view modes:
  center/left/right/ego: candidates are M demos with that fixed view.
  random: candidates are all M demos × 4 views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

VIEWS_ALL = ("left", "center", "right", "ego")
FIXED_VIEW_MODES = set(VIEWS_ALL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_u32(*parts: Any, seed: int = 0) -> int:
    s = "||".join(str(p) for p in parts) + f"||seed={seed}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def candidate_sort_key(c: dict[str, Any]) -> tuple[int, str]:
    # human_demo_003 -> 3
    sid = c["support_id"]
    try:
        n = int(str(sid).split("_")[-1])
    except Exception:
        n = 10**9
    return n, c["support_view"]


def build_candidates(
    *,
    support_bank_root: Path,
    task_name: str,
    task_config: str,
    view_mode: str,
    num_human_demos: int | None,
) -> list[dict[str, Any]]:
    root = support_bank_root / "human" / task_name / task_config
    if not root.exists():
        raise FileNotFoundError(f"Missing support bank task/config root: {root}")

    if view_mode == "random":
        views = list(VIEWS_ALL)
    elif view_mode in FIXED_VIEW_MODES:
        views = [view_mode]
    else:
        raise ValueError(f"Unsupported view_mode={view_mode}; use {sorted(FIXED_VIEW_MODES | {'random'})}")

    demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("human_demo_")])
    if num_human_demos is not None:
        expected = {f"human_demo_{i:03d}" for i in range(num_human_demos)}
        demo_dirs = [p for p in demo_dirs if p.name in expected]

    candidates: list[dict[str, Any]] = []
    for demo_dir in demo_dirs:
        caption_path = demo_dir / "caption.txt"
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        for view in views:
            view_dir = demo_dir / view
            meta_path = view_dir / "meta.json"
            frames_npy = view_dir / "frames.npy"
            if not meta_path.exists() or not frames_npy.exists():
                continue
            meta = read_json(meta_path)
            candidates.append(
                {
                    "support_type": "human",
                    "support_id": demo_dir.name,
                    "support_view": view,
                    "support_frames_npy": str(frames_npy),
                    "support_caption": caption,
                    "support_caption_path": str(caption_path),
                    "support_frame_progress": meta.get("progress", []),
                }
            )
    candidates.sort(key=candidate_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No support candidates for {task_name}/{task_config}, view_mode={view_mode}, root={root}")
    return candidates


def default_rounds(view_mode: str, candidates: list[dict[str, Any]]) -> int:
    # If view=random, candidates are demo-view pairs. Otherwise candidates are demos.
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-repo", required=True, type=Path, help="Path containing meta/episode_origin.jsonl")
    parser.add_argument("--support-bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--view-mode", default="center", choices=["center", "left", "right", "ego", "random"])
    parser.add_argument("--num-human-demos", type=int, default=10)
    parser.add_argument("--num-support-rounds", type=int, default=None,
                        help="If omitted, use number of candidates: 10 for fixed view, 40 for random if M=10.")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    origin_path = args.lerobot_repo / "meta" / "episode_origin.jsonl"
    if not origin_path.exists():
        raise FileNotFoundError(f"Missing episode origin: {origin_path}")
    origins = load_jsonl(origin_path)

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for origin in origins:
        task_name = origin["task_name"]
        task_config = origin["task_config"]
        key = (task_name, task_config)
        if key not in cache:
            cache[key] = build_candidates(
                support_bank_root=args.support_bank_root,
                task_name=task_name,
                task_config=task_config,
                view_mode=args.view_mode,
                num_human_demos=args.num_human_demos,
            )
            print(f"[INFO] {task_name}/{task_config}: {len(cache[key])} candidates for view_mode={args.view_mode}")
        candidates = cache[key]
        rounds = args.num_support_rounds or default_rounds(args.view_mode, candidates)

        # Episode-specific deterministic permutation. Even if training uses only early rounds,
        # different trajectories will use different demo/view subsets.
        rng_seed = stable_u32(
            task_name,
            task_config,
            origin.get("source_episode_index", origin.get("local_episode_index", 0)),
            origin["global_episode_index"],
            seed=args.shuffle_seed,
        )
        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(len(candidates)).tolist()

        for round_id in range(rounds):
            c = candidates[perm[round_id % len(candidates)]]
            record = {
                "global_episode_index": int(origin["global_episode_index"]),
                "support_round_id": int(round_id),
                "task_name": task_name,
                "task_config": task_config,
                "local_episode_index": int(origin.get("local_episode_index", -1)),
                "source_episode_index": int(origin.get("source_episode_index", -1)),
                "episode_length": int(origin["episode_length"]),
                "view_mode": args.view_mode,
                **c,
            }
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(records)} support records: {args.output}")


if __name__ == "__main__":
    main()
