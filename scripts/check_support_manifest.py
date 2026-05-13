#!/usr/bin/env python3
"""Validate human support manifest and support bank files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-support-rounds", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--check-npy-shape", action="store_true", help="Load every frames.npy and verify shape/dtype. Slower but safer.")
    args = parser.parse_args()

    origins = load_jsonl(args.lerobot_repo / "meta" / "episode_origin.jsonl")
    records = load_jsonl(args.manifest)
    expected_eps = {int(r["global_episode_index"]) for r in origins}

    by_ep = defaultdict(list)
    support_counter = Counter()
    view_counter = Counter()

    errors: list[str] = []
    for r in records:
        ep = int(r["global_episode_index"])
        round_id = int(r["support_round_id"])
        by_ep[ep].append(round_id)

        npy_path = Path(r["support_frames_npy"])
        if not npy_path.exists():
            errors.append(f"missing frames.npy: {npy_path}")
        elif args.check_npy_shape:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.dtype != np.uint8:
                errors.append(f"bad dtype {arr.dtype}: {npy_path}")
            if arr.ndim != 4 or arr.shape[0] != args.num_frames or arr.shape[-1] != 3:
                errors.append(f"bad shape {arr.shape}: {npy_path}")

        progress = r.get("support_frame_progress", [])
        if len(progress) != args.num_frames:
            errors.append(f"bad progress length for ep={ep}, round={round_id}: {len(progress)}")
        if progress and (min(progress) < -1e-6 or max(progress) > 1 + 1e-6):
            errors.append(f"progress out of range for ep={ep}, round={round_id}: {progress}")

        caption = str(r.get("support_caption", "")).strip()
        if not caption:
            errors.append(f"empty caption for ep={ep}, round={round_id}")
        caption_path = Path(r.get("support_caption_path", ""))
        if not caption_path.exists():
            errors.append(f"missing caption_path for ep={ep}, round={round_id}: {caption_path}")

        support_counter[(r.get("task_name"), r.get("task_config"), r.get("support_id"))] += 1
        view_counter[r.get("support_view")] += 1

    missing_eps = expected_eps - set(by_ep)
    if missing_eps:
        errors.append(f"missing episodes in manifest: first={sorted(missing_eps)[:20]}, count={len(missing_eps)}")

    if args.num_support_rounds is not None:
        expected_rounds = set(range(args.num_support_rounds))
        for ep in expected_eps:
            got = set(by_ep.get(ep, []))
            if got != expected_rounds:
                errors.append(f"bad rounds for ep={ep}: got={sorted(got)}, expected={sorted(expected_rounds)}")
                if len(errors) > 50:
                    break

    print(f"[INFO] origins episodes: {len(expected_eps)}")
    print(f"[INFO] manifest records: {len(records)}")
    print(f"[INFO] view counts: {dict(view_counter)}")
    print(f"[INFO] unique support ids: {len(support_counter)}")

    if errors:
        print("[ERROR] manifest validation failed:")
        for e in errors[:100]:
            print("  -", e)
        raise SystemExit(1)
    print("[OK] support manifest validation passed")


if __name__ == "__main__":
    main()
