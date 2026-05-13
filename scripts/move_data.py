#!/usr/bin/env python3
"""
Prepare multi-task ALOHA-style HDF5 training_data for iclpi.

This script copies/symlinks processed_data outputs into:

training_data/<model_name>/
  ├── <task_0>/episode_0 ... episode_549
  ├── <task_1>/episode_0 ... episode_549
  └── episode_origin_local.jsonl

It also records the exact source of every local episode, which is later consumed
by convert_aloha_data_to_lerobot_robotwin.py to create meta/episode_origin.jsonl.

Default rule:
  demo_clean      episode_0..49   -> local episode_0..49
  demo_randomized episode_0..499  -> local episode_50..549
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

DEFAULT_TASKS = [
    "click_alarmclock",
    "press_stapler",
    "open_laptop",
    "rotate_qrcode",
    "handover_mic",
    "pick_dual_bottles",
    "grab_roller",
    "move_stapler_pad",
    "place_empty_cup",
    "place_bread_basket",
    "place_a2b_left",
    "stack_blocks_two",
    "place_fan",
]

EPISODE_RE = re.compile(r"^episode_(\d+)$")


def parse_tasks(tasks: str | None) -> list[str]:
    if not tasks:
        return DEFAULT_TASKS
    return [t.strip() for t in tasks.split(",") if t.strip()]


def iter_episodes(task_dir: Path) -> list[tuple[int, Path]]:
    episodes: list[tuple[int, Path]] = []
    if not task_dir.exists():
        raise FileNotFoundError(f"Processed task dir not found: {task_dir}")
    for p in task_dir.iterdir():
        if not p.is_dir():
            continue
        m = EPISODE_RE.match(p.name)
        if m:
            episodes.append((int(m.group(1)), p))
    episodes.sort(key=lambda x: x[0])
    return episodes


def copy_or_link(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="source_data")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names. Default: 13 source tasks.")
    parser.add_argument("--clean-config", type=str, default="demo_clean")
    parser.add_argument("--clean-num", type=int, default=50)
    parser.add_argument("--randomized-config", type=str, default="demo_randomized")
    parser.add_argument("--randomized-num", type=int, default=500)
    parser.add_argument("--randomized-offset", type=int, default=50)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    out_root = args.training_root / args.model_name
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for task in tasks:
        print(f"[TASK] {task}")
        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)

        specs = [
            {
                "task_config": args.clean_config,
                "expert_num": args.clean_num,
                "offset": 0,
                "expected_num": args.clean_num,
            },
            {
                "task_config": args.randomized_config,
                "expert_num": args.randomized_num,
                "offset": args.randomized_offset,
                "expected_num": args.randomized_num,
            },
        ]

        for spec in specs:
            src_task_dir = args.processed_root / f"{task}-{spec['task_config']}-{spec['expert_num']}"
            episodes = iter_episodes(src_task_dir)
            if len(episodes) != spec["expected_num"]:
                raise RuntimeError(
                    f"{src_task_dir}: expected {spec['expected_num']} episodes, got {len(episodes)}"
                )

            for src_ep_idx, src_ep_dir in episodes:
                local_ep_idx = src_ep_idx + int(spec["offset"])
                dst_ep_dir = task_out / f"episode_{local_ep_idx}"
                print(f"  {spec['task_config']}: episode_{src_ep_idx} -> {task}/episode_{local_ep_idx}")
                copy_or_link(src_ep_dir, dst_ep_dir, args.mode, args.overwrite)

                records.append(
                    {
                        "task_name": task,
                        "local_episode_index": local_ep_idx,
                        "task_config": spec["task_config"],
                        "source_episode_index": src_ep_idx,
                        "source_episode_dir": str(src_ep_dir),
                        "training_episode_dir": str(dst_ep_dir),
                    }
                )

    records.sort(key=lambda r: (r["task_name"], r["local_episode_index"]))
    manifest_path = out_root / "episode_origin_local.jsonl"
    write_jsonl(manifest_path, records)
    print(f"[OK] Wrote local origin manifest: {manifest_path}")
    print(f"[OK] Total episodes: {len(records)}")


if __name__ == "__main__":
    main()
