#!/usr/bin/env python3
"""Build a human-video support bank for in-context pi0/pi0.5 training.

Input layout per task/config:
  <raw_data_root>/<task_name>/<task_config>/human_video/
    human_demo_000/
      left.mp4
      center.mp4
      right.mp4
      ego.mp4
      caption.txt

Output layout:
  <support_bank_root>/human/<task_name>/<task_config>/human_demo_000/
    caption.txt
    center/
      frames.npy     # [K, resize, resize, 3], uint8, RGB
      frame_000.png  # optional visual check
      ...
      meta.json

Only human-video is supported in this version. Robot-video support was intentionally
removed from this data flow.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HUMAN_VIEWS = ("left", "center", "right", "ego")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def read_caption(demo_dir: Path) -> str:
    path = demo_dir / "caption.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def sample_video(video_path: Path, num_frames: int, resize: int) -> tuple[np.ndarray, list[int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: decode all frames if metadata is unavailable.
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")
        total = len(frames)
        idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
        return np.asarray([frames[i] for i in idx], dtype=np.uint8), idx, total

    indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    cap.release()
    return np.asarray(frames, dtype=np.uint8), indices, total


def save_png_frames(frames: np.ndarray, out_dir: Path) -> None:
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"frame_{i:03d}.png"), frame_bgr)


def process_demo(
    *,
    task_name: str,
    task_config: str,
    demo_dir: Path,
    out_demo_dir: Path,
    views: list[str],
    num_frames: int,
    resize: int,
    save_png: bool,
    strict: bool,
) -> int:
    caption = read_caption(demo_dir)
    out_demo_dir.mkdir(parents=True, exist_ok=True)
    (out_demo_dir / "caption.txt").write_text(caption + ("\n" if caption else ""), encoding="utf-8")

    processed = 0
    for view in views:
        video_path = demo_dir / f"{view}.mp4"
        if not video_path.exists():
            msg = f"Missing video: {video_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}; skip")
            continue

        out_view_dir = out_demo_dir / view
        out_view_dir.mkdir(parents=True, exist_ok=True)
        frames, sampled_indices, total = sample_video(video_path, num_frames=num_frames, resize=resize)
        np.save(out_view_dir / "frames.npy", frames)
        if save_png:
            save_png_frames(frames, out_view_dir)

        progress = [float(i / max(total - 1, 1)) for i in sampled_indices]
        meta = {
            "support_type": "human",
            "task_name": task_name,
            "task_config": task_config,
            "support_id": demo_dir.name,
            "view": view,
            "source_video": str(video_path),
            "caption_path": str(out_demo_dir / "caption.txt"),
            "num_original_frames": total,
            "num_support_frames": int(num_frames),
            "resize": int(resize),
            "sampled_indices": sampled_indices,
            "progress": progress,
            "frames_npy": str(out_view_dir / "frames.npy"),
        }
        (out_view_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--support-bank-root", required=True, type=Path)
    parser.add_argument("--task-names", default=None, help="Comma separated task names. If omitted, scan all tasks.")
    parser.add_argument("--task-configs", required=True, help="Comma separated configs, e.g. demo_clean,demo_randomized")
    parser.add_argument("--views", default="left,center,right,ego")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--save-png", action="store_true", help="Also save frame_*.png for visual inspection.")
    parser.add_argument("--strict", action="store_true", help="Error on missing captions/videos instead of skipping.")
    args = parser.parse_args()

    task_names = parse_csv(args.task_names)
    if task_names is None:
        task_names = sorted([p.name for p in args.raw_data_root.iterdir() if p.is_dir()])
    task_configs = parse_csv(args.task_configs) or []
    views = parse_csv(args.views) or list(HUMAN_VIEWS)
    bad_views = sorted(set(views) - set(HUMAN_VIEWS))
    if bad_views:
        raise ValueError(f"Unsupported human views: {bad_views}; allowed={HUMAN_VIEWS}")

    total_processed = 0
    for task_name in task_names:
        for task_config in task_configs:
            human_root = args.raw_data_root / task_name / task_config / "human_video"
            if not human_root.exists():
                print(f"[WARN] No human_video dir: {human_root}; skip")
                continue
            demo_dirs = sorted([p for p in human_root.iterdir() if p.is_dir() and p.name.startswith("human_demo_")])
            if not demo_dirs:
                print(f"[WARN] No human_demo_* under {human_root}; skip")
                continue
            for demo_dir in demo_dirs:
                out_demo_dir = args.support_bank_root / "human" / task_name / task_config / demo_dir.name
                n = process_demo(
                    task_name=task_name,
                    task_config=task_config,
                    demo_dir=demo_dir,
                    out_demo_dir=out_demo_dir,
                    views=views,
                    num_frames=args.num_frames,
                    resize=args.resize,
                    save_png=args.save_png,
                    strict=args.strict,
                )
                total_processed += n
                print(f"[OK] {task_name}/{task_config}/{demo_dir.name}: processed {n} views")
    print(f"[DONE] processed support views: {total_processed}")


if __name__ == "__main__":
    main()
