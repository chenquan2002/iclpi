"""Support-context data loading utilities for human-video in-context training.

This module intentionally keeps support frames/captions separate from robot observation.
It does not put support images into data["image"].

Typical use:
  base_dataset = LeRobotDataset(...)
  dataset = SupportRoundDataset(base_dataset, support_rounds_per_cycle=10)
  transform = AddSupportContext(...)

The training pipeline still needs to integrate this transform before Observation.from_dict,
and then tokenize support_caption separately.
"""
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _to_int(x: Any) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    return int(x)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class SupportManifest:
    """Lookup episode origin and support records.

    support_manifest keys are (global_episode_index, support_round_id). If the requested
    support_round_id is larger than available rounds, it is wrapped by modulo per episode.
    """

    def __init__(self, episode_origin_path: str | Path, support_manifest_path: str | Path):
        self.origin: dict[int, dict[str, Any]] = {}
        for r in load_jsonl(episode_origin_path):
            self.origin[int(r["global_episode_index"])] = r

        self.support: dict[tuple[int, int], dict[str, Any]] = {}
        self.rounds_by_ep: dict[int, list[int]] = defaultdict(list)
        for r in load_jsonl(support_manifest_path):
            ep = int(r["global_episode_index"])
            rd = int(r["support_round_id"])
            self.support[(ep, rd)] = r
            self.rounds_by_ep[ep].append(rd)
        for ep in list(self.rounds_by_ep):
            self.rounds_by_ep[ep] = sorted(set(self.rounds_by_ep[ep]))

    def get_origin(self, global_episode_index: int) -> dict[str, Any]:
        return self.origin[global_episode_index]

    def get_support(self, global_episode_index: int, support_round_id: int = 0) -> dict[str, Any]:
        rounds = self.rounds_by_ep.get(global_episode_index)
        if not rounds:
            raise KeyError(f"No support records for global_episode_index={global_episode_index}")
        # Manifest is usually 0..R-1. Wrap to support arbitrary training length.
        rd = rounds[support_round_id % len(rounds)]
        return self.support[(global_episode_index, rd)]


class SupportFrameCache:
    """Small LRU cache for frames.npy arrays."""

    def __init__(self, max_items: int = 1024):
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, frames_npy: str | Path) -> np.ndarray:
        key = str(frames_npy)
        if key in self.cache:
            arr = self.cache.pop(key)
            self.cache[key] = arr
            return arr
        arr = np.load(key)  # [K, H, W, 3], uint8, RGB
        if self.max_items > 0:
            while len(self.cache) >= self.max_items:
                self.cache.popitem(last=False)
            self.cache[key] = arr
        return arr


class SupportRoundDataset:
    """Dataset wrapper that creates support_round_id without needing real epoch info.

    base_idx = idx % len(base_dataset)
    support_round_id = idx // len(base_dataset)

    If training lasts longer than one full wrapper pass, the DataLoader will restart and
    support_round_id sequence repeats. AddSupportContext also wraps round ids by modulo.
    """

    def __init__(self, base_dataset: Any, support_rounds_per_cycle: int):
        self.base_dataset = base_dataset
        self.base_len = len(base_dataset)
        self.support_rounds_per_cycle = int(support_rounds_per_cycle)
        if self.support_rounds_per_cycle <= 0:
            raise ValueError("support_rounds_per_cycle must be positive")

    def __len__(self) -> int:
        return self.base_len * self.support_rounds_per_cycle

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = int(idx) % self.base_len
        support_round_id = int(idx) // self.base_len
        item = self.base_dataset[base_idx]
        # Make a shallow dict copy so we can add metadata without mutating base item.
        item = dict(item)
        item["support_round_id"] = support_round_id
        return item


class AddSupportContext:
    """Transform that injects support_images/caption/progress into a data dict.

    Required input fields after AlohaInputsPreserveMeta or equivalent:
      episode_index, frame_index, optional support_round_id

    Added fields:
      support_images: [K, H, W, 3] uint8 RGB
      support_image_mask: [K] bool
      support_frame_progress: [K] float32
      chunk_progress: [1] float32
      support_caption: str
    """

    def __init__(
        self,
        episode_origin_path: str | Path,
        support_manifest_path: str | Path,
        cache_size: int = 1024,
    ):
        self.manifest = SupportManifest(episode_origin_path, support_manifest_path)
        self.cache = SupportFrameCache(max_items=cache_size)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        global_ep = _to_int(data["episode_index"])
        frame_idx = _to_int(data["frame_index"])
        support_round_id = _to_int(data.get("support_round_id", 0))

        origin = self.manifest.get_origin(global_ep)
        support = self.manifest.get_support(global_ep, support_round_id)

        frames = self.cache.get(support["support_frames_npy"])
        progress = np.asarray(support["support_frame_progress"], dtype=np.float32)
        episode_length = int(origin["episode_length"])
        chunk_progress = frame_idx / max(episode_length - 1, 1)

        data["support_images"] = frames
        data["support_image_mask"] = np.ones((frames.shape[0],), dtype=bool)
        data["support_frame_progress"] = progress
        data["chunk_progress"] = np.asarray([chunk_progress], dtype=np.float32)
        data["support_caption"] = str(support.get("support_caption", ""))
        return data
