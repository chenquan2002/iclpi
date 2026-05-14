"""Support-video context loading for in-context pi0/pi0.5 training.

ICL SUPPORT:
This module keeps support images/captions separate from robot observations.
It is inserted into the OpenPI data pipeline after robot policy-specific input
transforms, before model consumption.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np

import openpi.models.tokenizer as _tokenizer


class SupportManifest:
    """Lookup table for (global_episode_index, support_round_id) -> support context.

    ICL SUPPORT:
    The support manifest is generated offline from support_bank and LeRobot
    meta/episode_origin.jsonl. Training only performs deterministic lookup.
    """

    def __init__(self, support_manifest_path: str | Path):
        self.path = Path(support_manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Support manifest not found: {self.path}")

        self._records: dict[tuple[int, int], dict[str, Any]] = {}
        self._episode_lengths: dict[int, int] = {}

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ep = int(record["global_episode_index"])
                rnd = int(record.get("support_round_id", 0))
                key = (ep, rnd)
                if key in self._records:
                    raise ValueError(f"Duplicate support manifest key: {key}")
                self._records[key] = record
                if "episode_length" in record:
                    self._episode_lengths[ep] = int(record["episode_length"])

        if not self._records:
            raise ValueError(f"Support manifest is empty: {self.path}")

    def get(self, global_episode_index: int, support_round_id: int) -> dict[str, Any]:
        key = (int(global_episode_index), int(support_round_id))
        if key not in self._records:
            raise KeyError(
                f"No support record for global_episode_index={global_episode_index}, "
                f"support_round_id={support_round_id}. Manifest={self.path}"
            )
        return self._records[key]

    def episode_length(self, global_episode_index: int) -> int:
        ep = int(global_episode_index)
        if ep not in self._episode_lengths:
            raise KeyError(f"No episode_length for global_episode_index={ep} in {self.path}")
        return self._episode_lengths[ep]


class SupportFrameCache:
    """Small LRU cache for support frames.npy arrays.

    ICL SUPPORT:
    Prevents each batch from repeatedly loading and decoding support frames.
    `frames.npy` is expected to be [K,H,W,3], uint8, RGB.
    """

    def __init__(self, max_items: int = 1024):
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, frames_npy: str | Path) -> np.ndarray:
        path = str(frames_npy)
        if path in self._cache:
            value = self._cache.pop(path)
            self._cache[path] = value
            return value

        arr = np.load(path)
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 support frames, got {arr.dtype}: {path}")
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected support frames shape [K,H,W,3], got {arr.shape}: {path}")

        if self.max_items > 0:
            while len(self._cache) >= self.max_items:
                self._cache.popitem(last=False)
            self._cache[path] = arr
        return arr


class SupportRoundDataset:
    """Dataset wrapper that adds a deterministic support_round_id to each sample.

    ICL SUPPORT:
    The base dataset is not copied. The wrapper logically expands it by
    support_rounds_per_cycle. This lets the same robot trajectory be paired
    with different precomputed support contexts.
    """

    def __init__(self, dataset, support_rounds_per_cycle: int):
        if support_rounds_per_cycle < 1:
            raise ValueError("support_rounds_per_cycle must be >= 1")
        self._dataset = dataset
        self._base_len = len(dataset)
        self._rounds = int(support_rounds_per_cycle)

    def __len__(self) -> int:
        return self._base_len * self._rounds

    def __getitem__(self, index: SupportsIndex):
        idx = int(index.__index__() if hasattr(index, "__index__") else index)
        base_idx = idx % self._base_len
        support_round_id = idx // self._base_len

        item = dict(self._dataset[base_idx])
        item["support_round_id"] = np.asarray(support_round_id, dtype=np.int64)
        return item


class AddSupportContext:
    """Transform that appends support images, caption tokens, and progress.

    ICL SUPPORT:
    Expected input keys after policy/repack transforms:
      - episode_index
      - frame_index
      - support_round_id

    Added output keys:
      - support_images
      - support_image_mask
      - support_frame_progress
      - chunk_progress
      - support_caption_tokens
      - support_caption_mask
    """

    def __init__(
        self,
        support_manifest_path: str | Path,
        *,
        num_support_frames: int = 8,
        support_cache_size: int = 1024,
        support_caption_max_len: int = 128,
    ):
        self.manifest = SupportManifest(support_manifest_path)
        self.cache = SupportFrameCache(max_items=support_cache_size)
        self.num_support_frames = int(num_support_frames)
        self.support_caption_max_len = int(support_caption_max_len)
        self._tokenizer: _tokenizer.PaligemmaTokenizer | None = None

    def _caption_tokenizer(self) -> _tokenizer.PaligemmaTokenizer:
        if self._tokenizer is None:
            self._tokenizer = _tokenizer.PaligemmaTokenizer(self.support_caption_max_len)
        return self._tokenizer

    @staticmethod
    def _as_int(x: Any) -> int:
        arr = np.asarray(x)
        return int(arr.reshape(-1)[0])

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "episode_index" not in data or "frame_index" not in data:
            raise KeyError(
                "AddSupportContext requires episode_index and frame_index. "
                "Check RepackTransform and policy input transforms preserve metadata."
            )

        if "support_round_id" not in data:
            # Fixed-support fallback, but SupportRoundDataset is preferred.
            data["support_round_id"] = np.asarray(0, dtype=np.int64)

        global_ep = self._as_int(data["episode_index"])
        frame_idx = self._as_int(data["frame_index"])
        support_round_id = self._as_int(data["support_round_id"])

        support = self.manifest.get(global_ep, support_round_id)
        episode_length = int(support.get("episode_length", self.manifest.episode_length(global_ep)))

        frames = self.cache.get(support["support_frames_npy"])
        if frames.shape[0] != self.num_support_frames:
            raise ValueError(
                f"Expected {self.num_support_frames} support frames, got {frames.shape[0]} "
                f"from {support['support_frames_npy']}"
            )

        progress = np.asarray(support["support_frame_progress"], dtype=np.float32)
        if progress.shape != (self.num_support_frames,):
            raise ValueError(
                f"support_frame_progress should have shape ({self.num_support_frames},), got {progress.shape}"
            )

        chunk_progress = frame_idx / max(episode_length - 1, 1)

        # ICL SUPPORT: support images are independent from robot observation images.
        data["support_images"] = frames
        data["support_image_mask"] = np.ones((self.num_support_frames,), dtype=bool)
        data["support_frame_progress"] = progress
        data["chunk_progress"] = np.asarray([chunk_progress], dtype=np.float32)

        caption = str(support.get("support_caption", ""))
        tokens, mask = self._caption_tokenizer().tokenize(caption, state=None)
        data["support_caption_tokens"] = tokens.astype(np.int32)
        data["support_caption_mask"] = mask.astype(bool)
        return data
