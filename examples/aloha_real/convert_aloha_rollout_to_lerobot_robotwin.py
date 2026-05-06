"""
Convert processed rollout episodes into a LeRobotDataset for pi05 training.

Compared with the previous rollout converter, this version adds a clean hook
for a standalone value function module:

    from scripts.value_function import compute_value_labels

The converter itself remains responsible only for:
- reading processed episode data
- reading rollout_info.json
- calling compute_value_labels(...)
- writing all fields into the LeRobotDataset
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Standalone value function hook.
# Put scripts/value_function.py in your project and keep only this import stable.
# from value_function import compute_value_labels
try:
    from .value_function import compute_value_labels
except ImportError:
    from value_function import compute_value_labels

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]

MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

EXTRA_SCALAR_FEATURES: dict[str, dict[str, Any]] = {
    "source_episode_id": {"dtype": "int64", "shape": (1,)},
    "source_step_id": {"dtype": "int64", "shape": (1,)},
    "round_id": {"dtype": "int64", "shape": (1,)},
    "checkpoint_id": {"dtype": "int64", "shape": (1,)},
    "episode_success": {"dtype": "int64", "shape": (1,)},
    "termination": {"dtype": "int64", "shape": (1,)},
    "truncation": {"dtype": "int64", "shape": (1,)},
    "done": {"dtype": "int64", "shape": (1,)},
    "policy_timeout": {"dtype": "int64", "shape": (1,)},
    "step_limit_reached": {"dtype": "int64", "shape": (1,)},
    "step_reward": {"dtype": "float32", "shape": (1,)},
    "step_reward_diff": {"dtype": "float32", "shape": (1,)},
    "acp_value": {"dtype": "float32", "shape": (1,)},
    "acp_advantage": {"dtype": "float32", "shape": (1,)},
    "acp_indicator": {"dtype": "int64", "shape": (1,)},
}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # Keep task only; current JAX pi05 stack can derive prompt from task.
    # features["task"] = {"dtype": "string", "shape": (1,)}

    for key, spec in EXTRA_SCALAR_FEATURES.items():
        features[key] = spec

    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep



def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep



def load_raw_images_per_camera(ep: h5py.File) -> dict[str, np.ndarray]:
    imgs_per_cam: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam



def load_processed_episode(ep_path: Path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(ep)

    return imgs_per_cam, state, action, velocity, effort


_EPISODE_DIR_RE = re.compile(r"episode_(\d+)$")
_EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.hdf5$")



def _extract_episode_id_from_dir(ep_dir: Path) -> int:
    m = _EPISODE_DIR_RE.search(ep_dir.name)
    if not m:
        raise ValueError(f"Invalid processed episode directory name: {ep_dir}")
    return int(m.group(1))



def _find_episode_hdf5_files(processed_root: Path) -> dict[int, Path]:
    ep_map: dict[int, Path] = {}
    for ep_dir in processed_root.glob("episode_*"):
        if not ep_dir.is_dir():
            continue
        episode_id = _extract_episode_id_from_dir(ep_dir)
        candidates = list(ep_dir.glob("episode_*.hdf5"))
        if not candidates:
            continue
        if len(candidates) > 1:
            matched = []
            for c in candidates:
                m = _EPISODE_FILE_RE.search(c.name)
                if m and int(m.group(1)) == episode_id:
                    matched.append(c)
            if len(matched) == 1:
                ep_map[episode_id] = matched[0]
            elif matched:
                ep_map[episode_id] = sorted(matched, key=lambda p: p.name)[0]
            else:
                ep_map[episode_id] = sorted(candidates, key=lambda p: p.name)[0]
        else:
            ep_map[episode_id] = candidates[0]
    if not ep_map:
        raise FileNotFoundError(f"No processed episode hdf5 files found under: {processed_root}")
    return ep_map



def _load_instruction(ep_dir: Path) -> str:
    instructions_path = ep_dir / "instructions.json"
    if not instructions_path.exists():
        raise FileNotFoundError(f"Missing instructions.json: {instructions_path}")

    with open(instructions_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    instructions = db.get("instructions", [])
    if not instructions:
        raise ValueError(f"No instructions found in: {instructions_path}")

    return instructions[0]



def _load_rollout_info(ep_dir: Path) -> dict[str, Any]:
    rollout_info_path = ep_dir / "rollout_info.json"
    if not rollout_info_path.exists():
        raise FileNotFoundError(f"Missing rollout_info.json: {rollout_info_path}")
    with open(rollout_info_path, "r", encoding="utf-8") as f:
        return json.load(f)



def _to_int_scalar(value: Any) -> np.ndarray:
    v = -1 if value is None else int(value)
    return np.asarray([v], dtype=np.int64)



def _to_bool_scalar(value: Any) -> np.ndarray:
    return np.asarray([1 if bool(value) else 0], dtype=np.int64)



def _to_float_scalar(value: Any) -> np.ndarray:
    v = 0.0 if value is None else float(value)
    return np.asarray([v], dtype=np.float32)



def populate_dataset(
    dataset: LeRobotDataset,
    episode_map: dict[int, Path],
    *,
    episodes: list[int] | None = None,
    value_config: dict[str, Any] | None = None,
) -> LeRobotDataset:
    available_episode_ids = sorted(episode_map.keys())
    selected_episode_ids = available_episode_ids if episodes is None else episodes

    for episode_id in tqdm.tqdm(selected_episode_ids):
        if episode_id not in episode_map:
            raise ValueError(
                f"Requested episode_{episode_id} not found under processed root. "
                f"Available ids: {available_episode_ids}"
            )

        ep_path = episode_map[episode_id]
        ep_dir = ep_path.parent

        imgs_per_cam, state, action, velocity, effort = load_processed_episode(ep_path)
        instruction = _load_instruction(ep_dir)
        rollout_info = _load_rollout_info(ep_dir)

        num_frames = int(state.shape[0])
        steps = rollout_info.get("steps", [])
        meta = rollout_info.get("meta", {})

        if len(steps) != num_frames:
            raise ValueError(
                f"Step/frame length mismatch for {ep_path}: "
                f"len(steps)={len(steps)} vs num_frames={num_frames}"
            )

        # ===== The only new part you need to swap later =====
        labels = compute_value_labels(
            rollout_info=rollout_info,
            num_frames=num_frames,
            config=value_config,
        )
        if len(labels) != num_frames:
            raise ValueError(
                f"Value label length mismatch for {ep_path}: "
                f"len(labels)={len(labels)} vs num_frames={num_frames}"
            )
        # ================================================

        for i in range(num_frames):
            step = steps[i]
            label = labels[i]
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
                "source_episode_id": _to_int_scalar(episode_id),
                "source_step_id": _to_int_scalar(i),
                "round_id": _to_int_scalar(meta.get("round_id")),
                "checkpoint_id": _to_int_scalar(meta.get("checkpoint_id")),
                "episode_success": _to_bool_scalar(meta.get("episode_success", False)),
                "termination": _to_bool_scalar(step.get("termination", False)),
                "truncation": _to_bool_scalar(step.get("truncation", False)),
                "done": _to_bool_scalar(step.get("done", False)),
                "policy_timeout": _to_bool_scalar(meta.get("policy_timeout", False)),
                "step_limit_reached": _to_bool_scalar(step.get("step_limit_reached", False)),
                # reward fields now come from the value function hook first
                "step_reward": _to_float_scalar(label.get("step_reward", step.get("reward", 0.0))),
                "step_reward_diff": _to_float_scalar(label.get("step_reward_diff", step.get("reward_diff", 0.0))),
                "acp_value": _to_float_scalar(label.get("acp_value", 0.0)),
                "acp_advantage": _to_float_scalar(label.get("acp_advantage", 0.0)),
                "acp_indicator": _to_int_scalar(label.get("acp_indicator", 0)),
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset



def port_rollout_aloha(
    processed_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    value_config: dict[str, Any] | None = None,
):
    episode_map = _find_episode_hdf5_files(processed_dir)
    sample_hdf5_files = [episode_map[min(episode_map.keys())]]

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(sample_hdf5_files),
        has_velocity=has_velocity(sample_hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(dataset, episode_map, episodes=episodes, value_config=value_config)

    if push_to_hub:
        dataset.push_to_hub()



def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]



def parse_value_config_json(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    return json.loads(value)



def parse_args():
    parser = argparse.ArgumentParser(description="Convert processed rollout episodes to LeRobotDataset")
    parser.add_argument("--processed-dir", type=Path, required=True, help="Path to round_k/processed")
    parser.add_argument("--repo-id", type=str, required=True, help="Target LeRobot repo_id")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--is-mobile", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated real episode ids to convert, e.g. '0,3,8'. Default: all episodes.",
    )
    parser.add_argument(
        "--value-config-json",
        type=str,
        default=None,
        help='JSON string passed to compute_value_labels, e.g. {"gamma":1.0,"indicator_threshold":0.3}',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    port_rollout_aloha(
        processed_dir=args.processed_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        is_mobile=args.is_mobile,
        mode=args.mode,
        episodes=parse_episode_list(args.episodes),
        value_config=parse_value_config_json(args.value_config_json),
    )
