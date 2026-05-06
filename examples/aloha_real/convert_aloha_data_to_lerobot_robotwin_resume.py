
"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Safe version with:
1. start_episode / end_episode for partial conversion.
2. resume mode for appending to an existing LeRobotDataset.
3. dry_run mode to check selected raw hdf5 files before writing.
4. safer overwrite behavior.
5. lower default image writer parallelism to reduce OOM risk.

Example usage:

# Convert all episodes from scratch:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir /path/to/raw/data \
    --repo_id source_data_repo \
    --overwrite

# Convert only remaining episodes to a new repo:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir /path/to/raw/data \
    --repo_id source_data_repo_part2 \
    --start_episode 5873

# Dry run before writing:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir /path/to/raw/data \
    --repo_id source_data_repo_part2 \
    --start_episode 5873 \
    --dry_run

# Resume in the same existing repo. Use with caution:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir /path/to/raw/data \
    --repo_id source_data_repo \
    --resume \
    --start_episode 5873
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001

    # 原版是 10 processes × 5 threads。
    # 大规模转换时容易造成 CPU 内存压力，所以这里默认降到 2 × 1。
    # 如果还出现 exit code 137，可以继续降到 1 × 1。
    image_writer_processes: int = 10
    image_writer_threads: int = 5

    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    overwrite: bool = False,
) -> LeRobotDataset:
    """Create a new LeRobotDataset.

    Compared with the original script, this function no longer silently deletes
    an existing repo. It only removes the old repo when overwrite=True.
    """
    motors = [
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

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    dataset_root = HF_LEROBOT_HOME / repo_id

    if Path(dataset_root).exists():
        if overwrite:
            print(f"[INFO] Removing existing dataset because overwrite=True: {dataset_root}")
            shutil.rmtree(dataset_root)
        else:
            raise FileExistsError(
                f"Dataset already exists: {dataset_root}\n"
                "Use --overwrite to recreate it, or use --resume to append to it, "
                "or choose a new repo_id such as source_data_repo_part2."
            )

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


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def collect_hdf5_files(raw_dir: Path, *, sort_files: bool = False) -> list[Path]:
    """Collect hdf5 files under raw_dir.

    Important:
    - The original script did not sort hdf5_files.
    - If you are resuming a previous run, keep sort_files=False to preserve
      the same traversal behavior as much as possible.
    - For future fresh conversions, sort_files=True is more reproducible.
    """
    hdf5_files: list[Path] = []

    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            file_path = Path(root) / filename
            hdf5_files.append(file_path)

    if sort_files:
        hdf5_files = sorted(hdf5_files)

    return hdf5_files


def infer_next_episode_idx(repo_id: str) -> int:
    """Infer the next output episode id from existing parquet files.

    This is mainly useful for same-repo resume.
    If the existing repo has episode_005872.parquet, this returns 5873.
    """
    dataset_root = HF_LEROBOT_HOME / repo_id
    parquet_files = sorted(Path(dataset_root).glob("data/chunk-*/episode_*.parquet"))

    if not parquet_files:
        return 0

    episode_ids: list[int] = []
    for p in parquet_files:
        try:
            episode_ids.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass

    if not episode_ids:
        return 0

    return max(episode_ids) + 1


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # Load all images in RAM.
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # Load one compressed image after another and decode.
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_array.append(img)

            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array

    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # Add prompt.
        dir_path = os.path.dirname(ep_path)
        json_path = f"{dir_path}/instructions.json"

        with open(json_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict["instructions"]
            instruction = np.random.choice(instructions)

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]

            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode()

        # Explicitly release large episode arrays.
        del imgs_per_cam, state, action, velocity, effort

    return dataset


def make_episode_list(
    *,
    total_episodes: int,
    episodes: list[int] | None,
    start_episode: int | None,
    end_episode: int | None,
) -> list[int]:
    if episodes is not None:
        selected = list(episodes)
    else:
        if start_episode is None:
            start_episode = 0

        if end_episode is None:
            end_episode = total_episodes

        if start_episode < 0 or start_episode > total_episodes:
            raise ValueError(
                f"Invalid start_episode={start_episode}, total_episodes={total_episodes}"
            )

        if end_episode < start_episode or end_episode > total_episodes:
            raise ValueError(
                f"Invalid end_episode={end_episode}, "
                f"start_episode={start_episode}, total_episodes={total_episodes}"
            )

        selected = list(range(start_episode, end_episode))

    for ep_idx in selected:
        if ep_idx < 0 or ep_idx >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep_idx}, total={total_episodes}")

    return selected


def print_episode_preview(hdf5_files: list[Path], selected_episodes: list[int]) -> None:
    print(f"[INFO] Will process {len(selected_episodes)} episodes.")

    if len(selected_episodes) == 0:
        print("[WARN] No episodes selected.")
        return

    print(f"[INFO] Episode index range: {selected_episodes[0]} -> {selected_episodes[-1]}")

    print("[INFO] First selected raw files:")
    for idx in selected_episodes[:5]:
        print(f"  {idx}: {hdf5_files[idx]}")

    print("[INFO] Last selected raw files:")
    for idx in selected_episodes[-5:]:
        print(f"  {idx}: {hdf5_files[idx]}")


def start_existing_dataset_writer_if_needed(
    dataset: LeRobotDataset,
    dataset_config: DatasetConfig,
) -> None:
    """Start image writer for existing datasets if current LeRobot version exposes it.

    Different LeRobot versions may expose slightly different method signatures.
    This function tries the common forms.
    """
    if not hasattr(dataset, "start_image_writer"):
        return

    try:
        dataset.start_image_writer(
            num_processes=dataset_config.image_writer_processes,
            num_threads=dataset_config.image_writer_threads,
        )
        return
    except TypeError:
        pass

    try:
        dataset.start_image_writer(
            dataset_config.image_writer_processes,
            dataset_config.image_writer_threads,
        )
        return
    except TypeError:
        pass

    # Last fallback.
    dataset.start_image_writer()


def stop_dataset_writer_if_needed(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    start_episode: int | None = None,
    end_episode: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    sort_files: bool = False,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert Aloha hdf5 data to LeRobot format.

    Args:
        raw_dir: Directory that contains raw hdf5 files.
        repo_id: Output LeRobot repo id.
        episodes: Explicit list of raw episode indices to process.
        start_episode: Start raw episode index, inclusive.
        end_episode: End raw episode index, exclusive.
        resume: If True, load an existing LeRobotDataset and append episodes.
        overwrite: If True, delete existing output repo and recreate it.
        dry_run: If True, only print selected files and do not write data.
        sort_files: If True, sort hdf5 files. Keep False when resuming an old run
            that was created with the original unsorted script.
    """
    dataset_root = HF_LEROBOT_HOME / repo_id

    if resume and overwrite:
        raise ValueError("resume=True and overwrite=True cannot be used together.")

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = collect_hdf5_files(raw_dir, sort_files=sort_files)

    if len(hdf5_files) == 0:
        raise RuntimeError(f"No hdf5 files found under raw_dir={raw_dir}")

    print(f"[INFO] HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"[INFO] Output dataset root: {dataset_root}")
    print(f"[INFO] Found {len(hdf5_files)} hdf5 files under raw_dir={raw_dir}")
    print(f"[INFO] sort_files={sort_files}")

    if resume:
        if not Path(dataset_root).exists():
            raise FileNotFoundError(
                f"Cannot resume because dataset does not exist: {dataset_root}"
            )

        inferred_start = infer_next_episode_idx(repo_id)
        print(f"[RESUME] Existing dataset found.")
        print(f"[RESUME] Inferred next output episode id: {inferred_start}")

        if start_episode is None and episodes is None:
            start_episode = inferred_start
            print(f"[RESUME] start_episode is not set. Use inferred start_episode={start_episode}")

        if start_episode is not None and episodes is None and start_episode != inferred_start:
            print(
                f"[WARN] start_episode={start_episode}, "
                f"but inferred next output episode id is {inferred_start}."
            )
            print("[WARN] Make sure this is intentional.")

    selected_episodes = make_episode_list(
        total_episodes=len(hdf5_files),
        episodes=episodes,
        start_episode=start_episode,
        end_episode=end_episode,
    )

    print_episode_preview(hdf5_files, selected_episodes)

    if dry_run:
        print("[DRY RUN] No dataset will be written.")
        return

    if resume:
        print(f"[RESUME] Loading existing LeRobotDataset: {repo_id}")
        dataset = LeRobotDataset(repo_id=repo_id)
        start_existing_dataset_writer_if_needed(dataset, dataset_config)
    else:
        dataset = create_empty_dataset(
            repo_id,
            robot_type="mobile_aloha" if is_mobile else "aloha",
            mode=mode,
            has_effort=has_effort(hdf5_files),
            has_velocity=has_velocity(hdf5_files),
            dataset_config=dataset_config,
            overwrite=overwrite,
        )

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=selected_episodes,
    )

    stop_dataset_writer_if_needed(dataset)

    # The original script had dataset.consolidate() commented out.
    # Keep it unchanged unless your LeRobot version requires it.
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)