"""
Script to convert Pre-processed End-Effector HDF5 data to the LeRobot dataset v2.0 format.

Usage: python convert_ee_data_to_lerobot.py --raw-dir processed_data_ee/task-setting-50 --repo-id <org>/<dataset-name>
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
import cv2  # 确保导入cv2


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    # === 关键修改 1: 定义 16维 EE 状态名称 ===
    # 顺序依据 process_data_ee.py: [L_Pose(7), L_Grip(1), R_Pose(7), R_Grip(1)]
    # Pose 内部顺序: x, y, z, qw, qx, qy, qz
    state_names = [
        # Left Arm (7)
        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_quat_w", "left_quat_x", "left_quat_y", "left_quat_z",
        # Left Gripper (1)
        "left_gripper",
        
        # Right Arm (7)
        "right_pos_x", "right_pos_y", "right_pos_z",
        "right_quat_w", "right_quat_x", "right_quat_y", "right_quat_z",
        # Right Gripper (1)
        "right_gripper",
    ]

    # 定义相机名称 (保持与 process_data_ee.py 一致)
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        # 注意：这里我们依然使用 "observation.state" 作为 key，
        # 这样在 Training Config 的 Repack Transform 中可以直接映射。
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names), ),
            "names": [
                state_names,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_names), ),
            "names": [
                state_names,
            ],
        },
    }

    # EE 模式下通常没有 velocity 和 effort，直接移除

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640), # 请确保这里的尺寸与 resize 后的尺寸一致
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50, # 请确认你的数据采集频率
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        # 检查是否压缩，我们的 process_data_ee.py 保存的是 jpeg bytes，所以 dim==1
        # 但有些 HDF5 可能直接存 decode 后的
        if camera not in ep["/observations/images"]:
             print(f"Warning: {camera} not found in HDF5")
             continue

        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
]:
    with h5py.File(ep_path, "r") as ep:
        # === 关键修改 2: 读取 state 而不是 qpos ===
        # process_data_ee.py 生成的 key 是 "observations/state"
        if "/observations/state" in ep:
            state = torch.from_numpy(ep["/observations/state"][:])
        elif "/observations/qpos" in ep:
             # 兼容旧代码，防止 key 没改过来
             state = torch.from_numpy(ep["/observations/qpos"][:])
        else:
            raise KeyError(f"No state or qpos found in {ep_path}")

        action = torch.from_numpy(ep["/action"][:])

        # EE 模式下没有 velocity / effort
        
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        # 读取数据 (不再返回 velocity 和 effort)
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        
        num_frames = state.shape[0]
        
        # Load Instructions
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        instruction = "Do the task." # Default fallback
        if os.path.exists(json_Path):
            with open(json_Path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict.get('instructions', ["Do the task."])
                # 随机选择一条指令用于 meta data
                instruction = np.random.choice(instructions)

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # 不再添加 velocity 和 effort

            dataset.add_frame(frame)
        
        dataset.save_episode()

    return dataset


def port_ee(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "EE_TASK",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # 如果已存在，先清理
    if (HF_LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset at {HF_LEROBOT_HOME / repo_id}")
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise ValueError(f"raw_dir does not exist: {raw_dir}")

    # 递归查找所有 hdf5 文件
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))
            
    # 按照 episode 数字排序，保证顺序一致
    # 假设文件名格式是 episode_X.hdf5
    try:
        hdf5_files.sort(key=lambda p: int(p.stem.split('_')[-1]))
    except:
        print("Warning: Could not sort files by episode number, using default sort.")
        hdf5_files.sort()

    print(f"Found {len(hdf5_files)} HDF5 files.")

    dataset = create_empty_dataset(
        repo_id,
        # Robot Type 只是元数据，设为 custom_ee 或 aloha 均可
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        dataset_config=dataset_config,
    )
    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    
    # LeRobot v2.0 可能需要 consolidate，视版本而定，通常 save_episode 后会自动处理
    # dataset.consolidate() 

    if push_to_hub:
        dataset.push_to_hub()
    
    print(f"Conversion complete! Dataset saved to {HF_LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    tyro.cli(port_ee)