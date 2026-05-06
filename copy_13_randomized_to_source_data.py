from pathlib import Path
import shutil
import re
import os
import sys

SRC_BASE = Path("/data/RoboTwin/policy/pi05/processed_data")
DST_BASE = Path("/data/RoboTwin/data_NAS/training_data/source_data")

TASKS = [
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

TASK_CONFIG = "demo_randomized"
EXPERT_NUM = 500
OFFSET = 50

episode_pat = re.compile(r"^episode_(\d+)$")


def copy_dir_no_overwrite(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {dst}")

    dst.mkdir(parents=True, exist_ok=False)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            copy_dir_no_overwrite(item, target)
        elif item.is_file():
            if target.exists():
                raise FileExistsError(f"目标文件已存在，拒绝覆盖: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
        elif item.is_symlink():
            link_target = os.readlink(item)
            os.symlink(link_target, target)
        else:
            print(f"[WARN] 跳过未知类型: {item}")


# 先收集所有需要复制的 episode，并做全局冲突检查
all_jobs = []

print("=" * 80)
print("开始预检查：只要发现目标 episode 已存在，立即停止，不复制任何数据")
print("=" * 80)

for task in TASKS:
    src_task_dir = SRC_BASE / f"{task}-{TASK_CONFIG}-{EXPERT_NUM}"
    dst_task_dir = DST_BASE / task

    print(f"检查任务: {task}")

    if not src_task_dir.exists():
        print(f"[ERROR] 源目录不存在: {src_task_dir}")
        sys.exit(1)

    dst_task_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for p in src_task_dir.iterdir():
        if not p.is_dir():
            continue
        m = episode_pat.match(p.name)
        if m:
            old_idx = int(m.group(1))
            episodes.append((old_idx, p))

    episodes.sort(key=lambda x: x[0])

    if len(episodes) != EXPERT_NUM:
        print(f"[ERROR] {task}: 期望 {EXPERT_NUM} 个 episode，但实际检测到 {len(episodes)} 个")
        sys.exit(1)

    for old_idx, src_ep in episodes:
        new_idx = old_idx + OFFSET
        dst_ep = dst_task_dir / f"episode_{new_idx}"

        if dst_ep.exists():
            print("=" * 80)
            print("[ERROR] 发现目标 episode 已存在，立即停止：")
            print(f"任务: {task}")
            print(f"源:   {src_ep}")
            print(f"目标: {dst_ep}")
            print("为避免覆盖，脚本未执行复制。")
            print("=" * 80)
            sys.exit(1)

        all_jobs.append((task, old_idx, src_ep, dst_ep))

print("=" * 80)
print(f"预检查通过。准备复制 episode 数量: {len(all_jobs)}")
print("=" * 80)

# 通过预检查后再开始复制
for task, old_idx, src_ep, dst_ep in all_jobs:
    print(f"复制 [{task}]: episode_{old_idx} -> {dst_ep.name}")
    copy_dir_no_overwrite(src_ep, dst_ep)

print("=" * 80)
print("全部复制完成")
print("源目录未删除，数据仍保留在 processed_data 中")
print("=" * 80)
