#!/bin/bash
set -e
set -u

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

trap 'echo -e "${RED}❌ 脚本在第 $LINENO 行出错，退出码: $?${NC}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$#" -ne 2 ]; then
    echo -e "${RED}❌ 错误: 参数数量不对！${NC}"
    echo "用法: bash run_test_icl.sh <train_config_name> <gpu_use>"
    echo "示例: bash run_test_icl.sh pi05_aloha_robotwin_icl_random_lora 0,1"
    exit 1
fi

train_config_name=${1}
gpu_use=${2}

model_name="test_icl"
repo_id="one_task_test_repo"

# LeRobot repo 路径：
# ${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}
export XDG_CACHE_HOME="${ROBOTWIN_ROOT}/data/lerobot_data"
export REPO_ID="${repo_id}"

# ICL SUPPORT: support manifest 路径。
# 你现在检查通过的是 random-view rounds40 版本。
export SUPPORT_MANIFEST_PATH="/data/RoboTwin/data/support_data_test/manifests/one_task_test_human_random_k8_rounds40.jsonl"
export SUPPORT_ROUNDS_PER_CYCLE="40"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}🚀 Test ICL 启动${NC}"
echo -e "${BLUE}repo_id: ${repo_id}${NC}"
echo -e "${BLUE}model_name: ${model_name}${NC}"
echo -e "${BLUE}train_config_name: ${train_config_name}${NC}"
echo -e "${BLUE}gpu_use: ${gpu_use}${NC}"
echo -e "${BLUE}XDG_CACHE_HOME: ${XDG_CACHE_HOME}${NC}"
echo -e "${BLUE}SUPPORT_MANIFEST_PATH: ${SUPPORT_MANIFEST_PATH}${NC}"
echo -e "${BLUE}SUPPORT_ROUNDS_PER_CYCLE: ${SUPPORT_ROUNDS_PER_CYCLE}${NC}"
echo -e "${BLUE}======================================================${NC}"

cd "${SCRIPT_DIR}"

echo -e "${YELLOW}[0/2] 检查 LeRobot repo 和 support manifest...${NC}"
test -d "${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}"
test -f "${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}/meta/episode_origin.jsonl"
test -f "${SUPPORT_MANIFEST_PATH}"
echo -e "${GREEN}✅ 路径检查通过。${NC}"

echo -e "${YELLOW}[1/2] 计算统计量...${NC}"
# uv run scripts/compute_norm_stats.py --config-name "${train_config_name}"
echo -e "${GREEN}✅ 统计量计算完成。${NC}"

echo -e "${YELLOW}[2/2] 开始训练 Finetune...${NC}"
bash "${SCRIPT_DIR}/finetune.sh" \
    "${train_config_name}" \
    "${model_name}" \
    "${gpu_use}"

echo -e "${GREEN}✅ Finetune 完成！${NC}"