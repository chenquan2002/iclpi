train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use

echo "========== GPU DEBUG =========="
echo "PID=$$"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run python - <<'PY'
import os
import jax

print("PID:", os.getpid())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("jax.device_count():", jax.device_count())
print("jax.local_device_count():", jax.local_device_count())
print("jax.devices():")
for d in jax.devices():
    print(" ", d)
PY

echo "================================"

echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite