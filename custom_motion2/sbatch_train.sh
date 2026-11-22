#!/bin/bash
#SBATCH --job-name=fineart_vla_fold_clothes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=8-00:00:00
#SBATCH --output=logs/fineart_vla_fold_clothes_%j.out
#SBATCH --error=logs/fineart_vla_fold_clothes_%j.err
#
# Fine-tune FineART-VLA on the Motion2 (VM_M2) fold-clothes dataset with LeRobot.
# Everything that defines the run lives in custom_motion2/fineart_vla_fold_clothes.json
# (written to the checkpoint as train_config.json); this script only sets the cluster env,
# the dataset root and the launcher.
#
#   3x H100, FSDP2 over all GPUs (parallelism.dp_shard=-1), fp32 master weights + bf16 autocast,
#   batch_size=8 per GPU -> global batch 24, 500k steps.
#
# Usage, from the lerobot checkout on the cluster:
#   mkdir -p logs                                      # Slurm opens --output before this runs
#   SMOKE=1 sbatch custom_motion2/sbatch_train.sh      # 20 steps into outputs/..._smoke, then exits
#   sbatch custom_motion2/sbatch_train.sh              # the real run
#   RESUME=1 sbatch custom_motion2/sbatch_train.sh     # continue from checkpoints/last
#
# Env knobs:
#   DATASET_ROOT  dataset directory (default: /mnt/data/dungnt232_1/data/m2v2_fold_clothes_merged_clean_arms16)
#   OUTPUT_DIR    run directory      (default: outputs/fineart_vla_m2v2_fold_clothes[_smoke])
#   CONFIG        training JSON      (default: custom_motion2/fineart_vla_fold_clothes.json)
#   HF_HOME       HF cache           (default: /mnt/data/dungnt232_1/hf_cache)
# Extra args pass through to lerobot-train, e.g. --steps=100000 --trackio.enable=true

set -euo pipefail

# Slurm runs a spool copy of this script, so locate the checkout via the submit dir.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/lerobot}"

CONFIG="${CONFIG:-custom_motion2/fineart_vla_fold_clothes.json}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/dungnt232_1/data/m2v2_fold_clothes_merged_clean_arms16}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
if [[ "$SMOKE" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/fineart_vla_m2v2_fold_clothes_smoke}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/fineart_vla_m2v2_fold_clothes}"
fi

# The dataset must carry the edited metadata: flat joint names with the grippers renamed to
# kLeftGripper/kRightGripper (so relative_exclude_joints=["gripper"] keeps exactly dims 14-15
# absolute) and relative-action stats in meta/stats.json. A copy with the original names
# would silently train the grippers as deltas, so refuse it.
INFO="$DATASET_ROOT/meta/info.json"
if [[ ! -r "$INFO" ]]; then
  echo "[train] ERROR: no dataset at '$DATASET_ROOT' (missing meta/info.json). Set DATASET_ROOT." >&2
  exit 1
fi
if ! grep -q '"kLeftGripper"' "$INFO" || ! grep -q '"kRightGripper"' "$INFO"; then
  echo "[train] ERROR: $INFO still has the original gripper names. Copy the edited meta/ over." >&2
  exit 1
fi

# The PaliGemma tokenizer comes from the GATED repo google/paligemma-3b-pt-224, and the weights
# (~17 GB with the tokenizer) from lerobot/FineART-VLA-subtask-midtrained. Keep the cache on the
# shared volume, and carry the token across since redirecting HF_HOME orphans ~/.cache's one.
export HF_HOME="${HF_HOME:-/mnt/data/dungnt232_1/hf_cache}"
if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || echo "[train] WARNING: no HF_TOKEN; gated google/paligemma-3b-pt-224 will 401 on a cache miss" >&2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NGPU="${SLURM_GPUS_ON_NODE:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
# Dataloader workers are per rank; leave ~2 cores per rank for the trainer itself.
NUM_WORKERS=$(( (CPUS - 2 * NGPU) / NGPU ))
(( NUM_WORKERS < 1 )) && NUM_WORKERS=1
PORT=$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))

ARGS=()
if [[ "$RESUME" == "1" ]]; then
  # A resume reads the run's own saved config, so dataset root and the rest come from there.
  RESUME_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -r "$RESUME_CONFIG" ]]; then
    echo "[train] ERROR: nothing to resume at '$RESUME_CONFIG'." >&2
    exit 1
  fi
  ARGS+=(--config_path="$RESUME_CONFIG" --resume=true)
else
  ARGS+=(--config_path="$CONFIG" --dataset.root="$DATASET_ROOT" --output_dir="$OUTPUT_DIR")
  if [[ "$SMOKE" == "1" ]]; then
    ARGS+=(--steps=20 --log_freq=1 --save_freq=20 --job_name=fineart_vla_m2v2_fold_clothes_smoke)
  fi
fi
ARGS+=(--num_workers="$NUM_WORKERS")

echo "[train] $(hostname) $(date -Is) cwd=$PWD"
echo "[train] config=$CONFIG dataset=$DATASET_ROOT output=$OUTPUT_DIR smoke=$SMOKE resume=$RESUME"
echo "[train] gpus=$NGPU (global batch = 8 x $NGPU) workers/rank=$NUM_WORKERS port=$PORT HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# torchrun, not `accelerate launch --mixed_precision ...`: lerobot-train refuses accelerate's env
# vars so train_config.json stays the single description of the run (mixed precision, FSDP and
# wrap units are in the JSON). `uv run` because a batch job has no activated venv.
exec uv run -- torchrun \
  --nnodes=1 \
  --nproc-per-node="$NGPU" \
  --master-port="$PORT" \
  -m lerobot.scripts.lerobot_train \
  "${ARGS[@]}" \
  "$@"
