#!/bin/bash
#SBATCH --job-name=fineart_vla_serving_coffee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=worker-0
#SBATCH --gpus=3
#SBATCH --cpus-per-task=45
#SBATCH --mem-per-cpu=8192
#SBATCH --time=8-00:00:00
#SBATCH --output=logs/fineart_vla_serving_coffee_%j.out
#SBATCH --error=logs/fineart_vla_serving_coffee_%j.err
#
# Fine-tune FineART-VLA on the Astribot serving-brewed-coffee dataset with LeRobot.
# Everything that defines the run lives in custom_astribot/fineart_vla_serving_coffee.json
# (written to the checkpoint as train_config.json); this script only sets the cluster env,
# the dataset root and the launcher.
#
#   worker-0, 3x H100, 45 CPUs x 8 GB. FSDP2 over all GPUs (parallelism.dp_shard=-1), fp32 master
#   weights + bf16 autocast, batch_size=8 per GPU -> global batch 24, 500k steps, eval loss on 19
#   held-out episodes every 5k.
#
# Every file the job writes (checkpoints, logs, HF/uv/torch caches, temp files) lands under
# $DATA_HOME=/mnt/data/dungnt232_1, never in $HOME.
#
# Usage, from the lerobot checkout on the cluster (/mnt/data/dungnt232_1/repos/lerobot):
#   # once: build the FineART copy next to the OpenTau dataset (the source is not modified)
#   uv run python custom_astribot/make_fineart_dataset.py \
#       --src /mnt/data/dungnt232_1/data/serving_brewed_coffee_lrb_opentau \
#       --dst /mnt/data/dungnt232_1/data/serving_brewed_coffee_lrb_opentau_fineart
#   mkdir -p logs                                       # Slurm opens --output before this runs
#   SMOKE=1 sbatch custom_astribot/sbatch_train.sh      # 20 steps into ..._smoke, then exits
#   sbatch custom_astribot/sbatch_train.sh              # the real run
#   RESUME=1 sbatch custom_astribot/sbatch_train.sh     # continue from checkpoints/last
#
# Env knobs:
#   DATASET_ROOT  derived dataset    (default: $DATA_HOME/data/serving_brewed_coffee_lrb_opentau_fineart)
#   OUTPUT_DIR    run directory      (default: $DATA_HOME/outputs/lerobot/fineart_vla_astribot_serving_coffee[_smoke])
#   CONFIG        training JSON      (default: custom_astribot/fineart_vla_serving_coffee.json)
# Extra args pass through to lerobot-train, e.g. --steps=100000 --num_workers=8

set -euo pipefail

DATA_HOME=/mnt/data/dungnt232_1

# Slurm runs a spool copy of this script, so locate the checkout via the submit dir.
cd "${SLURM_SUBMIT_DIR:-$DATA_HOME/repos/lerobot}"

CONFIG="${CONFIG:-custom_astribot/fineart_vla_serving_coffee.json}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_HOME/data/serving_brewed_coffee_lrb_opentau_fineart}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
if [[ "$SMOKE" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$DATA_HOME/outputs/lerobot/fineart_vla_astribot_serving_coffee_smoke}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$DATA_HOME/outputs/lerobot/fineart_vla_astribot_serving_coffee}"
fi

# Training needs the derived copy (subtask language columns, flat joint names, relative-action
# stats), not the OpenTau source: on the source, the recipe finds no subtasks and the grippers
# would silently be trained as deltas.
if [[ ! -r "$DATASET_ROOT/meta/fineart_conversion.json" ]]; then
  echo "[train] ERROR: '$DATASET_ROOT' is not a FineART copy (no meta/fineart_conversion.json)." >&2
  echo "[train]        Build it with custom_astribot/make_fineart_dataset.py (see the header)." >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT/videos/" ]]; then
  echo "[train] ERROR: '$DATASET_ROOT/videos' is missing or a dangling symlink." >&2
  exit 1
fi

# --- keep every generated file under $DATA_HOME ---------------------------------------------
export PATH="$HOME/.local/bin:$PATH"                     # uv (batch jobs get no login profile)
export UV_CACHE_DIR="$DATA_HOME/temp/uv_cache"
export UV_PYTHON_INSTALL_DIR="$DATA_HOME/tools/uv_python"
export HF_HOME="$DATA_HOME/hf_cache"                     # weights, tokenizers, token, datasets cache
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export XDG_CACHE_HOME="$DATA_HOME/cache"                 # catch-all for libraries that use ~/.cache
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TRITON_HOME="$XDG_CACHE_HOME/triton_home"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv_compute"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
FAST_TOKENIZER_CACHE="$XDG_CACHE_HOME/lerobot/fast_tokenizers"
export TMPDIR="$DATA_HOME/temp/lerobot_job_${SLURM_JOB_ID:-local}"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$FAST_TOKENIZER_CACHE" "$(dirname "$OUTPUT_DIR")"
trap 'rm -rf "$TMPDIR"' EXIT

# The PaliGemma tokenizer comes from the GATED repo google/paligemma-3b-pt-224; the token lives in
# $HF_HOME/token. Fall back to ~/.cache's copy only to read it.
if [[ -z "${HF_TOKEN:-}" && ! -r "$HF_HOME/token" && -r "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi

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
  ARGS+=(--config_path="$CONFIG" --dataset.root="$DATASET_ROOT" --output_dir="$OUTPUT_DIR"
         --policy.fast_tokenizer_cache_dir="$FAST_TOKENIZER_CACHE")
  if [[ "$SMOKE" == "1" ]]; then
    ARGS+=(--steps=20 --log_freq=1 --save_freq=20 --eval_steps=10 --max_eval_samples=96
           --job_name=fineart_vla_astribot_serving_coffee_smoke)
  fi
fi
ARGS+=(--num_workers="$NUM_WORKERS")

echo "[train] $(hostname) $(date -Is) cwd=$PWD"
echo "[train] config=$CONFIG dataset=$DATASET_ROOT output=$OUTPUT_DIR smoke=$SMOKE resume=$RESUME"
echo "[train] gpus=$NGPU (global batch = 8 x $NGPU) workers/rank=$NUM_WORKERS port=$PORT"
echo "[train] HF_HOME=$HF_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME TMPDIR=$TMPDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# torchrun, not `accelerate launch --mixed_precision ...`: lerobot-train refuses accelerate's env
# vars so train_config.json stays the single description of the run (mixed precision, FSDP and
# wrap units are in the JSON). --no-sync: the env is synced once on the login node, and a job
# must not rewrite the venv under a running sibling job.
uv run --no-sync -- torchrun \
  --nnodes=1 \
  --nproc-per-node="$NGPU" \
  --master-port="$PORT" \
  -m lerobot.scripts.lerobot_train \
  "${ARGS[@]}" \
  "$@"
