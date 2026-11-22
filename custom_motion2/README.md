# FineART-VLA on Motion2 fold-clothes (`custom_motion2/`)

Fine-tunes LeRobot's FineART-VLA (π0.5 + language head + FAST action tokens) on
`m2v2_fold_clothes_merged_clean_arms16`. The robot conventions come from openpi's `custom_motion2`.

| Property       | Value                                                                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Robot          | `VM_M2` (Motion2), bimanual, 30 fps                                                                                                                                        |
| Dataset        | LeRobot v3.0, 1,657 episodes, 5,195,355 frames (~48 h), one task: `Fold clothes`                                                                                           |
| State / action | 16-dim `[7 left-arm joints, 7 right-arm joints, kLeftGripper, kRightGripper]`, absolute joint targets on disk                                                              |
| Action space   | delta joints (dims 0–13) w.r.t. the current state, **absolute** grippers (dims 14–15)                                                                                      |
| Cameras used   | `cam_head` (base view), `cam_left_wrist`, `cam_right_wrist`. `cam_torso` is in the data but not a policy input (it is still decoded by the loader, a throughput cost only) |
| Init weights   | `lerobot/FineART-VLA-subtask-midtrained` @ `b6cf4faf` (what `lerobot/fineart_vla_base` resolves to)                                                                        |
| Language       | goal-only: every frame is a low-level sample with prompt `Fold clothes, State: <bins>;`, no text loss                                                                      |
| Losses         | flow matching (×10) + FAST action-token CE (×1), knowledge insulation on                                                                                                   |
| Compute        | 3× H100, FSDP2 across all 3 GPUs, fp32 master weights + bf16 autocast                                                                                                      |
| Schedule       | batch 8/GPU = 24 global, 500k steps (~12M samples, ~2.3 epochs), AdamW 2.5e-5 → 5e-6 cosine, 2k warmup                                                                     |

## Files

- `fineart_vla_fold_clothes.json`: the full `lerobot-train` config. The policy block is the base checkpoint's
  own `config.json` with the overrides below, so the FAST token layout (`fast_skip_tokens=128`,
  `max_action_tokens=384`), `tokenizer_max_length=128`, `flow_num_repeats=1` etc. match the weights.
- `sbatch_train.sh`: the Slurm launcher (`SMOKE=1`, `RESUME=1`, `DATASET_ROOT=...`).

## Dataset metadata edits (already applied to the local copy)

Two edits in `meta/`. The originals are kept as `info.json.orig` and `stats.json.orig`:

1. **`info.json`**: the `names` of `observation.state` and `action` were a nested list (`[[...]]`) and
   the grippers were called `kLeft` / `kRight`. They are now a flat list, with the grippers renamed
   `kLeftGripper` / `kRightGripper`. Without this, LeRobot's relative-action mask (substring match on
   `relative_exclude_joints`) silently made _every_ dim relative, grippers included. With
   `["kLeft", "kRight"]` it would also have caught every `kLeft*` / `kRight*` arm joint.
2. **`stats.json`**: only the `action` entry was replaced, with stats in relative space (all 5.1M valid
   50-step chunks, grippers absolute) from `lerobot-edit-dataset --operation.type recompute_stats
--operation.relative_action true`. Every other entry is the original. Training reads normalization
   stats straight from this file, so absolute-action stats would normalize the deltas wrongly. The relative
   q01–q99 were checked against exact percentiles of 200k sampled chunks (max error 0.7% of range).

   Do **not** take that tool's full output: its per-episode quantile aggregation put
   `observation.state` q01/q99 almost at the min/max (up to 1.28 rad off the true percentiles, which the
   original file matches to 1e-3). With quantile normalization, that would have squashed the state input.

**Copy the edited `meta/` to the cluster along with the data.** `sbatch_train.sh` refuses to start on a
copy that still has the old gripper names.

## Why fp32 + FSDP, and why these wrap units

With `dtype=bfloat16`, LeRobot's FineART stores the Gemma backbone and action expert as bf16 parameters
under plain AdamW (no fp32 master copy). At LR 2.5e-5 → 5e-6, most updates are below bf16's resolution and
round away. So the policy is fp32 and `accelerator.mixed_precision=bf16` supplies bf16 compute. Full fp32
params, grads and Adam state (~66 GB) don't fit next to activations on one 80 GB GPU, so FSDP2 shards them
(~22 GB static per GPU over 3 GPUs).

FSDP2 gathers a unit's parameters in that unit's `forward()` hook. FineART's joint prefix/action
attention loop calls each decoder layer's `q/k/v/o_proj`, norms and `mlp` **directly**, never the
layer's own `forward`. So whole decoder layers cannot be units. The units are the modules that _are_
called through `forward`: `GemmaMLP` (18 VLM + 18 action-expert layers) and `SiglipEncoderLayer` (27).
Together they hold 2.45B of the 4.14B parameters. The remainder (embeddings, `lm_head`, attention
projections, norms) is the root unit, gathered for the whole forward pass. `compile_model` is off for this
reason too.

This FSDP layout has been checked on meta tensors only. **Run `SMOKE=1` first** and check that it
reaches step 20, saves a checkpoint, and that per-GPU memory looks sane.

## Run on the cluster

```bash
# once: code + env
cd /mnt/data/dungnt232_1/repos && git clone <your lerobot fork> lerobot && cd lerobot
uv sync --locked --extra fineart_vla --extra training

# once: data (118 GB), including the edited meta/
rsync -a --info=progress2 /home/kronze/Desktop/m2v2_fold_clothes_merged_clean_arms16/ \
    <cluster>:/mnt/data/dungnt232_1/data/m2v2_fold_clothes_merged_clean_arms16/

# optional, if compute nodes lack internet: pre-warm the HF cache (gated PaliGemma needs a token)
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download lerobot/FineART-VLA-subtask-midtrained --revision b6cf4faf2630f971dd390bc32365e9d9c330da8e
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download google/paligemma-3b-pt-224 --include "tokenizer*" "special_tokens_map.json" "added_tokens.json"
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download lerobot/fast-action-tokenizer --revision 3d4d8dd847c5333f1cb0f54d00074ba22b14a5b7

mkdir -p logs
SMOKE=1 sbatch custom_motion2/sbatch_train.sh   # 20 steps, then inspect logs/ and nvidia-smi peaks
sbatch custom_motion2/sbatch_train.sh           # 500k steps
RESUME=1 sbatch custom_motion2/sbatch_train.sh  # requeue after the 8-day limit / a crash
```

What to watch in the log: `flow_loss` and `fast_action_loss` separately (there is no `text_loss`; it is
disabled). The first run also fits a dataset-specific FAST tokenizer (rank 0, a few minutes; the other
ranks wait) and logs its reconstruction RMSE.

## Disk

`save_freq=25000` gives 20 checkpoints. Each holds the gathered fp32 `model.safetensors` (~16.6 GB), the
DCP model shards (~16.6 GB, for fast resume) and the DCP optimizer state (~33 GB), roughly 65 GB each
and **~1.3 TB for the full run**. Prune `training_state/` and `pretrained_model/pytorch_model_fsdp_0/` from
old checkpoints you only keep for evaluation, or set `--checkpoint_format=safetensors` to drop the model shards.

## Deploying

The checkpoint's processors expect the dataset's names: `observation.state` (16), `observation.images.cam_head`,
`observation.images.cam_left_wrist`, `observation.images.cam_right_wrist`, and the task `Fold clothes`
(same string as training). The postprocessor turns the predicted deltas back into absolute joint targets
using the current state. With `lerobot-rollout`, pass `--task="Fold clothes"`, or in `--interactive=true`
mode use `/subtask Fold clothes`. `/autosteer` will not work: this run trains no subtask generation, so
the saved goal-only recipe has no `${subtask}` target and a next-subtask query raises an error.
