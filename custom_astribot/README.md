# FineART-VLA on Astribot serving-brewed-coffee (`custom_astribot/`)

Fine-tunes LeRobot's FineART-VLA (π0.5 + trained language head + FAST action tokens) on the Astribot
serving-brewed-coffee dataset, with the per-frame Flora subtask captions as FineART's subtask timeline.
The model learns to predict the current subtask from the task, and to act on that subtask.

| Property            | Value                                                                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Robot               | `astribot_s1`, bimanual, 30 fps                                                                                                                                                                  |
| Source dataset      | cluster `/mnt/data/dungnt232_1/data/serving_brewed_coffee_lrb_opentau`, local `~/Documents/Work/data/serving_brew_opt`. LeRobot v3.0, 374 episodes, 429,388 frames, task `serving brewed coffee` |
| Training dataset    | derived copy `<source>_fineart` (see below). The source is not modified, so OpenTau keeps working                                                                                                |
| State / action      | 16-dim `[arm_left_j0..6, arm_right_j0..6, gripper_left, gripper_right]`                                                                                                                          |
| Action space        | delta joints (dims 0–13) w.r.t. the current state, **absolute** grippers (dims 14–15), as in the OpenTau config                                                                                  |
| Cameras             | `cam_head`, `cam_left_wrist`, `cam_right_wrist` (360×640, resized/padded to 224×224)                                                                                                             |
| Subtasks            | 3,526 segments, 418 distinct captions (Flora-generated, video-repaired, **not human-verified**)                                                                                                  |
| Recipe              | the base's `subtask.yaml`: 30% `task → current subtask` text (CE), 70% `subtask + state → actions`                                                                                               |
| Init weights        | `lerobot/FineART-VLA-subtask-midtrained` @ `b6cf4faf` (what `lerobot/fineart_vla_base` resolves to)                                                                                              |
| Losses              | flow (×10) + FAST action tokens (×1) on low-level samples, text CE (×1) on high-level samples, knowledge insulation on                                                                           |
| Stop at subtask end | `hold_after_subtask_end=true`: chunk steps past the active subtask's end hold its last action (label unchanged)                                                                                  |
| Compute             | 3× H100, FSDP2 across all 3 GPUs, fp32 master weights + bf16 autocast                                                                                                                            |
| Schedule            | batch 8/GPU = 24 global, 500k steps (~29 epochs of the 409k training frames), AdamW 2.5e-5 → 5e-6 cosine, 2k warmup                                                                              |
| Eval                | the last 19 episodes (5%) held out, eval loss every 5k steps on 4,800 samples                                                                                                                    |

## Files

- `make_fineart_dataset.py`: builds the derived training copy. Run once per machine; it takes about 20 s on CPU.
- `fineart_vla_serving_coffee.json`: the full `lerobot-train` config. The policy block is the base checkpoint's
  own `config.json` with overrides, so the FAST token layout (`fast_skip_tokens=128`, `max_action_tokens=384`),
  `tokenizer_max_length=128` etc. match the weights.
- `sbatch_train.sh`: the Slurm launcher (`SMOKE=1`, `RESUME=1`, `DATASET_ROOT=...`).

## What the derived copy changes

`<dst>/meta` and `<dst>/data` are copies (~50 MB) and `<dst>/videos` is a symlink to the source's videos.

1. **Subtask language columns.** `language_persistent` holds one `style=subtask` row per `response` caption
   segment, timestamped at the segment's first frame and repeated on every frame of the episode.
   `language_events` is empty. This is the layout `lerobot-annotate` writes. Captions are lightly normalized
   (whitespace, capitalized first letter, one trailing period), which takes 503 raw phrasings to 418;
   adjacent segments that become identical are merged. The `response` column itself is kept.
2. **Flat joint names.** The source stores the names as a nested list, which LeRobot's relative-action mask
   reads as a single name. With `relative_exclude_joints=["gripper"]`, the grippers would then silently be
   trained as deltas. The script asserts the resulting mask is 14 relative + 2 absolute.
3. **Relative-space action stats.** Only `stats.json`'s `action` entry is replaced: delta stats over all
   valid 50-step chunks, grippers absolute. The state stats are kept, after the script checks their
   q01/q99 against the data. Don't use `lerobot-edit-dataset recompute_stats` for this, because its
   per-episode quantile aggregation puts state q01/q99 near the min/max.

## Stopping when the commanded subtask is done

Without intervention the policy does **not** stop after a `/subtask`. A sample is labelled with the subtask
active at its first frame, but its 50-step (1.7 s) action chunk runs into the next subtask in 35% of
training samples. The demonstrator also moves fastest at boundaries (0.52 rad/s vs a 0.26 rad/s median),
and only 9 of 3,179 boundaries are followed by 1 s of stillness. So the model learns that the end of X
flows into the next motion.

`hold_after_subtask_end=true` (a FineART processor step, training only) fixes the targets. For a sample
labelled X, chunk steps inside X stay the real demonstration, and steps after X's end become X's last
commanded action, repeated. The label stays X, so "X" means "do X, then stand still". Samples labelled
with the next subtask are untouched, so no trajectory data is lost; that motion is still learned under
its own label. On this dataset 33% of samples get a modified tail. Episode ends were already held: the
dataset pads past the last frame by repeating its action, and those steps are in the loss.

Trade-off: under `/autosteer`, moving through a boundary now waits for the planner to switch the
subtask, so the robot can pause until the next planner query. Use a short `--autosteer_interval_s`
(about 1 s instead of the default 10).

## Verified locally (CPU, real frames)

- Config parses and validates against the current dataclasses.
- Train/eval split: 409,408 / 19,980 frames. Video decodes through the symlink.
- Rendered samples:
  - high-level: `User: serving brewed coffee\nAssistant: <subtask>.<eos>`, supervising only the subtask + EOS, no action loss
  - low-level: `User: <subtask>, State: <16 bins>;`, with flow + FAST loss
- Worst-case prompt is 106 of 128 tokens, so no subtask gets truncated.
- FAST tokenizer refit on this data: reconstruction RMSE 0.009 (limit 0.1), ≤79 tokens per chunk (cap 384).
- Preprocessor → postprocessor recovers absolute actions to about 1e-7.
- FSDP wrap units (`GemmaMLP`, `SiglipEncoderLayer`): checked on meta tensors only. **Run `SMOKE=1` first.**
  Why these units: FineART's joint attention calls each decoder layer's projections, norms and `mlp`
  directly, never the layer's `forward()`, so whole decoder layers can't be FSDP2 units. `compile_model`
  is off for the same reason.

## Run on the cluster

```bash
cd /mnt/data/dungnt232_1/repos && git clone <your lerobot fork> lerobot && cd lerobot
uv sync --locked --extra fineart_vla --extra training

uv run python custom_astribot/make_fineart_dataset.py \
    --src /mnt/data/dungnt232_1/data/serving_brewed_coffee_lrb_opentau \
    --dst /mnt/data/dungnt232_1/data/serving_brewed_coffee_lrb_opentau_fineart

# optional, if compute nodes lack internet: pre-warm the HF cache (gated PaliGemma needs a token)
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download lerobot/FineART-VLA-subtask-midtrained --revision b6cf4faf2630f971dd390bc32365e9d9c330da8e
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download google/paligemma-3b-pt-224 --include "tokenizer*" "special_tokens_map.json" "added_tokens.json"
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download lerobot/fast-action-tokenizer --revision 3d4d8dd847c5333f1cb0f54d00074ba22b14a5b7

mkdir -p logs
SMOKE=1 sbatch custom_astribot/sbatch_train.sh   # 20 steps + an eval pass; check memory and that it saves
sbatch custom_astribot/sbatch_train.sh           # 500k steps
RESUME=1 sbatch custom_astribot/sbatch_train.sh  # requeue after the 8-day limit / a crash
```

What to watch: `flow_loss`, `fast_action_loss` and `text_loss` separately, and the eval loss. At ~29 epochs
the eval loss will likely bottom out well before 500k steps, so pick checkpoints by eval loss and by
rollouts, not by the last step.

## Disk

`save_freq=25000` gives 20 checkpoints of about 65 GB each: fp32 `model.safetensors`, DCP model shards and
DCP optimizer state, so **~1.3 TB** in total. Prune `training_state/` and `pretrained_model/pytorch_model_fsdp_0/`
from checkpoints you only keep for evaluation.

## Deploying

The processors expect `observation.state` (16), `observation.images.cam_head`, `observation.images.cam_left_wrist`
and `observation.images.cam_right_wrist`. The postprocessor turns predicted deltas back into absolute joint
targets. With `lerobot-rollout --interactive=true --autosteer_interval_s=1`:

- `/subtask <text>` runs one subtask, and the robot should come to rest when it's done. Use one of the
  training captions; `meta/fineart_conversion.json` has the counts and the parquet `language_persistent`
  rows have the exact strings.
- `/autosteer serving brewed coffee` lets the model generate its own subtasks. Keep the interval short so
  it doesn't pause at each boundary.
- A trained hold is not a safety stop: keep a hand on `/stop` for the first rollouts. Also note that with
  `n_action_steps=50` the robot executes up to 1.7 s of a chunk before it re-plans.
