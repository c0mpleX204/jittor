# StraightPCF CVM Reproduction Temporary Report

Date: 2026-07-21

This is a temporary runbook for the official-style StraightPCF CVM experiment.
It records the exact cloud paths, temporary YAML files, continuation checkpoints,
prediction commands, and scoring commands used in the current run.

Do not treat the temporary YAML files below as committed config files. Write them
on the cloud with `cat > ...` when reproducing the run.

## Current Goal

Reproduce the full StraightPCF-style CVM stage before training DistanceModule.

Current model target:

```text
StraightPCFCoupledVelocityModule
```

Implementation entry:

```text
src/model/straightpcf_vm_dm.py
```

Model registration:

```text
src/model/parse.py
```

The CVM is initialized from the previous best single Surface-Straight VM:

```text
/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl
```

Previous best single VM local score:

```text
CD 54.0110
P2S 71.0060
final 62.5085
```

## Directory Map

Cloud runtime code directory:

```text
/root/src/starter_code
```

Temporary full git repo directory:

```text
/root/src/jittor_surface_straight_branch
```

Training mesh dataset:

```text
/root/autodl-tmp/dataset_train
```

Local clean test set, used only for offline scoring:

```text
/root/autodl-tmp/true_test
```

Local noisy test set, used as model input:

```text
/root/autodl-tmp/noisy_test
```

Patch cache used by this CVM run:

```text
/root/autodl-tmp/cache_surface_straight_nf12000_seed123
```

Important cache file:

```text
/root/autodl-tmp/cache_surface_straight_nf12000_seed123/train_cache.txt
```

Each cached patch file:

```text
/root/autodl-tmp/cache_surface_straight_nf12000_seed123/patches/00000000/patch.npz
```

Expected fields in `patch.npz`:

```text
pc_noisy
pc_clean
pc_mix
pc_time
```

## Sync Latest Code

Run this when local changes have been pushed.

```bash
TMP=/root/src/jittor_surface_straight_branch

cd "$TMP"
git fetch origin exp/surface-straight-vm
git checkout exp/surface-straight-vm
git reset --hard origin/exp/surface-straight-vm

rsync -av "$TMP/release4/starter_code/" /root/src/starter_code/

cd /root/src/starter_code
python -m py_compile src/model/straightpcf_vm_dm.py src/model/parse.py
grep -n "StraightPCFCoupledVelocityModule" src/model/parse.py
```

If the cloud remote is not named `origin`, check:

```bash
git remote -v
```

and replace `origin` with the actual remote name.

## Build Or Verify 12000 Patch Cache

The current run uses a fixed 12000-item cache. This is not model pretraining;
it only precomputes OBJ loading, surface sampling, noise, normalization, patch
construction, and `pc_time`.

```bash
cd /root/src/starter_code

CACHE=/root/autodl-tmp/cache_surface_straight_nf12000_seed123

if [ ! -f "$CACHE/train_cache.txt" ]; then
  python tools/build_patch_cache.py \
    --dataset-dir /root/autodl-tmp/dataset_train \
    --list datalist/train.txt \
    --transform configs/transform/surface_straight.yaml \
    --out-dir "$CACHE" \
    --num-items 12000 \
    --seed 123 \
    --overwrite
fi

wc -l "$CACHE/train_cache.txt"

python - <<'PY'
import numpy as np
p = "/root/autodl-tmp/cache_surface_straight_nf12000_seed123/patches/00000000/patch.npz"
d = np.load(p)
print(d.files)
print({k: d[k].shape for k in d.files})
PY
```

Expected count:

```text
12000
```

Expected shapes:

```text
pc_noisy: (1, 1000, 3)
pc_clean: (1, 1000, 3)
pc_mix:   (1, 1000, 3)
pc_time:  (1,)
```

## Common Temporary YAML

These files are shared by all CVM stages in this run.

```bash
cd /root/src/starter_code

CACHE=/root/autodl-tmp/cache_surface_straight_nf12000_seed123

cat > configs/transform/cache_passthrough_tmp.yaml <<'YAML'
train_transform: {}
validate_transform: {}
predict_transform: {}
YAML

cat > configs/data/train_cache_surface_12000_tmp.yaml <<YAML
train_dataset:
  shuffle: True
  batch_size: 2
  num_workers: 0
  datapath:
    input_dataset_dir: $CACHE
    use_prob: False
    loader: npz_patch
    data_name: patch.npz
    ignore_check: True
    data_path:
      cache: [
        [$CACHE/train_cache.txt, 1.0],
      ]
YAML

cat > configs/model/straightpcf_cvm_tmp.yaml <<'YAML'
__target__: StraightPCFCoupledVelocityModule
num_modules: 2
tot_its: 4
num_train_points: 128
dsm_sigma: 0.01
consistency_loss_weight: 10.0
init_ckpt: /root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl

velocity_model:
  frame_knn: 16
  num_train_points: 128
  feat_embedding_dim: 256
  decoder_hidden_dim: 64
  dsm_sigma: 0.01
  denoise_steps: 4
YAML
```

Important:

- `num_workers: 0` is intentional. The cache already removes the expensive CPU
  preprocessing.
- `batch_size: 2` gives `6000 batches/epoch` for 12000 cached patches.
- `init_ckpt` is only for the first CVM stage. Continuation stages use task
  `load_ckpt`.
- Keep each `YAML` terminator on its own line. Do not paste the next command on
  the same line as `YAML`.

## Stage 1: Train CVM To Total Epoch 9

This stage starts from the single Surface-Straight VM checkpoint through
`init_ckpt`. Do not add task-level `load_ckpt` here.

Output directory:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep
```

```bash
cd /root/src/starter_code

cat > configs/system/straightpcf_cvm_12000_ep9_tmp.yaml <<'YAML'
__target__: vm
ckpt_save_dir: /root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep
ckpt_save_name: checkpoint
YAML

cat > configs/task/train_straightpcf_cvm_cache12000_ep9_tmp.yaml <<'YAML'
mode: train
debug: False

components:
  data: train_cache_surface_12000_tmp
  transform: cache_passthrough_tmp
  system: straightpcf_cvm_12000_ep9_tmp
  model: straightpcf_cvm_tmp

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.00005

trainer:
  epochs: 10
YAML

python run.py --task configs/task/train_straightpcf_cvm_cache12000_ep9_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_straightpcf_cvm_cache12000_ep9.log
```

Checkpoint meaning:

```text
checkpoint_9.pkl = total epoch 9
```

Observed score:

```text
CD 52.4533
P2S 73.4140
final 62.9337
```

## Stage 2: Continue CVM To Total Epoch 19

Input checkpoint:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep/checkpoint_9.pkl
```

Output directory:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep9_ft10_lr5e5
```

```bash
cd /root/src/starter_code

cat > configs/system/straightpcf_cvm_12000_ep19_tmp.yaml <<'YAML'
__target__: vm
ckpt_save_dir: /root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep9_ft10_lr5e5
ckpt_save_name: checkpoint
YAML

cat > configs/task/train_straightpcf_cvm_cache12000_ep19_tmp.yaml <<'YAML'
mode: train
debug: False
load_ckpt: /root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep/checkpoint_9.pkl

components:
  data: train_cache_surface_12000_tmp
  transform: cache_passthrough_tmp
  system: straightpcf_cvm_12000_ep19_tmp
  model: straightpcf_cvm_tmp

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.00005

trainer:
  epochs: 10
YAML

python run.py --task configs/task/train_straightpcf_cvm_cache12000_ep19_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_straightpcf_cvm_cache12000_ep19.log
```

Checkpoint meaning:

```text
checkpoint_9.pkl = total epoch 19
```

Observed score:

```text
CD 53.7864
P2S 75.5082
final 64.6473
```

## Stage 3: Continue CVM To Total Epoch 29

Input checkpoint:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep9_ft10_lr5e5/checkpoint_9.pkl
```

Output directory:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5
```

```bash
cd /root/src/starter_code

cat > configs/system/straightpcf_cvm_12000_ep29_tmp.yaml <<'YAML'
__target__: vm
ckpt_save_dir: /root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5
ckpt_save_name: checkpoint
YAML

cat > configs/task/train_straightpcf_cvm_cache12000_ep29_tmp.yaml <<'YAML'
mode: train
debug: False
load_ckpt: /root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep9_ft10_lr5e5/checkpoint_9.pkl

components:
  data: train_cache_surface_12000_tmp
  transform: cache_passthrough_tmp
  system: straightpcf_cvm_12000_ep29_tmp
  model: straightpcf_cvm_tmp

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.00005

trainer:
  epochs: 10
YAML

python run.py --task configs/task/train_straightpcf_cvm_cache12000_ep29_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_straightpcf_cvm_cache12000_ep29.log
```

Checkpoint meaning:

```text
checkpoint_9.pkl = total epoch 29
```

Observed score:

```text
CD 55.7714
P2S 76.3234
final 66.0474
```

## Stage 4: Continue CVM From Total Epoch 29 For 30 More Epochs

Input checkpoint:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5/checkpoint_9.pkl
```

Output directory:

```text
/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep29_ft30_lr5e5
```

```bash
cd /root/src/starter_code

cat > configs/system/straightpcf_cvm_12000_ep59_tmp.yaml <<'YAML'
__target__: vm
ckpt_save_dir: /root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep29_ft30_lr5e5
ckpt_save_name: checkpoint
YAML

cat > configs/task/train_straightpcf_cvm_cache12000_ep59_tmp.yaml <<'YAML'
mode: train
debug: False
load_ckpt: /root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5/checkpoint_9.pkl

components:
  data: train_cache_surface_12000_tmp
  transform: cache_passthrough_tmp
  system: straightpcf_cvm_12000_ep59_tmp
  model: straightpcf_cvm_tmp

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.00005

trainer:
  epochs: 30
YAML

python run.py --task configs/task/train_straightpcf_cvm_cache12000_ep59_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_straightpcf_cvm_cache12000_ep29_ft30.log
```

Checkpoint meanings:

```text
checkpoint_9.pkl  = total epoch 39
checkpoint_19.pkl = total epoch 49
checkpoint_29.pkl = total epoch 59
```

Scores pending.

## Prediction Data YAML

Use this for full local noisy test prediction.

```bash
cd /root/src/starter_code

cat > configs/data/predict_noisy_test_tmp.yaml <<'YAML'
predict_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: /root/autodl-tmp/noisy_test
    use_prob: False
    loader: npy
    data_name: noisy.npy
    ignore_check: True
    data_path:
      shapenet: [
        [./datalist/noisy_test.txt, 1.0],
      ]
YAML
```

## Full Prediction And Scoring Template

Set `CKPT` and `EXP`, then run the whole block.

```bash
cd /root/src/starter_code

CKPT=/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5/checkpoint_9.pkl
EXP=straightpcf_cvm_cache12000_total_ep29

cat > configs/task/predict_${EXP}_tmp.yaml <<YAML
mode: predict
debug: False
load_ckpt: $CKPT

components:
  data: predict_noisy_test_tmp
  transform: predict_npy
  system: vm
  model: straightpcf_cvm_tmp

writer:
  __target__: vm
  save_dir: /root/autodl-tmp/${EXP}_raw
  save_name: denoised
YAML

python run.py --task configs/task/predict_${EXP}_tmp.yaml

python tools/local_collect_predictions.py \
  --raw-dir /root/autodl-tmp/${EXP}_raw \
  --out-dir /root/autodl-tmp/${EXP} \
  --list datalist/noisy_test.txt \
  --input-prefix noisy_test \
  --overwrite

mkdir -p /root/autodl-tmp/${EXP}_score

python tools/score_cd_p2s_official.py \
  --pred_dir /root/autodl-tmp/${EXP} \
  --noisy_dir /root/autodl-tmp/noisy_test \
  --gt_dir /root/autodl-tmp/true_test \
  --mesh_dir /root/autodl-tmp/dataset_train \
  --out_dir /root/autodl-tmp/${EXP}_score \
  --workers 8 | tee /root/autodl-tmp/${EXP}_score/score.log
```

If collect cannot find predictions, use the fallback collector:

```bash
python tools/local_collect_predictions.py \
  --raw-dir /root/autodl-tmp \
  --out-dir /root/autodl-tmp/${EXP} \
  --list datalist/noisy_test.txt \
  --input-prefix noisy_test \
  --overwrite
```

## Half-Set Quick Scoring

Use this only for trend checking. It must not be compared directly against
previous full-100 scores unless the same half-set baseline is also measured.

Create a fixed half list:

```bash
cd /root/src/starter_code

python - <<'PY'
from pathlib import Path
src = Path("datalist/noisy_test.txt")
lines = [x.strip() for x in src.read_text().splitlines() if x.strip()]
quick = lines[::2]
Path("datalist/noisy_test_half.txt").write_text("\n".join(quick) + "\n")
print(len(quick), "/", len(lines))
PY

cat > configs/data/predict_noisy_test_half_tmp.yaml <<'YAML'
predict_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: /root/autodl-tmp/noisy_test
    use_prob: False
    loader: npy
    data_name: noisy.npy
    ignore_check: True
    data_path:
      shapenet: [
        [./datalist/noisy_test_half.txt, 1.0],
      ]
YAML
```

Then use the normal prediction template but replace:

```yaml
data: predict_noisy_test_tmp
```

with:

```yaml
data: predict_noisy_test_half_tmp
```

Important:

- Full set currently means the local `datalist/noisy_test.txt`, expected to be
  100 samples.
- Half set should be about 50 samples.
- Check actual counts with:

```bash
wc -l datalist/noisy_test.txt
wc -l datalist/noisy_test_half.txt
```

## Current CVM Result Table

All rows below use full local noisy test unless marked otherwise.

| Model | Checkpoint | CD | P2S | Final |
|---|---|---:|---:|---:|
| single Surface-Straight VM | `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl` | 54.0110 | 71.0060 | 62.5085 |
| official-style CVM total ep4 | `/root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep/checkpoint_4.pkl` | 52.9509 | 69.7321 | 61.3415 |
| official-style CVM total ep9 | `/root/autodl-tmp/experiments/straightpcf_cvm_from_b2ft10_cache12000_b2_lr5e5_10ep/checkpoint_9.pkl` | 52.4533 | 73.4140 | 62.9337 |
| official-style CVM total ep19 | `/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep9_ft10_lr5e5/checkpoint_9.pkl` | 53.7864 | 75.5082 | 64.6473 |
| official-style CVM total ep29 | `/root/autodl-tmp/experiments/straightpcf_cvm_cache12000_ep19_ft10_lr5e5/checkpoint_9.pkl` | 55.7714 | 76.3234 | 66.0474 |

## Next Planned Steps

1. Finish Stage 4 and measure total ep39, ep49, ep59.
2. If CVM continues improving, keep the best CVM checkpoint.
3. Train `StraightPCFVelocityDistanceModule` using the best CVM checkpoint.
4. Later, rebuild cache with additional noise variants or multi-seed cache, but
   keep the same reproduction structure:
   - change `CACHE`
   - change output experiment directories
   - keep common model YAML shape the same

## Notes For Future Noise Runs

Patch cache stores the sampled points, noise, patch construction, and time. If
the transform noise settings are changed, the cache must be rebuilt. Changing
YAML after a cache has been built does not change the data inside existing
`patch.npz` files.

Recommended naming pattern:

```text
/root/autodl-tmp/cache_surface_straight_nf12000_seed123
/root/autodl-tmp/cache_surface_straight_nf12000_seed456
/root/autodl-tmp/cache_surface_straight_mixed_nf12000_seed123
```

When using multiple cache lists, write a new data YAML with multiple entries:

```yaml
data_path:
  cache: [
    [/root/autodl-tmp/cache_surface_straight_nf12000_seed123/train_cache.txt, 1.0],
    [/root/autodl-tmp/cache_surface_straight_nf12000_seed456/train_cache.txt, 1.0],
  ]
```
