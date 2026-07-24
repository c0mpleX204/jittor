# Risk-Aware CDRefine Attention Runbook

Date: 2026-07-24

This note supersedes the older CVM-only planning notes for the current
submission candidate. Keep the v1.4a result as the safe candidate before trying
new experiments.

## Current Best

- Pipeline: `noisy -> v1.3b DM -> v1.4a RiskAwareCDRefine`
- v1.3b DM checkpoint:
  `/root/autodl-tmp/experiments/v1.3b_dm/checkpoint_0.pkl`
- v1.4a CDRefine checkpoint:
  `/root/autodl-tmp/experiments/v1.4a_cdrefine_attn/checkpoint_3.pkl`
- Full local test score:
  - CD: `65.4856`
  - P2S: `82.0600`
  - Final: `73.7728`
- Current submission candidate:
  `/root/autodl-tmp/result.zip`

## Sync

```bash
TMP=/root/src/jittor_surface_straight_branch

cd "$TMP"
git fetch origin exp/surface-straight-vm
git checkout exp/surface-straight-vm
git reset --hard origin/exp/surface-straight-vm

rsync -av "$TMP/release4/starter_code/" /root/src/starter_code/

cd /root/src/starter_code
python -m py_compile src/model/cd_refine.py src/model/parse.py \
  tools/build_patch_cache.py tools/build_refine_cache.py
```

## Refine Cache

Training uses the existing refine cache:

```text
/root/autodl-tmp/cache_refine_v1.3b_dm_edge_mix6w
```

It should contain:

```text
pc_noisy
pc_mix
pc_clean
pc_clean_corr
pc_stage1
pc_edge_risk
pc_normal
```

Quick check:

```bash
python - <<'PY'
import numpy as np
p = "/root/autodl-tmp/cache_refine_v1.3b_dm_edge_mix6w/patches/00000000/patch.npz"
d = np.load(p)
print(d.files)
print({k: d[k].shape for k in d.files})
PY
```

## Reproduce v1.4a Training

The committed v1.4a configs are reconstructed from the current code defaults
and latest run notes because the original cloud temp YAML was not present in
the local repo. Treat the measured checkpoint and score as source of truth if
the old cloud temp YAML differs.

```bash
cd /root/src/starter_code
python run.py --task configs/task/train_v1_4a_cdrefine_attn_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_v1.4a_cdrefine_attn.log
```

The measured best remains:

```text
/root/autodl-tmp/experiments/v1.4a_cdrefine_attn/checkpoint_3.pkl
```

Do not overwrite the existing best checkpoint unless intentionally rerunning the
same experiment.

## Current Priority: Repair CD

The official submission returned:

```text
score=65.44
CD_score=50.80
P2S_score=80.08
mean_CD_pred=0.000122
mean_CD_noisy=0.000246
mean_P2S_pred=0.000067
mean_P2S_noisy=0.000196
```

This means P2S is not the bottleneck. CD must be repaired directly; target CD
score should be at least `70`. Deprioritize v1.4b P2S-protection unless CD is
already fixed.

### Step 1: No-Training Weighted Patch Inference

This uses the same v1.4a checkpoint but replaces single-best-patch output
selection with soft weighted multi-patch aggregation. It is the fastest test
because it does not retrain.

```bash
cd /root/src/starter_code

python run.py --task configs/task/predict_noisy_test_v1_4a_cdrefine_attn_weighted_tmp.yaml

python tools/local_collect_predictions.py \
  --raw-dir /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test_raw \
  --out-dir /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test \
  --list datalist/noisy_test.txt \
  --input-prefix noisy_test \
  --overwrite

mkdir -p /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test_score

python tools/score_cd_p2s_official.py \
  --pred_dir /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test \
  --noisy_dir /root/autodl-tmp/noisy_test \
  --gt_dir /root/autodl-tmp/true_test \
  --mesh_dir /root/autodl-tmp/dataset_train \
  --out_dir /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test_score \
  --workers 8 | tee /root/autodl-tmp/v1.4a_cdrefine_attn_weighted_noisy_test_score/score.log
```

If CD improves, sweep `predict_patch_weight_temperature` around `0.20`, `0.35`,
`0.50`, and `0.80`.

### Step 2: v1.4e Wide-CD Surface-Constrained Fine-Tune

This is the most direct response to the official low-CD result. The hypothesis:
`delta_scale=0.02` was too small for lateral distribution repair, but leaving
the surface must still be discouraged.

v1.4e keeps the same set-CD idea as v1.4d, but uses:

- larger residual budget: `delta_scale=0.08`
- no pointwise correspondence anchor
- no pointwise surface anchor
- stronger one-sided surface-set penalty: `surface_set_weight=0.45`
- light density matching
- no normal/tangent hard movement restriction

Run:

```bash
cd /root/src/starter_code

python run.py --task configs/task/train_v1_4e_cdrefine_attn_wide_surface_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_v1.4e_cdrefine_attn_wide_surface.log
```

Score every checkpoint. If P2S collapses, reduce `delta_scale` to `0.06` or
raise `surface_set_weight` to `0.70`. If CD barely moves, lower
`surface_set_weight` to `0.25`.

### Fallback: v1.4d Set-CD Soft-Surface Fine-Tune

This is the preferred CD repair experiment after the official `CD_score=50.80`
result. It stops treating the original clean correspondence as a strict
pointwise regression target. Instead:

- main supervision is set-level Chamfer/CD to `pc_clean_corr`
- pointwise `point_anchor` is disabled
- pointwise `surface_anchor` is disabled
- surface adherence is only a soft one-sided set penalty
- normal/tangent movement restrictions are disabled
- weighted patch aggregation is used at prediction time

Run:

```bash
cd /root/src/starter_code

python run.py --task configs/task/train_v1_4d_cdrefine_attn_cdset_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_v1.4d_cdrefine_attn_cdset.log
```

Use v1.4d if v1.4e moves too aggressively.

### Fallback: v1.4c CD-Heavy Fine-Tune

This continues from v1.4a and pushes CD harder:

- larger residual budget: `delta_scale=0.035`
- stronger Chamfer: `chamfer_num_points=512`, `chamfer_loss_weight=2.0`
- light density matching: `density_loss_weight=0.03`
- tangent delta penalty: allow normal correction but reduce sliding along the
  surface, which can hurt coverage
- weighted patch aggregation at prediction time

```bash
cd /root/src/starter_code

python run.py --task configs/task/train_v1_4c_cdrefine_attn_cd_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_v1.4c_cdrefine_attn_cd.log
```

This is now a fallback because it still includes directional delta penalties.
Use v1.4d first.

## Full Local Scoring

```bash
cd /root/src/starter_code

EXP=v1.4e_cdrefine_attn_wide_surface_noisy_test

python run.py --task configs/task/predict_noisy_test_v1_4e_cdrefine_attn_wide_surface_tmp.yaml

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

For v1.4a verification, use:

```bash
python run.py --task configs/task/predict_noisy_test_v1_4a_cdrefine_attn_tmp.yaml
```

and change `EXP` to `v1.4a_cdrefine_attn_noisy_test`.

## Official Packaging

Generate raw predictions:

```bash
cd /root/src/starter_code
python run.py --task configs/task/predict_submit_v1_4a_cdrefine_attn_tmp.yaml
```

Collect and package with `shapenet/` as the only top-level zip directory:

```bash
python tools/local_collect_predictions.py \
  --raw-dir /root/autodl-tmp/v1.4a_official_submit_raw \
  --out-dir /root/autodl-tmp/v1.4a_official_submit \
  --list datalist/test.txt \
  --input-prefix dataset_test_noisy \
  --overwrite

cd /root/autodl-tmp/v1.4a_official_submit
zip -r /root/autodl-tmp/result.zip shapenet

unzip -t /root/autodl-tmp/result.zip
unzip -l /root/autodl-tmp/result.zip | grep 'denoised.npy$' | wc -l
unzip -l /root/autodl-tmp/result.zip | awk 'NR==4 {print $4}'
```

Expected:

- exactly `200` `denoised.npy` files
- first listed top-level path starts with `shapenet/`

## Caution

The code supports risk-aware local attention through `pc_edge_risk`. Do not add
direct noisy residual conditioning as the next default experiment; earlier direct
`pc_noisy - pc_stage1` conditioning failed. If trying it later, keep it as an
explicit ablation and score on the full local set.
