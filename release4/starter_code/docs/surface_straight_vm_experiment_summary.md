# Surface-Straight VM Experiment Summary

Status: historical as of 2026-07-24. The current best submission candidate is
`v1.4a RiskAwareCDRefine`; see
`docs/cdrefine_attention_v1_4a_runbook.md`. Keep this document for the earlier
single-VM and CVM progression details only.

Date: 2026-07-19

## Repository State

- Local repo: `D:\LiMinglong\g2\jittor`
- Cloud run dir: `/root/src/starter_code`
- Active branch: `exp/surface-straight-vm`
- Latest pushed commit: `18ba3a8 Keep VM predictions under save_dir for absolute inputs`
- Important commit chain:
  - `6041ac3 Add surface straight VM experiment`
  - `1b5df05 Reduce surface straight VM memory footprint`
  - `c4ea0ea Add low-memory surface straight training config`
  - `18ba3a8 Keep VM predictions under save_dir for absolute inputs`

## Motivation

Old v1/v2 failed because they were only shape-similar to PCF/StraightPCF. They did not reproduce the actual StraightPCF training setup:

- patch-level shared interpolation time `t`
- fixed straight velocity target from high-noise endpoint to target endpoint
- stable VM stage before any distance module

This branch replaces the failed v1 gated single-step delta with a Surface-Straight VM stage.

## Main Code Changes

- `src/model/dcvm.py`
  - Replaced old gated delta model with Surface-Straight VM.
  - Training target:
    - `pc_current = t * pc_surface + (1 - t) * pc_noisy_l2`
    - `target_velocity = pc_surface - pc_noisy_l2`
  - Inference uses 4 denoise steps by default:
    - `pcl_next += (1 / denoise_steps) * pred_velocity`
  - Current model has no separate DistanceModule yet. It implicitly uses distance ratio `r = 1.0`.

- `src/data/augment.py`
  - `AugmentAddNoise` can create a fixed high-noise endpoint via `l2_noise_std`.
  - `AugmentPatch` supports:
    - `straight_time: True` for one shared `t` per patch.
    - `surface_target: True` for nearest sampled clean surface endpoint.
    - `use_noisy_l2: True` for StraightPCF-style high-noise endpoint.
  - Fixed `linear` augmentation so sampled clean/noisy/L2 points are transformed too.

- `configs/transform/surface_straight.yaml`
  - New transform for Surface-Straight VM.

- `configs/data/train_surface_straight_autodl_tmp.yaml`
  - Low-memory cloud training config:
    - `batch_size: 1` initially, later manually tested `batch_size: 2`
    - `num_workers: 0`
    - `num_files` adjusted per experiment

- `src/system/vm.py`
  - Fixed writer behavior for absolute input paths. Without this fix, predictions may be written into `/root/autodl-tmp/noisy_test/.../denoised.npy`.

## Baseline And Failed Branches

| Experiment | CD score | P2S score | Final score | Notes |
|---|---:|---:|---:|---|
| baseline VM | about 40 | about 40 | about 40 | Original baseline reference |
| DCVM v1 | 33.0016 | 38.8159 | 35.9087 | Failed; gated single-step delta |
| DCVM v2 | 28.2347 | 33.2452 | 30.7399 | Failed; nearest clean/surface target alone was not enough |

## Surface-Straight VM Results

Local score set:

- GT: `/root/autodl-tmp/true_test`
- noisy: `/root/autodl-tmp/noisy_test`
- mesh: `/root/autodl-tmp/dataset_train`
- scoring script: `tools/score_cd_p2s_official.py`

| Experiment | Checkpoint | CD score | P2S score | Final score | Notes |
|---|---|---:|---:|---:|---|
| `surface_straight_vm` | `/root/autodl-tmp/experiments/surface_straight_vm/checkpoint_19.pkl` | 37.5307 | 45.9111 | 41.7209 | Low-memory first run, `num_files=1000`, `batch=1`, 20 epochs |
| `nf3000_b1_e33` | `/root/autodl-tmp/experiments/surface_straight_vm_nf3000_b1_e50/checkpoint_32.pkl` | 47.8046 | 57.5400 | 52.6723 | 33 completed epochs; major improvement |
| `b2_ft10` | `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl` | 54.0110 | 71.0060 | 62.5085 | Current best; continued from checkpoint 32 with `batch=2`, `lr=5e-5`, 10 more epochs |
| `b2_ft20` | `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_19.pkl` | 53.5667 | 70.9049 | 62.2358 | Slightly worse than ft10 |
| `b2_ft40` | `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_39.pkl` | 52.7489 | 71.5232 | 62.1361 | Higher P2S, lower CD; final worse |
| `lr2e-5_ft5` | `/root/autodl-tmp/experiments/surface_straight_vm_from_b2ft10_nf3000_b2_lr2e5_10ep/checkpoint_4.pkl` | 51.4271 | 70.2633 | 60.8452 | Small-lr fine-tune from best got worse |

Current best:

- Name: `b2_ft10`
- Checkpoint: `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl`
- Score: `Mean final score: 62.5085`
- Interpretation: Continuing from the 52.67 checkpoint with `batch_size=2` and `lr=5e-5` improves both CD and P2S sharply. Longer training then slightly hurts final score because CD drops while P2S stays high.

## Observations

1. Surface-Straight VM is clearly valid.
   - It moved from low 30s/40ish to 62.5.

2. The model currently only learns velocity.
   - It predicts a 3D vector containing direction and magnitude.
   - Inference always walks the predicted velocity with implicit ratio `r = 1.0`.

3. Longer training after `b2_ft10` does not improve final score.
   - `b2_ft40` has the best P2S but worse CD.
   - This suggests over-projecting toward the surface may hurt point distribution / CD.

4. Small-lr fine-tuning did not help.
   - `lr=2e-5` from `b2_ft10` dropped to 60.85.
   - The current limit is likely not simple convergence.

5. Memory behavior matters.
   - `frame_knn=32`, larger batch, or many workers can OOM in Jittor.
   - Stable settings so far:
     - `frame_knn: 16`
     - `batch_size: 1 or 2`
     - `num_workers: 0`

6. Cloud writer caveat.
   - If the `18ba3a8` writer fix is not synced, predictions with absolute `input_dataset_dir` may be written into `/root/autodl-tmp/noisy_test/.../denoised.npy`.
   - In that case collect with:
     - `--raw-dir /root/autodl-tmp`
     - `--input-prefix noisy_test`

## Recommended Next Steps

Priority 1: inference scale search using current best checkpoint.

- Best checkpoint: `/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl`
- Try scale values without retraining:
  - `0.75`, `0.85`, `0.95`, `1.05`
- Current inference is effectively:
  - `pcl_next += (1.0 / denoise_steps) * pred_velocity`
- Search should test:
  - `pcl_next += (scale / denoise_steps) * pred_velocity`

Priority 2: prediction ensemble / checkpoint averaging.

- Candidates:
  - `b2_ft10`: best final, high CD and P2S
  - `b2_ft20`: close final
  - `b2_ft40`: highest P2S, lower CD
- Since point order is preserved, try pointwise averaged predictions:
  - `0.6 * ft10 + 0.2 * ft20 + 0.2 * ft40`
  - `0.7 * ft10 + 0.3 * ft40`

Priority 3: implement StraightPCF-style DistanceModule.

- Freeze or load current best VM.
- Train a second model to predict patch-level distance ratio `r`.
- Final inference:
  - `p_i_new = p_i + r * v_i`
- This should address the current CD/P2S tradeoff:
  - some patches should walk less than 1.0
  - some may walk more than 1.0

Priority 4: only after DistanceModule, consider global attention / global surface prior.

## Useful Cloud Commands

Score current best if predictions need to be regenerated:

```bash
cd /root/src/starter_code

find /root/autodl-tmp/noisy_test -name denoised.npy -delete

cat > configs/task/predict_noisy_test_dcvm_autodl_tmp.yaml <<'YAML'
mode: predict
debug: False
load_ckpt: /root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl

components:
  data: predict_noisy_test
  transform: predict_npy
  system: vm
  model: dcvm

writer:
  __target__: vm
  save_dir: /root/autodl-tmp/b2_ft10_denoisy_test_raw
  save_name: denoised
YAML

python run.py --task configs/task/predict_noisy_test_dcvm_autodl_tmp.yaml

python tools/local_collect_predictions.py \
  --raw-dir /root/autodl-tmp \
  --out-dir /root/autodl-tmp/b2_ft10_denoisy_test \
  --list datalist/noisy_test.txt \
  --input-prefix noisy_test \
  --overwrite

mkdir -p /root/autodl-tmp/b2_ft10_score

python tools/score_cd_p2s_official.py \
  --pred_dir /root/autodl-tmp/b2_ft10_denoisy_test \
  --noisy_dir /root/autodl-tmp/noisy_test \
  --gt_dir /root/autodl-tmp/true_test \
  --mesh_dir /root/autodl-tmp/dataset_train \
  --out_dir /root/autodl-tmp/b2_ft10_score \
  --workers 8 | tee /root/autodl-tmp/b2_ft10_score/score.log
```

## Prompt For New Conversation

Use this prompt in a new Codex conversation:

```text
我们在做 Jittor 点云去噪正赛项目，仓库本地路径 D:\LiMinglong\g2\jittor，云端运行路径 /root/src/starter_code。任务是输入 noisy 点云输出 denoised 点云，评分是 CD 和 P2S，final 约等于二者平均改善。

请不要凭记忆假设，先读仓库代码再行动。重要代码在 release4/starter_code。

当前有效主线是 Surface-Straight VM，分支 exp/surface-straight-vm，最新 pushed commit 18ba3a8。核心文件：
- src/model/dcvm.py
- src/data/augment.py
- src/system/vm.py
- configs/transform/surface_straight.yaml
- configs/data/train_surface_straight_autodl_tmp.yaml

历史结果：
- baseline VM 本地评分约 40。
- DCVM v1: CD 33.0016, P2S 38.8159, final 35.9087，失败。
- DCVM v2: CD 28.2347, P2S 33.2452, final 30.7399，失败。
- Surface-Straight first run checkpoint_19: CD 37.5307, P2S 45.9111, final 41.7209。
- Surface-Straight nf3000_b1 checkpoint_32: CD 47.8046, P2S 57.5400, final 52.6723。
- 当前最佳 b2_ft10 checkpoint_9: CD 54.0110, P2S 71.0060, final 62.5085。
- b2_ft20: final 62.2358。
- b2_ft40: CD 52.7489, P2S 71.5232, final 62.1361。
- lr2e-5 从最佳继续 ft5: final 60.8452，说明小 lr 微调没有提升。

当前最佳权重：
/root/autodl-tmp/experiments/surface_straight_vm_from32_nf3000_b2_lr5e5_40ep/checkpoint_9.pkl

当前模型只做 Velocity / Direction。训练目标是 nearest surface endpoint 的 straight velocity：
current = t * surface + (1 - t) * noisy_L2
target_velocity = surface - noisy_L2
推理默认 denoise_steps=4，隐含 distance ratio r=1.0。

重要现象：
- 继续训练后 P2S 可能继续高，但 CD 下降，final 不升。
- 当前上限瓶颈大概率是缺少 distance/step 控制，不是简单继续训练。
- 下一步优先做不训练的推理 scale 搜索：scale = 0.75, 0.85, 0.95, 1.05，把推理更新从 (1/steps)*velocity 改成 (scale/steps)*velocity。
- 然后尝试 checkpoint prediction ensemble：ft10/ft20/ft40 的点序一致，可以做点对点平均。
- 再实现 StraightPCF-style DistanceModule：冻结/加载最佳 VM，训练第二个 model 输出 patch-level ratio r，用 p_new = p + r * v。

注意：
- 4090 上 Jittor 容易因为 DGCNN/KNN + worker 内存放大 OOM。稳定训练设置为 frame_knn=16, batch_size=1/2, num_workers=0。
- 如果云端没有同步 writer 修复，预测可能写到 /root/autodl-tmp/noisy_test/.../denoised.npy。collect 时可用 --raw-dir /root/autodl-tmp --input-prefix noisy_test 兜底。

请从读代码开始，然后优先实现和验证推理 scale 搜索，不要先开大训练。
```
