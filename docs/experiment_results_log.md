# 点云降噪实验结果记录

本文档用于记录本地固定测试集上的 checkpoint / 方法对比结果，后续每跑一次实验就在表格中追加一行。

## 本地测试集设置

| 字段 | 当前记录 |
|---|---|
| 数据来源 | `dataset_train` 中的 ShapeNet OBJ 网格 |
| clean 生成目录 | `/root/autodl-tmp/jittor_pcd/true_test` |
| noisy 生成目录 | `/root/autodl-tmp/jittor_pcd/noisy_test` |
| pred 生成目录 | `/root/autodl-tmp/jittor_pcd/result/epoch_<E>/denoisy_test` |
| result 目录 | `/root/autodl-tmp/jittor_pcd/result/epoch_<E>` |
| 样本数 | 200 |
| 每样本点数 | 50000 |
| 噪声类型 | Laplace |
| 噪声标准差范围 | 0.005 ~ 0.020 |
| 随机种子 | 2026 |
| 评测指标 | CD score, P2S score, final score |
| P2S 依赖 | `point-cloud-utils` |

## 结果总表

| 日期 | 实验名 | checkpoint | 方法说明 | 样本数 | CD noisy | CD pred | CD score | P2S noisy | P2S pred | P2S score | Final score | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-18 | baseline_epoch20 | 20 | 官方 baseline；修复 patch 推理少点问题后推理 | 200 | 0.0013460407 | 0.0010650638 | 23.2705 | 0.0011982170 | 0.0009123413 | 29.0703 | 26.1704 | 本地固定测试集；非官方榜单分数 |
| 2026-05-18 | baseline_epoch40 | 40 | 官方 baseline；修复 patch 推理少点问题后推理 | 200 | 0.0013460407 | 0.0009842924 | 30.3515 | 0.0011982170 | 0.0008311424 | 37.5546 | 33.9531 | 本地固定测试集；非官方榜单分数 |
| 2026-05-18 | baseline_epoch60 | 60 | 官方 baseline；修复 patch 推理少点问题后推理 | 200 | 0.0013460407 | 0.0009008771 | 36.6243 | 0.0011982170 | 0.0007409785 | 46.1594 | 41.3919 | 本地固定测试集；非官方榜单分数 |
| 2026-05-18 | baseline_epoch80 | 80 | 官方 baseline；修复 patch 推理少点问题后推理 | 200 | 0.0013460407 | 0.0008599098 | 39.8544 | 0.0011982170 | 0.0007058233 | 49.1402 | 44.4973 | 本地固定测试集；非官方榜单分数 |
| 2026-05-18 | baseline_epoch99 | 99 | 官方 baseline；修复 patch 推理少点问题后推理 | 200 | 0.0013460407 | 0.0008418997 | 39.8671 | 0.0011982170 | 0.0006708888 | 51.5912 | 45.7291 | 当前最优；本地固定测试集；非官方榜单分数 |

## 原始 Summary JSON

### baseline_epoch20

```json
{
  "num_samples": 200,
  "mean_cd_noisy": 0.0013460406706187183,
  "mean_cd_pred": 0.001065063835126115,
  "mean_cd_score": 23.270495979296566,
  "mean_p2s_noisy": 0.0011982169553669383,
  "mean_p2s_pred": 0.0009123413086998131,
  "mean_p2s_score": 29.07034946027071,
  "mean_final_score": 26.170422719783637
}
```

### baseline_epoch40

```json
{
  "num_samples": 200,
  "mean_cd_noisy": 0.0013460406706187183,
  "mean_cd_pred": 0.0009842923667902222,
  "mean_cd_score": 30.35151844093589,
  "mean_p2s_noisy": 0.0011982169553669383,
  "mean_p2s_pred": 0.000831142402221178,
  "mean_p2s_score": 37.55459806547049,
  "mean_final_score": 33.953058253203196
}
```

### baseline_epoch60

```json
{
  "num_samples": 200,
  "mean_cd_noisy": 0.0013460406706187183,
  "mean_cd_pred": 0.000900877131242607,
  "mean_cd_score": 36.62434194365391,
  "mean_p2s_noisy": 0.0011982169553669383,
  "mean_p2s_pred": 0.000740978488003673,
  "mean_p2s_score": 46.15939053200828,
  "mean_final_score": 41.39186623783109
}
```

### baseline_epoch80

```json
{
  "num_samples": 200,
  "mean_cd_noisy": 0.0013460406706187183,
  "mean_cd_pred": 0.0008599097751972818,
  "mean_cd_score": 39.8543816906828,
  "mean_p2s_noisy": 0.0011982169553669383,
  "mean_p2s_pred": 0.0007058233298808836,
  "mean_p2s_score": 49.14024247337729,
  "mean_final_score": 44.49731208203004
}
```

### baseline_epoch99

```json
{
  "num_samples": 200,
  "mean_cd_noisy": 0.0013460406706187183,
  "mean_cd_pred": 0.0008418997062499164,
  "mean_cd_score": 39.867092344529645,
  "mean_p2s_noisy": 0.0011982169553669383,
  "mean_p2s_pred": 0.0006708887608159127,
  "mean_p2s_score": 51.59117291845043,
  "mean_final_score": 45.72913263149004
}
```

## 解读

| checkpoint | CD score | P2S score | Final score | 结论 |
|---:|---:|---:|---:|---|
| 20 | 23.2705 | 29.0703 | 26.1704 | 明显未收敛 |
| 40 | 30.3515 | 37.5546 | 33.9531 | 继续改善 |
| 60 | 36.6243 | 46.1594 | 41.3919 | 明显提升 |
| 80 | 39.8544 | 49.1402 | 44.4973 | 接近 epoch99 |
| 99 | 39.8671 | 51.5912 | 45.7291 | 当前本地最优 |

从 20 到 99，CD/P2S/Final 基本单调提升；epoch80 到 epoch99 的 CD score 几乎持平，但 P2S score 继续提升约 2.45 分，因此当前本地固定测试集上应优先选择 `checkpoint_99.pkl`。这不是官方 A 榜分数，只用于本地 checkpoint 选择和方法对比。

## 后续追加模板

| 日期 | 实验名 | checkpoint | 方法说明 | 样本数 | CD noisy | CD pred | CD score | P2S noisy | P2S pred | P2S score | Final score | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | method_name_epochXX | XX | 简述改动 | 200 |  |  |  |  |  |  |  |  |
