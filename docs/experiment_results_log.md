# 点云降噪实验结果记录

本文档用于记录本地固定测试集上的 checkpoint / 方法对比结果，后续每跑一次实验就在表格中追加一行。

## 本地测试集设置

| 字段 | 当前记录 |
|---|---|
| 数据来源 | `dataset_train` 中的 ShapeNet OBJ 网格 |
| clean 生成目录 | `/root/autodl-tmp/jittor_pcd/true_test` |
| noisy 生成目录 | `/root/autodl-tmp/jittor_pcd/noisy_test` |
| pred 生成目录 | `/root/autodl-tmp/jittor_pcd/denoisy_test` |
| result 目录 | `/root/autodl-tmp/jittor_pcd/result/epoch_99` |
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
| 2026-05-18 | baseline_epoch99 | 99 | 官方 baseline；修复 patch 推理少点问题后重新推理 | 200 | 0.0013460407 | 0.0008418997 | 39.8671 | 0.0011982170 | 0.0006708888 | 51.5912 | 45.7291 | 本地固定测试集；非官方榜单分数 |

## 原始 Summary JSON

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

| 指标 | 含义 | baseline_epoch99 |
|---|---|---:|
| `mean_cd_score` | 相比 noisy input 的 CD 改善比例映射到百分制 | 39.8671 |
| `mean_p2s_score` | 相比 noisy input 的 P2S 改善比例映射到百分制 | 51.5912 |
| `mean_final_score` | CD score 与 P2S score 各占 50% 的综合分 | 45.7291 |

该分数表示 baseline 在当前本地测试集上将综合误差大约降低了 45.7%。这不是官方 A 榜分数，只用于本地 checkpoint 选择和方法对比。

## 后续追加模板

| 日期 | 实验名 | checkpoint | 方法说明 | 样本数 | CD noisy | CD pred | CD score | P2S noisy | P2S pred | P2S score | Final score | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | method_name_epochXX | XX | 简述改动 | 200 |  |  |  |  |  |  |  |  |

