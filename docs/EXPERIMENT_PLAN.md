# 点云降噪鲁棒性实验计划

本计划默认：本地只改代码和文档；训练、推理、指标统计和批量导出都放到云算力平台执行。

## 0. 总目标

最终要证明三件事：

1. 普通 Gaussian noise 下，我们不明显输给 StraightPCF / 官方 baseline。
2. 非高斯噪声、局部强噪声、离群点、扫描线噪声下，我们更稳。
3. 边缘棱角、薄结构、未见类别上，我们的局部误差更小。

实验主线：

> 先诊断 StraightPCF / baseline 的失败模式，再逐步加入 mixed noise training、point-wise distance/confidence、geometry confidence。每一步都要有消融对照，不靠单次提交分数讲故事。

## 1. 产物目录

建议所有实验输出统一放在：

```text
outputs/
  checkpoints/          # 模型权重
  predictions/          # denoised npy / ply
  metrics/              # csv, json
  visualization/
    ply/                # CloudCompare 可读文件
    png/                # 自动截图和热力图
  reports/              # 每周观察记录
```

每次实验命名建议：

```text
YYYYMMDD_model_noise_trainset_note

例：
20260520_baseline_gaussian_val50
20260527_vm_mixed_noise_v1_val50
20260608_vm_mixed_pointdist_v2_val50
```

## 2. 阶段进度

### 第 0 阶段：云端准备

时间：0.5～1 天。

目标：

- 云平台能跑 Jittor。
- baseline 训练和推理命令能启动。
- 分析环境能导入 `numpy / scipy / trimesh / point-cloud-utils / pandas / matplotlib`。

产出：

- `jittor-pcd` 环境。
- `pcd-analysis` 环境。
- 一份云端环境记录：GPU 型号、CUDA、Python、Jittor 版本。

通过标准：

- `python -m jittor.test.test_cuda` 通过。
- baseline 的 `run.py --task configs/task/train_vm.yaml` 至少能启动一个 epoch 或一个 debug batch。

### 第 1 阶段：数据与工具，不改模型

时间：第 1 周。

目标：

- 建小验证集。
- 写 clean point sampling 脚本。
- 写 6 类噪声生成脚本。
- 写 CD / P2S / 分区域指标脚本。
- 写 `.ply` 导出脚本。

验证集划分：

- 普通物体：10 个。
- 边缘明显：10 个。
- 薄结构：10 个。
- 复杂结构：10 个。
- 未见类别或少见类别：10 个。

每个 mesh 采样：

- 50K points。
- 20K points。
- 10K points。

噪声类型：

- Gaussian：`sigma = 0.005 / 0.010 / 0.020 / 0.030`。
- Laplace：重尾噪声。
- Local strong：随机局部区域噪声放大 2～4 倍。
- Outlier：1% / 3% / 5% 点大偏移。
- Anisotropic：某一方向噪声更强，例如 z 方向 3 倍。
- Scanline：按坐标轴分层施加系统偏移或条带扰动。

产出：

- `data/val_small/`。
- `data/noise_benchmark/`。
- `outputs/metrics/noisy_metrics.csv`。
- 至少 1 组 Noisy / Clean / Error heatmap `.ply`。

通过标准：

- 不训练模型，也能对 noisy input 计算 CD/P2S。
- CloudCompare 能打开导出的 `.ply`。

### 第 2 阶段：Baseline / StraightPCF 诊断

时间：第 2 周。

目标：

- 跑 baseline 在 6 类噪声上的表现。
- 找出清晰 failure cases。
- 判断后续改法应该优先解决方向、距离、边缘平滑还是离群点污染。

实验：

- Noisy input。
- StraightPCF / official baseline。
- Oracle projection：noisy 点投影到最近 mesh 表面，只作为理论参考上限。

统计表：

```text
noise_type, cd_noisy, cd_baseline, cd_oracle, p2s_noisy, p2s_baseline, p2s_oracle
Gaussian
Laplace
Local strong
Outlier
Anisotropic
Scanline
```

重点观察：

- 非高斯噪声下是否仍有红色远离表面的点。
- 局部强噪声区是否移动不足。
- 离群点是否污染邻域。
- 边缘棱角是否被抹平。
- 薄结构两侧是否被糊到一起。
- 未见类别局部结构是否不稳定。
- P2S 好但 CD 差，或 CD 好但 P2S 差的样本。

产出：

- `outputs/metrics/baseline_diagnosis.csv`。
- `outputs/reports/week2_failure_cases.md`。
- 10 个 failure case 的 `.ply` 和截图。

通过标准：

- 能明确写出 baseline 最主要的 2～3 类失败模式。
- 后续 Version 1/2/3 的目标能对应到这些失败模式。

### 第 3 阶段：Version 1，Mixed Non-Gaussian Training

时间：第 3 周。

目标：

- 先不改模型结构，只改训练噪声。
- 验证混合非高斯噪声增强是否真的提高鲁棒性。

对照：

- A：原 baseline 训练。
- B：Gaussian-only 训练。
- C：Mixed non-Gaussian 训练。

重点指标：

- Gaussian CD/P2S 是否没有明显退化。
- Laplace / Local strong / Outlier / Scanline 是否提升。
- 未见类别是否更稳。

产出：

- `outputs/metrics/v1_mixed_noise_ablation.csv`。
- 一张噪声类型分组表。
- 一组 heatmap 对比图。

通过标准：

- Mixed noise 在至少 3 类非高斯噪声上稳定优于 Gaussian-only。
- Gaussian 指标不能明显崩。

### 第 4 阶段：Version 2，Point-wise Distance / Confidence

时间：第 4～5 周。

目标：

- 让每个点有自己的移动幅度或置信度。
- 解决“同一个 patch 内噪声强度不一致”的问题。

形式：

```text
Delta_i = d_i * v_i
```

其中：

- `v_i`：straight velocity / direction。
- `d_i`：每个点自己的移动幅度或噪声强度估计。

对照：

- A：baseline。
- B：mixed noise。
- C：mixed noise + point-wise distance/confidence。

重点指标：

- Local strong noise。
- Outlier noise。
- Edge / corner 区域 P2S。
- 推理时间和显存，作为副指标记录，不作为这一阶段主目标。

产出：

- `outputs/metrics/v2_pointwise_ablation.csv`。
- `outputs/reports/week5_pointwise_observation.md`。
- 局部强噪声和离群点 close-up。

通过标准：

- Local strong / Outlier 的 P2S 明显改善。
- 没有明显牺牲普通 Gaussian 表现。

### 第 5 阶段：Version 3，Geometry-aware Edge / Corner

时间：第 6～7 周。

目标：

- 加入局部几何信息或几何置信度。
- 重点改善边缘、棱角、薄结构。

可用特征：

- 局部密度。
- KNN PCA 特征值比例。
- 曲率 proxy。
- 法向 proxy。
- edge confidence。

形式：

```text
Delta_i = c_i * d_i * v_i
```

其中：

- `c_i`：几何置信度，用来控制边缘棱角附近的移动。

对照：

- A：baseline。
- B：mixed noise。
- C：mixed noise + point-wise distance。
- D：mixed noise + point-wise distance + geometry confidence。

重点指标：

- `P2S_edge`。
- `P2S_corner`。
- `P2S_thin`。
- 局部 close-up 和热力图。

产出：

- `outputs/metrics/v3_geometry_ablation.csv`。
- 边缘 / 棱角 / 薄结构 close-up 图。
- 一页模块贡献分析。

通过标准：

- edge/corner/thin 区域提升大于 flat 区域提升。
- 可视化上能看到边缘保留更好，而不是整体被抹平。

### 第 6 阶段：Unseen Category 泛化

时间：第 8 周。

目标：

- 证明方法不是只记住训练类别。
- 按类别划分 train / validation，留出一部分类别只在测试出现。

实验：

- Seen category validation。
- Unseen category validation。
- 6 类噪声下分别统计。

产出：

- `outputs/metrics/unseen_category.csv`。
- unseen category 的 Noisy / Baseline / Ours / GT 对比图。
- 一段结论：在哪些未见类别上提升明显，哪些仍失败。

通过标准：

- Ours 在 unseen category 上至少在非高斯噪声和局部结构指标中优于 baseline。
- 能解释失败样本，而不是只报平均数。

### 第 7 阶段：报告、消融和最终提交

时间：第 9 周及以后。

目标：

- 整理最终实验表。
- 固定最好模型。
- 准备答辩图和提交文件。
- 只有效果成立后，再考虑效率优化。

必备图表：

- 方法流程图。
- 噪声鲁棒性表格。
- 边缘棱角 close-up。
- 误差热力图。
- 未见类别结果。
- 消融实验。

效率优化候选：

- 减少 KNN 次数。
- 减少 patch overlap。
- 减少迭代次数。
- 蒸馏模型。
- 减少 hidden dim。
- 混合精度。
- 批量推理优化。

通过标准：

- 最终模型能在云端完整推理测试集。
- `result.zip` 格式正确。
- 报告里能用实验支撑三条核心结论。

## 3. 每周检查点

每周结束必须回答：

```text
1. 本周新增了什么代码或实验能力？
2. 本周跑了哪些实验？
3. 哪些指标提升，哪些指标下降？
4. 失败样本是什么？
5. 下一周优先解决哪个具体问题？
```

建议每周写到：

```text
outputs/reports/weekN_summary.md
```

## 4. 决策规则

- 如果 baseline 与 Oracle projection 很接近，该场景优化空间小，少投入。
- 如果 baseline 与 Oracle projection 差很多，优先作为 failure case。
- Version 1 不成立时，不急着做复杂结构，先检查噪声生成和训练分布。
- Version 2 主要看局部强噪声和离群点，不用指望它完全解决边缘抹平。
- Version 3 必须用局部指标证明，不只看全局 CD/P2S。
- 效率优化放到效果成立之后。

## 5. 核心表述

最终报告可以围绕这句话展开：

> 我们首先发现 StraightPCF 在规则高斯噪声下表现强，但在非高斯噪声、局部强噪声、离群点和复杂边缘区域中仍存在不稳定现象。为此，我们引入混合噪声训练、逐点距离估计和几何置信度控制，使模型在保持 straight-flow 高效性的同时，提高对未知噪声和未见类别的鲁棒性。
