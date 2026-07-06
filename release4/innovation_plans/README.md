# 点云降噪改进方案包

目的：这里不是调参清单，也不是“加几个模块然后包装成创新”。每个方向都从当前 baseline 的可视化现象出发，尝试改变点云降噪的建模方式。

当前观察：

```text
p2s_improvement.ply 基本全绿
noisy_p2s_heatmap 和 denoised_p2s_heatmap 都是蓝绿
denoised 仍然比较散，外层绿色点明显
```

这个现象更像：

```text
移动方向大体正确，但移动距离、收敛程度或表面投影机制不足。
```

因此优先考虑“如何让点更准确地回到表面”，而不是继续堆 epoch、调学习率、换噪声类型。

## 方案总览

| 文件 | 方向 | 核心改变 | 预期收益 | 风险 |
|---|---|---|---|---|
| `01_distance_calibrated_iterative.md` | 距离校准的方向-距离解耦 | 从直接回归 3D 位移，改为方向、距离、置信度分解，并做粗到细迭代 | 最贴近当前失败现象，改动中等 | 容易被误写成 StraightPCF 变体，需要强调距离校准和迭代误差反馈 |
| `02_neural_local_surface_projection.md` | 神经局部曲面投影 | 不直接预测 offset，而是预测局部隐式曲面，再把点投影回曲面 | 建模更硬，论文味强 | 实现较难，P2S/CD 不一定立刻涨 |
| `03_thickness_collapse_regularization.md` | 局部噪声厚度压缩 | 直接针对“点云仍然散”的壳层厚度问题建模 | 针对可视化症状明确 | 要小心不要把薄结构压坏 |
| `04_energy_selected_candidates.md` | 能量选择的多候选修复 | 每个点预测多个候选修复位置，用局部几何能量选择 | 处理歧义表面和局部错误方向 | 比单 offset 复杂，训练稳定性要验证 |

## 不要自欺的判断标准

一个方向能不能算创新，不看名字，而看它有没有回答一个具体失败模式：

| 失败模式 | 对应方案 |
|---|---|
| 全绿但仍散，说明方向对但距离不够 | 方案 01、03 |
| 点被拉近但没有形成清晰薄表面 | 方案 02、03 |
| 局部复杂结构可能有多个合理投影方向 | 方案 04 |
| 直接 displacement 学到保守平均位移 | 方案 01、04 |

## 相关工作边界

这些方向不是无人做过。相关背景包括：

- StraightPCF：方向/速度与距离标量的直线路径思想。
  https://openaccess.thecvf.com/content/CVPR2024/html/de_Silva_Edirimuni_StraightPCF_Straight_Point_Cloud_Filtering_CVPR_2024_paper.html
- PointCleanNet / PointFilter：直接预测 correction vector / displacement 的经典路线。
- MODNet：多 offset、多尺度融合思想。
  https://arxiv.org/abs/2208.14160
- DMRDenoise：从 displacement 转向 manifold reconstruction。
  https://luost.me/DMRDenoise/

所以这里的目标不是宣称“没人做过”，而是：

```text
针对当前 baseline 的具体失败现象，设计一个不同于简单 displacement regression 的机制。
```

## 代码入口

baseline 主要入口：

```text
starter_code/src/model/vm.py
starter_code/src/model/feature.py
starter_code/src/model/parse.py
starter_code/configs/model/vm.yaml
```

最稳妥做法：

```text
不要直接破坏 VelocityModule。
新建一个模型文件，例如 src/model/dcvm.py / surface_projection.py。
在 src/model/parse.py 里注册新模型。
新增 configs/model/*.yaml。
task 里只改 components.model。
```

这样 baseline 仍能复现实验，改进模型可以单独对比。

