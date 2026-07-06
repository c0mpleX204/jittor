# 潜在研究方向记录

更新日期：2026-05-18

## 先说结论

这些方向不是“完全没人做过”。点云降噪已经有很多相近思路：位移回归、方向/距离解耦、多尺度 offset、manifold 重建、逐点停止等都能找到相关论文。真正能做成创新，不能只靠叙事，而要先找到 baseline 或 StraightPCF 的具体失败模式，再提出一个能解释、能验证、能消融的机制。

参考文献入口：

- StraightPCF：VelocityModule + DistanceModule，强调 straight trajectory 和 distance scalar。https://openaccess.thecvf.com/content/CVPR2024/html/de_Silva_Edirimuni_StraightPCF_Straight_Point_Cloud_Filtering_CVPR_2024_paper.html
- DMRDenoise：不满足于直接 displacement，而是重建 underlying manifold 后重采样。https://luost.me/DMRDenoise/
- MODNet：多尺度/多 offset，再融合，属于自适应 offset 思路。https://arxiv.org/abs/2208.14160
- ASDN：逐点自适应停止迭代，核心是有些点不该继续被降噪。https://ojs.aaai.org/index.php/AAAI/article/view/32331

## 方向风险表

| 方向 | 不是没人做过的部分 | 可能还值得做的缺口 | 创新风险 |
|---|---|---|---|
| 方向 + 距离 + 置信度解耦 | StraightPCF 已经做了方向/速度与距离缩放；很多方法也做 offset 或多 offset | baseline 是直接预测 3D 位移，可尝试把位移拆成单位方向、距离标量、移动置信度，并证明比直接回归稳定 | 中高，容易被认为是 StraightPCF 变体 |
| 噪声感知位移 | 多尺度 offset、adaptive denoising、stop denoising 都有类似“不同点不同处理”的影子 | 如果能定义清楚“噪声程度”监督或自监督信号，而不是只加一个 confidence head，会更扎实 | 中，容易流于模块堆叠 |
| 局部表面投影 | 传统 MLS/局部曲面拟合和 DMRDenoise 都有 surface/manifold 思想 | 用神经网络预测局部 surface 参数，再把点投影过去，从 point-wise displacement 变成 surface-guided projection | 中低，机制更硬，但实现难 |
| 学习何时不该降噪 | ASDN 已经关注 adaptive stop | 可以把“denoise or preserve”作为核心问题，重点处理边缘、薄结构、已干净点被误移动 | 中，需要强失败案例支撑 |
| 误差分解 | 法向/切向、采样分布、结构保持都有人用过相关约束 | 把误差显式拆成表面偏离、切向漂移、采样分布变化三类，并分别评估 | 中，理论表达难度较高 |
| 边缘保护 | feature-preserving denoising 是老问题 | 如果能自动识别边缘失败并控制移动策略，可能有比赛收益 | 高，单独做容易变成常规 trick |

目前最值得先验证的不是“发明模块”，而是先回答：

1. baseline 在哪些样本上输得最明显？
2. 输是因为方向错、距离过大、距离过小、点数覆盖、还是边缘被抹？
3. epoch 越高时，是整体都变好，还是某些类别/结构变差？

## 断点训练

现在训练保存的是完整 checkpoint：

```text
model 参数
optimizer 状态
epoch / next_epoch
Python / NumPy 随机状态
```

`load_ckpt` 和 `resume_ckpt` 的含义不同：

```text
load_ckpt   只加载模型权重，适合推理或 warm start
resume_ckpt 恢复完整训练状态，适合断点续训
```

旧的 `checkpoint_99.pkl` 是历史遗留的权重文件，里面没有 optimizer 状态，所以无法恢复当时 Adam 的动量和二阶矩。它只能这样 warm start：

```yaml
load_ckpt: /root/autodl-tmp/jittor_pcd/experiments/vm/checkpoint_99.pkl

trainer:
  start_epoch: 100
  epochs: 30
```

从这次代码修改后保存出来的新 checkpoint 才能完整 resume，例如：

```yaml
resume_ckpt: /root/autodl-tmp/jittor_pcd/experiments/vm/checkpoint_129.pkl

trainer:
  epochs: 30
```

这时不需要手动写 `start_epoch`，程序会读取 checkpoint 里的 `next_epoch` 自动接着保存，比如从 130 开始。

判断是否值得继续训练的标准：

| 现象 | 判断 |
|---|---|
| epoch20/40/60/80/99 单调提升 | 说明还没明显到头，可以试 120 或 130 |
| CD 基本不动但 P2S 还提升 | 模型还在贴近表面，但点集覆盖变化小 |
| P2S 提升但可视化边缘变圆 | 可能开始过平滑，不能只看总分 |
| validation loss 震荡但本地 score 提升 | 以本地 CD/P2S 为准，loss 只是训练目标 |

## visualize 怎么看

每个 epoch 的结果目录大致是：

```text
result/epoch_99/
  scores/
    summary_cd_p2s_scores.json
    per_sample_cd_p2s_scores.csv
  visualization/
    shapenet__类别__模型/
      noisy_p2s_heatmap.ply
      noisy_p2s_heatmap.json
      denoised_p2s_heatmap.ply
      denoised_p2s_heatmap.json
      p2s_improvement.ply
      p2s_improvement.json
```

推荐顺序：

1. 先看 `summary_cd_p2s_scores.json`
   看 `mean_final_score`、`mean_cd_score`、`mean_p2s_score`，确定整体是否进步。

2. 再看 `per_sample_cd_p2s_scores.csv`
   按 `final_score` 从低到高排序，优先找最差样本。不要只看前几个可视化样本，因为默认可视化只是按列表取前几个，不一定是最失败的。

3. 打开 `p2s_improvement.ply`
   这个最重要。颜色含义：

   ```text
   绿色：denoised 比 noisy 更接近真实 mesh
   红色：denoised 反而比 noisy 更远
   灰色：变化很小
   ```

   如果大片绿色，说明降噪有效；如果边缘、薄结构、孔洞附近出现红色，要记录失败类型。

4. 打开 `denoised_p2s_heatmap.ply`
   蓝/青表示离表面近，黄/红表示离表面远。它适合看“修完后哪里还糟糕”。

5. 对照 `noisy_p2s_heatmap.ply`
   注意 heatmap 默认各自按百分位映射颜色，两个文件颜色不一定绝对同尺度。真正比较改善优先看 `p2s_improvement.ply`。

6. 看 `p2s_improvement.json`
   重点看这些字段：

   ```text
   p2s_score
   improved_point_ratio
   worsened_point_ratio
   p2s_improvement_mean
   p2s_improvement_p95
   movement_mean
   movement_p95
   movement_max
   ```

7. 给失败样本做文字标签

| 标签 | 观察标准 | 可能原因 |
|---|---|---|
| 方向错 | 红色点整体被推向错误表面 | 局部邻域混入其他结构 |
| 距离过大 | 形状变瘦、边缘变圆、点穿过表面 | 位移幅度控制不好 |
| 距离不足 | noisy 仍然散、denoised heatmap 仍然黄/红 | 模型不敢移动或欠拟合 |
| 边缘过平滑 | 棱角/翼/腿部附近红色增加 | 切向漂移或大 patch 平均化 |
| 局部强噪声失败 | 某一块区域红色集中 | 训练噪声分布与测试不匹配 |
| patch 融合痕迹 | 可视化出现块状边界 | patch 覆盖或融合策略问题 |

最终记录格式建议：

| epoch | model_id | final_score | 失败标签 | 观察 | 下一步假设 |
|---|---|---:|---|---|---|
| 99 | shapenet/... | 45.7 | 距离不足 | 大部分绿色但局部仍黄 | 继续训练或增加位移幅度建模 |
