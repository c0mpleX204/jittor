# 方案 03：局部噪声厚度压缩

## 问题观察

你看到的现象可以描述为：

```text
降噪后仍像一层厚壳，而不是薄表面。
```

P2S improvement 全绿说明点大多靠近了表面，但 visual 上仍散，说明平均距离改善不足以保证“厚度坍缩”。

因此可以把问题从：

```text
每个点的位移是否正确
```

改成：

```text
局部邻域的噪声厚度是否被压薄。
```

## 核心创新

为每个点的邻域估计一个局部法向方向 `n_i`，然后衡量邻域点在法向上的厚度：

```math
z_{ij} = n_i^\top(\hat p_j - \mu_i), \quad j \in \mathcal{N}(i)
```

其中：

```math
\mu_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} \hat p_j
```

定义局部厚度：

```math
T_i = Q_{0.9}(\{z_{ij}\}) - Q_{0.1}(\{z_{ij}\})
```

`Q` 是分位数。实现上如果 Jittor 不方便做 quantile，可以先用 top-k 近似。

厚度损失：

```math
L_{thick} = \frac{1}{N}\sum_i T_i^2
```

但只压厚度会把薄结构压坏，所以还要保持切向分布：

```math
C_i^{tan} =
\frac{1}{|\mathcal{N}(i)|}
\sum_{j\in\mathcal{N}(i)}
P_i(\hat p_j-\mu_i)(\hat p_j-\mu_i)^\top P_i
```

其中：

```math
P_i = I - n_i n_i^\top
```

切向分布保持：

```math
L_{tan} =
\frac{1}{N}\sum_i
\|C_i^{tan}(\hat P) - C_i^{tan}(P^{clean})\|_F^2
```

总损失：

```math
L =
L_{\Delta}
\lambda_{thick}L_{thick}
\lambda_{tan}L_{tan}
```

## 为什么这不是普通法向约束

普通法向约束关注：

```text
单个点沿不沿法向移动。
```

这里关注的是：

```text
一个局部邻域整体是否从厚壳压缩成薄表面，同时保持切向分布。
```

这直接对应你现在的 visual 观察：点整体还散。

## 法向从哪里来

三种选择：

1. 用 clean patch 的 PCA 法向作为训练时参考，最简单但推理时没有 clean。
2. 用 denoised patch 的 PCA 法向，训练和推理一致。
3. 网络预测 normal head，再用局部 PCA 做弱约束。

第一版建议：

```text
训练 loss 里用 pc_clean 的局部 PCA 法向计算目标厚度；
预测输出仍然只用模型点，不额外依赖 mesh。
```

## 代码改动提示词

```text
请在 release4/starter_code 中给改进模型加入局部厚度损失，不要改评测脚本。

具体要求：
1. 可以先基于 DistanceCalibratedVelocityModule 或原 VelocityModule 增加一个 loss_thick。
2. 在 get_supervised_loss 中，得到 pred_pc = pc_noisy + pred_delta。
3. 对 pred_pc 和 pc_clean 分别构建 KNN 邻域，K 可以新增配置 thick_knn。
4. 对每个 clean 邻域做 PCA，取最小特征值方向作为 normal。
   如果 Jittor 中特征分解不方便，第一版可用近似：
   - 使用 target displacement 的 normalize 作为 normal 近似；
   - 或者先在 numpy 数据增强阶段预计算 normal，但不要新增复杂脚本。
5. 计算邻域点沿 normal 的投影 z。
6. 用 top-k 近似 q90/q10：
   - q90 取 top k_high 的均值
   - q10 取 bottom k_low 的均值
   thickness = q90 - q10
7. loss_thick = mean((thickness_pred - thickness_clean)^2)
   clean 的 thickness 不一定为 0，因为真实点云有采样厚度和曲面变化。
8. loss = 原 loss + lambda_thick * loss_thick。
9. 在 configs/model/*.yaml 加 thick_knn 和 lambda_thick。

注意：
- 第一版不要过大 lambda_thick，防止把细结构压坏。
- 每轮只在被抽中的 num_train_points 上算邻域厚度，控制显存。
```

## 实验判断

这个方案主要看：

```text
denoised 点云是否从“厚壳”变薄
movement_mean 是否增加但红点不明显增加
薄结构是否被压坏
```

如果 P2S 提升、视觉变薄，但 CD 下降，需要增强切向分布保持。

