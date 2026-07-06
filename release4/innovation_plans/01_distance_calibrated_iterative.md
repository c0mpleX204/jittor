# 方案 01：距离校准的方向-距离解耦迭代降噪

## 问题观察

当前可视化现象是：

```text
p2s_improvement 基本全绿，但 denoised 点云仍然散。
```

这说明 baseline 不是完全“方向错”，而是更像：

```text
方向大体正确，但移动距离偏保守，点没有真正贴回表面。
```

baseline 当前形式：

```text
f_i = Encoder(P)_i
Delta_i = MLP(f_i)
hat p_i = p_i + Delta_i
```

这里 `Delta_i` 同时承担“往哪里走”和“走多远”。当局部 patch 有歧义时，MSE 容易学到保守平均位移。

## 核心创新

把位移拆成三个量：

```text
方向 direction：u_i
距离 distance：r_i
可靠度 confidence：c_i
```

最终位移：

```math
\Delta_i = c_i \cdot r_i \cdot \frac{u_i}{\|u_i\|_2 + \epsilon}
```

降噪点：

```math
\hat p_i = p_i + \Delta_i
```

它和 StraightPCF 的区别应该写清楚：

```text
StraightPCF 强调 straight trajectory 和 distance scalar。
这里的重点不是复刻它，而是针对 baseline 可视化中的“全绿但仍散”，做 distance calibration：
显式监督距离估计，增加欠降噪惩罚，并允许粗到细迭代修正残余误差。
```

## 训练目标

已知训练时有 clean 点：

```math
d_i^\* = p_i^{clean} - p_i^{noisy}
```

目标距离：

```math
r_i^\* = \|d_i^\*\|_2
```

目标方向：

```math
u_i^\* = \frac{d_i^\*}{r_i^\* + \epsilon}
```

预测：

```math
\tilde u_i = g_u(f_i), \quad
r_i = \mathrm{softplus}(g_r(f_i)), \quad
c_i = \sigma(g_c(f_i))
```

组合位移：

```math
\Delta_i = c_i r_i \frac{\tilde u_i}{\|\tilde u_i\|_2+\epsilon}
```

基础损失：

```math
L_{\Delta} = \frac{1}{N}\sum_i \|\Delta_i - d_i^\*\|_2^2
```

方向损失：

```math
L_{dir} = \frac{1}{N}\sum_i \left(1 - 
\left\langle 
\frac{\tilde u_i}{\|\tilde u_i\|_2+\epsilon}, u_i^\*
\right\rangle \right)
```

距离校准损失：

```math
L_{dist} = \frac{1}{N}\sum_i |r_i - r_i^\*|
```

欠降噪惩罚，专门对应“全绿但仍散”：

```math
L_{under} =
\frac{1}{N}\sum_i
\max(0, \alpha r_i^\* - c_i r_i)^2
```

总损失：

```math
L =
L_{\Delta}
\lambda_{dir}L_{dir}
\lambda_{dist}L_{dist}
\lambda_{under}L_{under}
```

其中 `alpha` 可以先设为 `0.8`，不是为了调参，而是表达“如果真实需要移动 r，模型不能长期只移动很小一部分”。

## 粗到细迭代

推理时做 `T=2` 或 `T=3` 步：

```math
p_i^{t+1}
= p_i^t + \eta_t c_i^t r_i^t
\frac{u_i^t}{\|u_i^t\|_2+\epsilon}
```

其中：

```text
t=0：粗修，允许较大移动
t=1/2：细修，重新提特征后小步靠近表面
```

这和训练学习率无关，它是点位置空间里的自适应修复步长。

## 代码改动提示词

把下面这段给代码助手：

```text
请在 release4/starter_code 中实现一个新模型 DistanceCalibratedVelocityModule，不要破坏原来的 VelocityModule。

具体要求：
1. 新建 src/model/dcvm.py，可以复用 FeatureExtraction 和 Decoder。
2. Encoder 仍用 FeatureExtraction。
3. 将原来 out_dim=3 的 decoder 拆成三个 head：
   - direction_head: 输出 3 维 raw direction。
   - distance_head: 输出 1 维，用 softplus 保证非负。
   - confidence_head: 输出 1 维，用 sigmoid 限制到 [0,1]。
4. 在 get_supervised_loss 中，target = pc_clean - pc_noisy。
   计算 target_dist = norm(target) 和 target_dir = target / (target_dist + eps)。
5. 组合 pred_delta = confidence * distance * normalize(direction)。
6. 返回 loss 字典，至少包括：
   loss_delta, loss_dir, loss_dist, loss_under, loss。
   先让 loss 作为加权总和，兼容当前 system 的 loss 配置。
7. 在 denoise_langevin_dynamics 中支持 num_steps=2 或 3：
   每一步重新 encoder 当前 pcl_next，再预测 direction/distance/confidence 并更新点。
8. 在 src/model/parse.py 注册 DistanceCalibratedVelocityModule。
9. 新增 configs/model/dcvm.yaml，保留 frame_knn、num_train_points、feat_embedding_dim、decoder_hidden_dim、dsm_sigma，并增加：
   lambda_dir, lambda_dist, lambda_under, under_alpha, denoise_steps。
10. 新增 configs/task/train_dcvm_autodl_tmp.yaml 时只改 components.model: dcvm，其他尽量复用现有配置。

注意：
- 不要新增无关脚本。
- 不要改动数据处理和评测脚本。
- baseline 的 vm.py 保持可复现。
```

## 实验判断

第一轮只看：

```text
epoch99 baseline vs dcvm 同训练轮数
```

重点比较：

```text
mean_final_score
mean_cd_score
mean_p2s_score
p2s_improvement 是否仍然全绿但散
movement_mean 是否明显增大但不过冲
```

如果：

```text
movement_mean 增大
denoised_p2s_heatmap 更蓝
红色没有明显增加
```

说明这个方向有效。

