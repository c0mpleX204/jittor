# 方案 04：能量选择的多候选修复

## 问题观察

复杂几何区域里，一个 noisy 点的局部邻域可能混入多个表面：

```text
飞机翼与机身交界
椅子腿与座面
薄片边缘
```

直接回归一个平均 displacement 时，模型可能输出一个保守中间值，结果就是：

```text
方向看起来大体对，但移动不够，或者落在两个表面之间。
```

## 核心创新

不让网络只给一个答案，而是给多个候选修复位置：

```math
q_{ik} = p_i + \Delta_{ik}, \quad k=1,\ldots,K
```

再为每个候选预测一个几何能量：

```math
E_{ik} = h(f_i, q_{ik}, \mathcal{N}(i))
```

推理时选择能量最低的候选：

```math
\hat p_i = q_{i,k^\*}, \quad
k^\* = \arg\min_k E_{ik}
```

或者使用 softmin：

```math
w_{ik} =
\frac{\exp(-E_{ik}/\tau)}
\sum_l \exp(-E_{il}/\tau)}
```

```math
\hat p_i = \sum_k w_{ik}q_{ik}
```

## 和 MODNet 的区别

MODNet 关注多 offset、多尺度融合。这里的重点不是多尺度加权，而是：

```text
把候选修复看成多个可能投影，然后用局部几何能量进行选择。
```

这更像“生成候选 + 验证候选”，不是简单平均多个 offset。

## 训练目标

候选位置：

```math
q_{ik}=p_i+\Delta_{ik}
```

找最接近 clean 的 oracle 候选：

```math
k_i^\* =
\arg\min_k \|q_{ik}-p_i^{clean}\|_2^2
```

候选覆盖损失：

```math
L_{oracle} =
\frac{1}{N}\sum_i
\min_k \|q_{ik}-p_i^{clean}\|_2^2
```

能量排序损失：

```math
L_{energy} =
\frac{1}{N}\sum_i
-\log
\frac{\exp(-E_{ik_i^\*})}
\sum_k \exp(-E_{ik})}
```

防止多个候选塌缩成同一个：

```math
L_{div} =
\frac{1}{N}\sum_i
\frac{1}{K(K-1)}
\sum_{k\ne l}
\exp\left(
-\frac{\|q_{ik}-q_{il}\|_2^2}{\sigma^2}
\right)
```

最终输出损失：

```math
L_{final} =
\frac{1}{N}\sum_i
\|\hat p_i-p_i^{clean}\|_2^2
```

总损失：

```math
L =
L_{final}
\lambda_o L_{oracle}
\lambda_e L_{energy}
\lambda_d L_{div}
```

## 几何能量可以包含什么

第一版不要太复杂：

```text
E_ik = MLP([f_i, Delta_ik, ||Delta_ik||, local_density_change])
```

更几何一点：

```math
E_{ik}
= a \cdot \mathrm{LocalResidual}(q_{ik})
+ b \cdot \mathrm{MovementPenalty}(\Delta_{ik})
+ c \cdot \mathrm{NeighborConsistency}(q_{ik})
```

训练时用 `L_energy` 监督它学会哪个候选更接近 clean。

## 代码改动提示词

```text
请在 release4/starter_code 中实现一个 EnergySelectedCandidateModule。

要求：
1. 新建 src/model/energy_candidates.py。
2. Encoder 复用 FeatureExtraction。
3. candidate_head 输出 K*3 维，reshape 为 (B,N,K,3)，表示 K 个 Delta。
4. energy_head 输出 K 维，表示每个候选的 energy。
5. q = pc_noisy.unsqueeze(2) + delta_candidates。
6. 训练时计算每个候选到 pc_clean 的 squared distance，得到 oracle index。
7. 使用 softmin(-energy/tau) 得到权重，输出 pred_pc = sum_k w_k q_k。
8. loss 包含：
   - loss_final: pred_pc vs pc_clean
   - loss_oracle: min candidate distance
   - loss_energy: energy 对 oracle candidate 的交叉熵
   - loss_div: 防止候选塌缩
9. 推理时默认使用 softmin 输出；也保留 hard argmin 开关。
10. 在 src/model/parse.py 注册新模型，新增 configs/model/energy_candidates.yaml。

注意：
- K 先设 3，不要一开始设太大。
- 不要改数据、评测、结果整理脚本。
- 必须保证输出点数等于输入点数。
```

## 实验判断

观察：

```text
是否减少“落在两层表面之间”的散点
是否在复杂结构处红点减少
是否比单 displacement 更敢移动
```

如果训练不稳定，先降低 `lambda_div`，并把输出改成 softmin 而不是 hard argmin。

