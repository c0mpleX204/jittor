# 方案 02：神经局部曲面投影

## 问题观察

直接位移回归把问题写成：

```text
noisy point -> displacement -> denoised point
```

但点云降噪的几何本质更像：

```text
把点投影回某个真实表面。
```

如果模型只输出 3D offset，它可能只学到“靠近一点”，却没有显式建模“表面在哪里”。这和你看到的“点整体变近但仍然散”是吻合的。

## 核心创新

不直接预测 `Delta_i`，而是让网络为每个点或局部 patch 预测一个局部隐式曲面：

```math
\phi_i(x) = 0
```

然后把 noisy point 投影到这个曲面上：

```math
\hat p_i =
p_i -
\frac{\phi_i(p_i)}
{\|\nabla \phi_i(p_i)\|_2^2 + \epsilon}
\nabla \phi_i(p_i)
```

这类似一次 Newton projection。这样模型输出的是“局部表面”，不是“一个死的位移向量”。

## 可实现的局部曲面形式

为了不把实现做爆，可以先用低阶曲面：

```math
\phi_i(x) =
n_i^\top (x - a_i)
+ (x-a_i)^\top A_i (x-a_i)
+ b_i
```

其中：

```text
n_i：局部法向方向，网络预测 3 维并归一化
a_i：局部锚点，可以直接用当前 noisy point 或预测小偏移后的点
A_i：低秩/对角曲率项，先用 3 维对角向量表示
b_i：偏置
```

第一版可以更简单：

```math
\phi_i(x) = n_i^\top(x-a_i)
```

也就是先做局部平面投影：

```math
\hat p_i = p_i - n_i^\top(p_i-a_i)n_i
```

关键不是“平面投影”本身，而是 `n_i` 和 `a_i` 来自神经特征，并且训练目标是投影后的点贴近 clean。

## 训练目标

投影点：

```math
\hat p_i = \Pi_{\phi_i}(p_i)
```

主损失：

```math
L_{proj} = \frac{1}{N}\sum_i \|\hat p_i - p_i^{clean}\|_2^2
```

clean 点应在预测曲面上：

```math
L_{surface} =
\frac{1}{N}\sum_i |\phi_i(p_i^{clean})|
```

曲率正则，防止曲面乱弯：

```math
L_{curv} =
\frac{1}{N}\sum_i \|A_i\|_F^2
```

总损失：

```math
L = L_{proj} + \lambda_s L_{surface} + \lambda_c L_{curv}
```

## 和已有方向的区别

DMRDenoise 一类方法强调重建 manifold 后重采样。这里不重采样，不改变点数，而是：

```text
每个输入点保留对应关系，通过神经预测的局部曲面投影回表面。
```

这更适合比赛，因为提交要求输出点数和输入一致。

## 代码改动提示词

```text
请在 release4/starter_code 中实现一个 SurfaceProjectionModule，不要改动原 VelocityModule。

具体要求：
1. 新建 src/model/surface_projection.py。
2. 复用 FeatureExtraction 作为 encoder。
3. decoder 输出局部平面参数：
   - normal_raw: 3 维
   - anchor_delta: 3 维
   可选：
   - curvature_diag: 3 维，第一版可以先不启用。
4. 对 normal_raw 做 normalize 得到 normal。
5. anchor = pc_noisy + anchor_delta。
6. 平面投影：
   signed_dist = sum((pc_noisy - anchor) * normal, dim=-1, keepdims=True)
   pc_proj = pc_noisy - signed_dist * normal
7. get_supervised_loss 中使用：
   loss_proj = mean(||pc_proj - pc_clean||^2)
   loss_surface = mean(abs(sum((pc_clean-anchor)*normal)))
   loss_anchor = mean(||anchor - pc_clean||^2)，权重较小。
8. predict_step 中输出 pc_proj。
9. 在 src/model/parse.py 注册 SurfaceProjectionModule。
10. 新增 configs/model/surface_projection.yaml。

注意：
- 第一版只做局部平面投影，不上二次曲面，先验证机制。
- 不要加新训练脚本；复用 run.py 和 task yaml。
- 输出仍必须保持和输入点数完全一致。
```

## 实验判断

适合观察：

```text
denoised_p2s_heatmap 是否从蓝绿变得更蓝
点云壳层厚度是否明显变薄
边缘是否被过度平面化
```

如果 P2S 提升但 CD 下降，说明投影贴表面但采样分布/形状覆盖可能变差，需要加入分布保持项。

