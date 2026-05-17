# 点云降噪 Baseline 与 StraightPCF 方法报告

本文梳理 `release4/starter_code` 中官方 Jittor baseline 的训练、推理、测试流程，并结合 StraightPCF: Straight Point Cloud Filtering 的论文思想，分析 baseline 与 StraightPCF 的关系，最后给出可用于比赛提分和论文创新的方向。

参考资料：
- StraightPCF 论文：CVPR 2024, *StraightPCF: Straight Point Cloud Filtering*, https://openaccess.thecvf.com/content/CVPR2024/html/de_Silva_Edirimuni_StraightPCF_Straight_Point_Cloud_Filtering_CVPR_2024_paper.html
- StraightPCF 官方代码：https://github.com/ddsediri/StraightPCF
- 本地 baseline 代码：`release4/starter_code`

## 1. Baseline 总体结构

baseline 以 `run.py` 作为统一入口，通过 YAML 配置组装四类组件：

- data：数据路径、loader、batch size、worker 数量。
- transform：训练/验证/推理时的数据采样、归一化、加噪、patch 构造。
- model：`VelocityModule`，负责预测逐点位移向量。
- system：训练、验证、推理循环，以及 checkpoint 保存和预测结果写出。

核心文件关系如下：

```text
run.py
  -> configs/task/train_vm.yaml 或 predict_vm.yaml
  -> configs/data/train.yaml / predict.yaml
  -> configs/transform/vm.yaml
  -> configs/model/vm.yaml
  -> src/data/*
  -> src/model/vm.py, src/model/feature.py
  -> src/system/spec.py, src/system/vm.py
```

从方法上看，这份 baseline 对应 StraightPCF 中较基础的单个 VelocityModule 思路：用动态图卷积提取 patch 局部几何特征，再用 MLP 解码每个点的三维位移。它没有完整实现 StraightPCF 论文中的 Coupled VelocityModule stack 和 DistanceModule。

## 2. 训练流程

训练入口为：

```bash
python run.py --task configs/task/train_vm.yaml
```

### 2.1 配置加载

`train_vm.yaml` 指定：

- mode：`train`
- data：`configs/data/train.yaml`
- transform：`configs/transform/vm.yaml`
- model：`configs/model/vm.yaml`
- optimizer：Adam, lr = `1e-5`
- epochs：100
- loss 权重：`loss: 1.0`

`run.py` 会根据配置创建 dataset、transform、model、system，然后调用 `system.train()`。

### 2.2 数据读取

训练数据来自：

```text
dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

`ObjLazyAsset` 使用 `trimesh` 读取 `.obj`，得到：

- `vertices`
- `faces`
- `path`
- `cls`

`configs/data/train.yaml` 中 `use_prob: True`，`num_files: 10000`，表示每个 epoch 采样约 10000 个训练样本索引。`batch_size=16`，每个 batch 中每个 shape 会生成一个训练 patch。

### 2.3 数据增强与 patch 构造

训练 transform 顺序为：

1. `sample`：从 mesh 表面采样 `32768` 个点，并额外包含 `1024` 个顶点样本。
2. `normalize_pc`：以 bbox 中心居中，并缩放到单位球。
3. `add_noise`：随机采样噪声标准差，范围 `[0.005, 0.020]`，代码中使用 Laplace 噪声。
4. `linear`：配置上包含随机旋转与缩放。
5. `patch`：构造局部 patch。

patch 构造逻辑在 `AugmentPatch`：

- 在 noisy 点云中随机选 `num_patches=1` 个 seed。
- 用 `cKDTree` 找到 noisy 点云中离 seed 最近的 `patch_size=1000` 个点。
- noisy patch 记为 `pat_A`，对应 clean patch 记为 `pat_B`。
- 采样随机时间 `t in (0, 1)`，构造中间状态：

```text
pat_t = t * pat_B + (1 - t) * pat_A
```

- 以同一条直线上的 seed 中间状态为中心，对 `pat_A / pat_B / pat_t` 做局部中心化。
- 输出到 `asset.meta`：

```text
pc_noisy = pat_A
pc_clean = pat_B
pc_mix   = pat_t
```

这里的思想已经接近 StraightPCF：把 noisy 和 clean 之间的路径看作一条线段，中间状态用于训练网络学习“沿直线推回表面”的速度。

### 2.4 模型结构

模型为 `VelocityModule`：

```text
input patch: pc_mix, shape (B, N, 3)
encoder: FeatureExtraction
decoder: MLP Decoder
output: pred_dir, shape (B, num_train_points, 3)
```

`FeatureExtraction` 使用三层 Dynamic EdgeConv：

- 第 1 层：输入 3 维坐标，输出 `embedding_dim / 8`
- 第 2 层：动态图重新建图，输出 `embedding_dim / 4`
- 第 3 层：拼接前两层特征，再输出 `embedding_dim`

EdgeConv 的 message 形式为：

```text
[x_i, x_j - x_i] -> MLP -> scatter mean -> residual linear(x_i)
```

这使模型能感知局部相对几何，而不是只看孤立点坐标。

Decoder 是三层 MLP：

```text
Linear(F,F) + BN + ReLU + Dropout
Linear(F,H) + BN + ReLU + Dropout
Linear(H,3)
```

输出为每个点的三维位移方向。

### 2.5 损失函数

`VelocityModule.get_supervised_loss()` 中：

1. 对 patch 中的点随机抽 `num_train_points=128` 个监督点。
2. 用 `pc_mix` 做特征提取。
3. 对抽样点预测位移 `pred_dir`。
4. 目标为：

```text
target = pc_clean - pc_noisy
loss = mean(sum((pred_dir - target)^2) / dsm_sigma)
```

从理论上解释，这等价于让网络在中间状态 `pc_mix` 上预测从高噪声端 `pc_noisy` 到 clean 端 `pc_clean` 的常量速度。这个设计对应 StraightPCF 的单 VM constant flow 思想。

### 2.6 训练循环与 checkpoint

`DummySystem.train()` 每个 epoch 执行：

1. `model.train()`
2. 遍历 train dataloader
3. forward 得到 loss
4. Adam 反向传播和更新
5. validation 阶段只计算同一监督 loss
6. 每个 epoch 保存一次 checkpoint：

```text
experiments/vm/checkpoint_<epoch>.pkl
```

注意：baseline 训练阶段不直接计算 CD/P2S，也没有内置 early stopping 或 best checkpoint 管理。因此如果每个 epoch 都拿去完整推理，会非常耗时。

## 3. 推理流程

推理入口为：

```bash
python run.py --task configs/task/predict_vm.yaml
```

### 3.1 加载 checkpoint 和测试数据

`predict_vm.yaml` 默认加载：

```text
experiments/vm/checkpoint_99.pkl
```

测试数据来自：

```text
dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

`NpyLazyAsset` 读取 `.npy`，得到 `sampled_vertices_noisy`，shape 为 `(N, 3)`。

### 3.2 Patch-based denoise

核心函数为 `patch_based_denoise()`。输入是一整个 noisy 点云 `(N, 3)`，由于 N 通常约 50000，不能直接整云输入网络，于是采用分块策略：

1. 计算 patch 数量：

```text
num_patches = int(seed_k * N / patch_size)
```

默认 `patch_size=1000`，`seed_k=6`，即每个点平均被多个 patch 覆盖。

2. 用 farthest point sampling 选 seed 点。
3. 对每个 seed，在整云上找 KNN，得到局部 patch。
4. 每个 patch 以 seed 为中心做局部中心化。
5. 调用 `model.denoise_langevin_dynamics()` 对 patch 降噪。
6. 每个 patch 内部默认执行 4 次小步迭代：

```text
pcl_next = pcl_next + (1 / 4) * pred_dir
```

7. 将 patch 加回 seed 中心。
8. 对每个原始点，从覆盖它的 patch 中选权重最大的那个 patch 的预测结果，拼回整云。

这里的推理本质是 Euler ODE 更新：网络预测一个速度/位移方向，点沿这个方向逐步移动。baseline 没有显式预测“该走多远”，因此可能出现欠移动或过移动。

### 3.3 结果写出

`VMWriter` 将预测结果保存为 `.npy`。比赛提交时需要保证最终结构为：

```text
result.zip
  shapenet/<synset_id>/<model_id>/denoised.npy
```

如果使用官方 baseline 默认 writer，需要确认输出目录和文件名是否已经在你的云端脚本中整理为提交格式。

## 4. 测试与评分流程

本地评测脚本主要做两类指标：

### 4.1 Chamfer Distance

对 predicted 点云和 clean 点云计算双向最近邻平方距离：

```text
CD(pred, gt) = mean_x min_y ||x-y||^2 + mean_y min_x ||y-x||^2
```

评测前以 GT 点云为参考做单位球归一化，同样的中心和尺度作用到 pred/noisy。

### 4.2 Point-to-Surface Distance

P2S 计算 pred 点到 mesh 表面的最近距离平方均值：

```text
P2S(pred, mesh) = mean_x min_y_on_mesh ||x-y||^2
```

代码中优先用 `point-cloud-utils` 做点到三角面距离。

### 4.3 百分制映射

评分以 noisy input 作为零分基线：

```text
cd_score  = clamp(100 * (1 - CD_pred  / CD_noisy),  0, 100)
p2s_score = clamp(100 * (1 - P2S_pred / P2S_noisy), 0, 100)
final = 0.5 * mean(cd_score) + 0.5 * mean(p2s_score)
```

所以模型的目标不是绝对 CD/P2S 越小越好这么简单，而是相对 noisy input 的改善比例越大越好。

## 5. StraightPCF 理论梳理

StraightPCF 的核心观点是：很多点云降噪方法把点沿随机或弯曲轨迹慢慢推回表面，因此需要很多迭代，还容易产生分布不均、孔洞、聚簇或过平滑。StraightPCF 希望把降噪建模为从高噪声分布到干净表面的最优传输，让点尽量沿直线路径移动。

### 5.1 直线路径建模

设 clean patch 为：

```text
X_1 = Y
```

高噪声 patch 为：

```text
X_0 = Y + sigma_H * noise
```

中间状态定义为线性插值：

```text
X_t = (1 - t) X_0 + t X_1,  t in [0, 1]
```

这意味着从 `X_0` 到 `X_1` 的理想轨迹就是直线。

### 5.2 ODE 与 VelocityModule

StraightPCF 将点云运动写成 ODE：

```text
dX_t = v(X_t) dt
```

如果路径是线性的，则理想速度是常量：

```text
v*(X_t) = X_1 - X_0
```

于是训练单个 VelocityModule 的目标为：

```text
L_A = E_t || v_theta(X_t) - (X_1 - X_0) ||^2
```

推理时用 Euler 更新：

```text
X_{next} = X_current + (1 / N) * v_theta(X_current)
```

baseline 的 `VelocityModule` 基本就是这个单 VM 思想的 Jittor 实现。

### 5.3 Coupled VelocityModule

单个 VM 的预测轨迹仍可能不够直，误差会逐步积累。StraightPCF 因此把一条轨迹分成 K 段，用 K 个 VelocityModule 级联。论文中经验上 K=2 在精度和效率之间较好。

它的训练目标包含两部分：

- 每个 VM 都要预测接近 `X_1 - X_0` 的常量速度。
- 更新后的中间状态要贴近对应的线性插值状态。

简化理解：

```text
L_B = direction_loss + lambda_1 * consistency_loss
```

方向项保证“速度方向正确”，一致性项保证“轨迹真的更直”。

### 5.4 DistanceModule

仅有常量速度会遇到一个问题：推理时不知道输入噪声到底对应线段上的哪个时间点。走少了会残留噪声，走多了会越过表面。

StraightPCF 引入 DistanceModule 预测一个 patch 级别的距离标量：

```text
d_phi(X_t) in [0, 1]
```

它表示当前 patch 离 clean surface 的相对距离。推理时用该标量缩放 VelocityModule 输出：

```text
X_next = X_current + (d_phi / T) * v_theta(X_current)
```

这就是 StraightPCF 中“方向”和“距离”解耦的关键：VelocityModule 负责往哪里走，DistanceModule 负责走多远。

### 5.5 StraightPCF 的实验结论

论文报告 StraightPCF 约 530K 参数，仅为 IterativePFN 的 17%。它在 PUNet、PCNet、Kinect、RueMadame 等数据上表现优良，特别强调：

- 更少迭代即可收敛。
- 点分布更接近 clean 分布，不容易出现孔洞和聚簇。
- 不依赖额外正则化或后处理。
- 但在低密度、高稀疏点云上仍有不足。

## 6. Baseline 与 StraightPCF 的关系

可以把当前 baseline 理解为：

```text
StraightPCF single VelocityModule 的简化版 / starter 版
```

相同点：

- 都使用 patch-based 输入。
- 都使用 Dynamic EdgeConv 提取局部特征。
- 都让网络预测逐点三维位移/速度。
- 都构造 noisy-clean 之间的中间状态。
- 推理都用 patch 分块和迭代更新。

缺失点：

- 没有 Coupled VelocityModule。
- 没有 DistanceModule。
- 没有显式轨迹一致性 loss。
- 没有根据噪声强度自适应控制步长。
- 没有点分布均匀性或几何结构保持项。

这也是后续创新的入口：baseline 只是一个较朴素的“方向预测器”，还有很多空间可以在“距离估计、几何约束、噪声鲁棒性、patch 融合、效率”上做文章。

## 7. 可创新方向

### 7.1 方向一：实现并改造 DistanceModule

最直接的提升是加入距离估计，但不要只照搬 StraightPCF。可以改成：

- patch 级距离标量：简单稳定，接近 StraightPCF。
- point-wise 距离：每个点预测自己的移动幅度，适合处理同一 patch 内噪声强度不均。
- uncertainty-aware 距离：同时预测均值和不确定性，用不确定性控制移动幅度。

论文创新点可以写成：将 StraightPCF 的 patch-level distance scalar 扩展为 point-wise adaptive denoising magnitude，以处理局部强噪声和非均匀噪声。

### 7.2 方向二：方向与幅度解耦

当前 baseline 直接输出三维位移 `Delta`。可以改成：

```text
Delta_i = alpha_i * normalize(v_i)
```

其中：

- `v_i` 学方向。
- `alpha_i` 学步长/距离。

这样可以降低网络同时学习方向和长度的难度，也更容易解释。进一步可以让 `alpha_i` 受噪声估计、局部曲率、密度影响。

### 7.3 方向三：几何感知的边缘保持

点云降噪最容易牺牲尖锐边、棱角和薄结构。可以加入局部 PCA 几何特征：

- 局部特征值比例。
- curvature proxy。
- surface variation。
- density。
- normal consistency。

再设计 geometry confidence：

```text
Delta_i = c_i * alpha_i * v_i
```

其中 `c_i` 控制边缘、薄结构处的移动幅度，避免过度平滑。

### 7.4 方向四：非高斯与复杂噪声鲁棒训练

StraightPCF 论文主要围绕 Gaussian noise。赛题数据和真实传感器噪声可能更复杂。可以设计 mixed noise training：

- Gaussian
- Laplace
- anisotropic noise
- local strong noise
- outliers
- scanline/banding noise

这类方向实验成本低，容易形成有效消融：baseline、Gaussian-only、mixed-noise、mixed-noise + adaptive distance。

### 7.5 方向五：鲁棒损失函数

MSE 对离群点敏感。可尝试：

- Huber loss
- Charbonnier loss
- Tukey biweight
- 分位数 loss
- uncertainty weighted regression

如果和 outlier/noise confidence 结合，可以形成“鲁棒位移回归”的论文故事。

### 7.6 方向六：表面约束与 P2S 导向

赛题最终有 P2S 指标，训练却只做点对点位移回归。可以考虑加入表面相关约束：

- 从 mesh 重新采样更多 clean surface points 做局部 CD。
- 近似点到三角面距离作为辅助监督。
- normal consistency，尤其用于边缘保持。
- projection-to-surface oracle 用作上界分析。

注意比赛规则不允许外部数据，但训练集 mesh 是提供数据，可以使用。

### 7.7 方向七：Patch 融合策略改进

当前 patch 推理后，每个点基本从某个“最佳 patch”取结果。可以改成：

- 多 patch 加权平均。
- 基于距离、置信度、预测不确定性联合加权。
- 对 patch 边界降低权重。
- 对重复预测结果做 robust aggregation，例如 median 或 trimmed mean。

这类改动不一定需要重新训练，适合作为快速推理提分实验。

### 7.8 方向八：自适应推理步数

固定 4 步或固定 niters 不适合不同噪声强度。可以加入：

- 根据预测距离决定步数。
- 根据连续两次位移范数小于阈值提前停止。
- 根据局部残差/置信度进行局部迭代。

这能同时服务精度和效率，在 B 榜可能有价值。

### 7.9 方向九：多尺度局部特征

单一 KNN 尺度可能对薄结构、局部强噪声不稳。可加入：

- 多 K 值 EdgeConv。
- 小尺度保边，大尺度估计表面趋势。
- 局部-全局融合。
- density-aware KNN 或 radius graph。

这是更偏结构创新的方向，训练成本较高，但论文空间更大。

### 7.10 方向十：轻量化与实验工程

如果目标包含 B 榜效率或论文实用性，可以做：

- 预采样/预生成 patch 缓存。
- KNN 缓存或近似 KNN。
- 少 patch overlap 的快速推理。
- teacher-student distillation。
- 小模型复现大模型效果。

这类方向适合作为附加贡献，不一定作为主创新。

## 8. 推荐研究路线

建议不要一开始就大改网络，而是按可控路径推进：

1. 先复现实验：固定 baseline、mini-val、完整验证脚本。
2. 做诊断：找出 baseline 在哪些噪声、哪些几何结构上失败。
3. 第一阶段创新：mixed noise + robust loss，成本低，快速验证鲁棒性。
4. 第二阶段创新：point-wise distance/confidence，解决“走多远”的问题。
5. 第三阶段创新：geometry-aware confidence，解决边缘和薄结构过平滑。
6. 最后再做效率优化和完整消融。

一个较完整的论文主线可以是：

```text
Straight flow 降噪虽然高效，但原始 patch-level 或固定步长策略难以处理非均匀噪声和复杂几何。
因此，我们提出一种 geometry-aware point-wise adaptive straight flow 方法：
用 VelocityModule 学习方向，用 point-wise distance/confidence 学习幅度，
再用局部几何置信度保护边缘与薄结构。
```

这样既继承 StraightPCF 的理论优势，又不是简单复现。

## 9. 实验设计建议

为了避免每次都完整训练和推理，可以设置三层实验：

- debug：少量模型，5 到 10 epoch，只验证代码和 loss 是否合理。
- mini-val：固定 20 到 50 个 shape，训练 20 到 40 epoch，用于方法筛选。
- full：只对最有希望的版本跑 100 epoch 和完整测试。

消融表建议：

```text
Baseline
Baseline + mixed noise
Baseline + robust loss
Baseline + point-wise distance
Baseline + point-wise distance + geometry confidence
Full model
```

指标建议除了全局 CD/P2S，还记录：

- 按噪声类型分组 CD/P2S。
- edge/corner/thin 区域 P2S。
- outlier 区域改善率。
- 推理时间和显存。
- 可视化 close-up 与 P2S heatmap。

最终论文不需要证明你试过所有排列组合，而要证明每个模块都对应一个明确失败模式，并在消融中稳定改善该失败模式。
