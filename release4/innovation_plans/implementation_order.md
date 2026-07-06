# 建议实施顺序

不要四个方向一起上。这样会无法判断哪一部分有效。

## 第一阶段：确认 baseline 失败模式

从 epoch99 或 epoch199 结果中选 10 个样本，记录：

```text
是否全绿但散
是否局部红点集中
是否边缘变圆
是否薄结构变厚
movement_mean / movement_p95
improved_point_ratio / worsened_point_ratio
```

如果大多数都是“全绿但散”，优先做方案 01。

## 第二阶段：最小模型改动

优先实现：

```text
方案 01：Distance-Calibrated Iterative Denoising
```

第一版只做：

```text
direction head
distance head
confidence head
loss_delta
loss_dir
loss_dist
```

暂时不加：

```text
复杂噪声增强
厚度损失
多候选
边缘保护
```

这样能干净地回答：

```text
直接位移回归是否因为距离估计不足而受限？
```

## 第三阶段：如果方案 01 有效

再加：

```text
粗到细迭代 T=2
L_under 欠降噪惩罚
```

观察：

```text
P2S 是否提高
红点是否增加
CD 是否受损
```

## 第四阶段：如果仍然散

考虑方案 03：

```text
局部厚度压缩
```

因为这直接针对“厚壳散点”。

## 第五阶段：如果复杂结构错

如果红点集中在飞机翼根、椅子腿、薄片边缘，再考虑：

```text
方案 04：能量选择多候选
```

## 第六阶段：论文主线可能写法

如果方案 01 + 03 有效，可以形成一条比较自洽的主线：

```text
现象：
直接 displacement regression 在评测上能改善 P2S，但可视化显示仍存在残余噪声厚壳。

原因：
MSE 位移回归倾向于学习保守平均位移，方向大体正确但距离校准不足。

方法：
提出距离校准的方向-距离解耦降噪网络，并引入局部厚度压缩约束，使点云从“靠近表面”进一步变成“贴合薄表面”。
```

这个故事比“加模块提升性能”更扎实。

