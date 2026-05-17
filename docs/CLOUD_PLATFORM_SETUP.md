# 云算力平台环境安装指南

本文件只用于云算力平台的 Linux 环境安装。不要在本地 Windows 环境执行这些安装命令；本地只维护代码、脚本和实验文档。

## 0. 适用前提

- 系统：Ubuntu 20.04 / 22.04 或云平台提供的兼容 Linux 镜像。
- GPU：NVIDIA GPU，平台已安装驱动。
- 环境管理：优先使用 Conda / Miniconda / Mambaforge。
- 项目目录示例：`/workspace/g2`。实际路径按云平台挂载目录调整。

首次进入云平台后先检查基础环境：

```bash
nvidia-smi
which python || true
python --version || true
which conda || true
gcc --version || true
g++ --version || true
nvcc --version || true
df -h
free -h
```

## 1. 比赛运行环境：Jittor

用于训练官方 baseline、跑 StraightPCF-style baseline、推理和最终提交。

```bash
# 如果云平台还没有 conda，请先按平台说明安装 Miniconda 或 Mambaforge。

conda create -n jittor-pcd python=3.9 -y
conda activate jittor-pcd

# Jittor 编译对 gcc/g++ 版本比较敏感，先使用 conda-forge 的 gcc 10 / g++ 10。
conda install -c conda-forge gcc=10 gxx=10 libgomp cmake ninja make -y

python -m pip install --upgrade pip setuptools wheel

cd /workspace/g2/jittor/release4/starter_code
python -m pip install -r requirements.txt

# 指标与实验脚本可能用到的补充依赖。
python -m pip install \
  pandas matplotlib tqdm scikit-learn \
  point-cloud-utils
```

如果云平台有系统 CUDA Toolkit，并且 `nvcc --version` 能正常输出，可以显式告诉 Jittor 使用它：

```bash
conda activate jittor-pcd
export nvcc_path=/usr/local/cuda/bin/nvcc
export cc_path=$(which g++)
```

建议把上面两行写入环境激活脚本，避免每次登录后忘记设置：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/jittor_env.sh" <<'EOF'
export nvcc_path=/usr/local/cuda/bin/nvcc
export cc_path=$(which g++)
EOF
```

验证 Jittor 是否可用：

```bash
conda activate jittor-pcd

python - <<'PY'
import jittor as jt
print("Jittor:", jt.__version__)
jt.flags.use_cuda = 1
x = jt.random((4, 4))
y = jt.random((4, 4))
z = x @ y
print("CUDA matmul OK:", z.shape)
PY

python -m jittor.test.test_example
python -m jittor.test.test_cuda
```

如果 `test_cuda` 失败，先不要改代码，按顺序排查：

```bash
nvidia-smi
nvcc --version
echo "$nvcc_path"
echo "$cc_path"
python - <<'PY'
import jittor as jt
print(jt.__version__)
print("has_cuda:", jt.has_cuda)
PY
```

常见处理：

- `gcc/g++ too new`：确认当前环境里 `which g++` 指向 conda 环境，而不是系统的 g++ 13/14。
- `nvcc not found`：确认云平台镜像是否带 CUDA Toolkit；有些平台只有驱动，没有 `nvcc`，需要换 CUDA 开发镜像。
- 第一次运行很慢：Jittor 会即时编译并写缓存，这是正常现象。
- 磁盘不够：清理旧实验输出、Jittor cache，或把数据和日志放到云平台的大容量数据盘。

## 2. 分析环境：Python + 点云工具

用于生成噪声数据、统计指标、导出可视化文件。这个环境可以建在云平台，也可以延后到需要做图时再建。

```bash
conda create -n pcd-analysis python=3.10 -y
conda activate pcd-analysis

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  numpy scipy pandas matplotlib tqdm scikit-learn \
  trimesh point-cloud-utils open3d
```

验证分析工具：

```bash
conda activate pcd-analysis

python - <<'PY'
import numpy as np
import pandas as pd
import trimesh
import open3d as o3d
import point_cloud_utils as pcu
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("trimesh:", trimesh.__version__)
print("open3d:", o3d.__version__)
print("point_cloud_utils OK")
PY
```

如果 `open3d` 安装或导入失败，不要阻塞训练。短期先用 `trimesh + point-cloud-utils + matplotlib` 完成指标和静态图，Open3D 放到后面专门处理。

## 3. 项目目录建议

云端建议固定这些目录，后续脚本都往这里写：

```bash
cd /workspace/g2

mkdir -p \
  data/raw \
  data/val_small \
  data/noise_benchmark \
  outputs/checkpoints \
  outputs/predictions \
  outputs/metrics \
  outputs/visualization/ply \
  outputs/visualization/png \
  outputs/reports
```

建议在每次云端登录后设置：

```bash
export PROJECT_ROOT=/workspace/g2
export STARTER_ROOT=$PROJECT_ROOT/jittor/release4/starter_code
export DATA_ROOT=$PROJECT_ROOT/data
export OUTPUT_ROOT=$PROJECT_ROOT/outputs
```

## 4. Baseline 运行命令模板

训练：

```bash
conda activate jittor-pcd
cd /workspace/g2/jittor/release4/starter_code
python run.py --task configs/task/train_vm.yaml
```

推理：

```bash
conda activate jittor-pcd
cd /workspace/g2/jittor/release4/starter_code
python run.py --task configs/task/predict_vm.yaml
```

打包提交：

```bash
cd /workspace/g2/jittor/release4/starter_code/results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

## 5. 可视化建议

云算力平台通常是无显示器的 headless 环境，所以不要把实验依赖建立在“打开一个 GUI 窗口”上。推荐分三层做：

### A. 必做：云端导出文件，本地查看

云端脚本批量导出：

- `.ply`：Noisy / Baseline / Ours / GT 点云。
- `.ply` 或 `.npz`：带 P2S/CD 局部误差颜色的热力图点云。
- `.png`：固定相机角度的三列或四列对比图。
- `.csv`：每个样本、每种噪声、每个区域的指标。

然后下载到本地，用 CloudCompare 或 MeshLab 人工检查。CloudCompare 很适合做答辩图，尤其适合手动旋转、裁剪 close-up、检查薄结构和边缘。


## 6. 推荐执行顺序

1. 先建 `jittor-pcd`，跑通官方 baseline。
2. 再建 `pcd-analysis`，跑通指标脚本和噪声生成脚本。
3. 可视化先导出 `.ply`，本地用 CloudCompare 看。
4. 等 failure case 明确后，再考虑 Open3D 离屏截图自动化。

## 参考

- Jittor 官方安装页：https://cg.cs.tsinghua.edu.cn/jittor/download/
- Open3D 官方文档：https://www.open3d.org/docs/
