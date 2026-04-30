# Jittor 1.3.10.0 + CUDA 安装指南

## 适用环境

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA RTX 4060 Laptop (compute capability 8.9) |
| 驱动 | 580.126 (CUDA 13.0) |
| g++ | 13.3.0 |
| glibc | 2.39 |
| Python | 3.12.3 |
| CUDA Toolkit | 12.8 |
| cuDNN | v8 (来自 Jittor 捆绑包) |

> **关键问题**: Ubuntu 24.04 的 g++ 13 + glibc 2.39 与 Jittor 捆绑的 nvcc 12.2 不兼容。需要混合使用系统 nvcc 12.8 + 捆绑 cuDNN v8。

## 安装步骤

### 1. 系统依赖

```bash
# CUDA Toolkit 12.8 (从 NVIDIA 官网安装)
# cuDNN v9 (apt 安装的也可以，但我们实际用捆绑包里的 v8)
sudo apt install cudnn9-cuda-12
```

### 2. 创建虚拟环境并安装 Jittor

```bash
python3 -m venv ~/jittor-env
source ~/jittor-env/bin/activate
pip install jittor==1.3.10.0
```

### 3. 下载 Jittor CUDA 捆绑包

```bash
python -m jittor_utils.install_cuda
```

这会下载 `cuda12.2_cudnn8_linux.tgz` (约 5.6 GB) 到 `~/.cache/jittor/jtcuda/`。下载完成后可以 `Ctrl+C` 退出（我们只需要里面的 cuDNN v8）。

### 4. 配置环境变量

将以下内容添加到 `~/jittor-env/bin/activate` 的末尾（VIRTUAL_ENV 设置之后）：

```bash
# Jittor: use system CUDA 12.8 nvcc instead of bundled CUDA 12.2
_OLD_VIRTUAL_NVCC_PATH="${nvcc_path:-}"
export nvcc_path=/usr/local/cuda/bin/nvcc
```

并在 `deactivate()` 函数中添加对应的恢复逻辑（在 `unset VIRTUAL_ENV` 之后）：

```bash
if [ -n "${_OLD_VIRTUAL_NVCC_PATH:-}" ] ; then
    nvcc_path="${_OLD_VIRTUAL_NVCC_PATH:-}"
    export nvcc_path
    unset _OLD_VIRTUAL_NVCC_PATH
else
    unset nvcc_path
fi
```

### 5. 修改 Jittor 源码（添加 cuDNN 搜索路径）

编辑 `~/jittor-env/lib/python3.12/site-packages/jittor/compile_extern.py`：

**修改 1** — `setup_cuda_lib()` 函数中，添加 jtcuda 路径变量并扩展 include 搜索：

```python
# 在 line ~266 "link_flags = """ 之后添加:
jtcuda_include = os.path.join(jit_utils.home(), ".cache", "jittor", "jtcuda", "cuda12.2_cudnn8_linux", "include")
jtcuda_lib = os.path.join(jit_utils.home(), ".cache", "jittor", "jtcuda", "cuda12.2_cudnn8_linux", "lib64")
```

然后将 `search_file([cuda_include, extra_include_path, "/usr/include"], ...)` 改为：
```python
search_file([cuda_include, extra_include_path, jtcuda_include, "/usr/include"], ...)
```

**修改 2** — 同样的函数中，扩展 lib 搜索路径：
```python
# 在 culib_path = search_file(...) 和 ex_cudnn_path = search_file(...) 的 dirs 列表中添加 jtcuda_lib
```

### 6. 验证安装

```bash
source ~/jittor-env/bin/activate
python -c "
import jittor as jt
print('Jittor:', jt.__version__)
print('CUDA:', jt.has_cuda)

a = jt.random((3,3))
b = jt.random((3,3))
c = a @ b
print('GPU matmul OK:', c)
"
```

预期输出：
```
Jittor: 1.3.10.0
CUDA: 1
GPU matmul OK: jt.Var([[0.89 1.60 0.95] ...])
```

## 技术要点

### 为什么需要混合方案

```
Jittor 捆绑包 (nvcc 12.2 + cuDNN v8)
├── nvcc 12.2  ──❌── g++ 13.3.0 有 _Float32 类型冲突
│                    C++ 标准库模板解析失败
└── cuDNN v8   ──✅── Jittor 代码兼容

系统 CUDA 12.8
├── nvcc 12.8  ──✅── 原生支持 g++ 13.3.0
└── cuDNN v9   ──❌── API 不兼容 (Jittor 用 v7/v8 API)

解决方案: nvcc 12.8 + cuDNN v8 = 两者取长补短
```

### nvcc_path 解析顺序

Jittor 在 `compiler.py` 中的 nvcc 查找顺序：
1. `~/.cache/jittor/jtcuda/` 存在 → 使用捆绑 nvcc
2. 环境变量 `nvcc_path`
3. 系统 PATH 中的 `nvcc`
4. `/usr/local/cuda/bin/nvcc`
5. 回退到下载捆绑包

我们通过环境变量在第 2 步拦截，强制使用系统 nvcc 12.8。

### 已知问题

- `cusparse` 库加载警告（非关键，不影响基本使用）
- 每次 pip 更新 jittor 后需要重新修改 `compile_extern.py`
