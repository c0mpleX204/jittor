# jittor-jintiansantiaoqu-denoising

本仓库用于计图人工智能挑战赛代码开源，包含热身赛代码与正赛点云降噪代码。

## 目录

- `release/`: 热身赛赛题一代码。
- `release2/`: 热身赛赛题二代码。
- `release4/starter_code/`: 正赛点云降噪 baseline 与 DCVM V1 实验代码。

## 环境安装

正赛代码建议使用 Python 3.9、CUDA 11.x、Jittor 1.3.x。

```bash
python -m pip install -r release4/starter_code/requirements.txt
python -m pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

热身赛赛题一还需要按 JittorGeometric 官方说明安装相关依赖。

## 数据准备

本仓库不包含数据集、模型权重和训练产物。

正赛数据目录示例：

```text
release4/starter_code/dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
release4/starter_code/dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

如数据放在其他位置，请修改 `release4/starter_code/configs/data/*.yaml` 中的 `input_dataset_dir`。

## 训练

正赛 baseline：

```bash
cd release4/starter_code
python run.py --task configs/task/train_vm.yaml
```

正赛 DCVM V1：

```bash
cd release4/starter_code
python run.py --task configs/task/train_dcvm_autodl_tmp.yaml
```

## 推理与评测

DCVM V1 官方测试集推理：

```bash
cd release4/starter_code
python run.py --task configs/task/predict_submit_dcvm_autodl_tmp.yaml
```

本地固定测试集评分与可视化工具位于：

```text
release4/starter_code/tools/local_score_and_visualize.py
```

## 结果说明

正赛评价指标为 Chamfer Distance 改善分与 Point-to-Surface 改善分的组合。仓库中的训练配置和路径需要根据实际云平台数据位置调整，复现实验结果可能因随机种子、硬件和训练轮次略有差异。

## License

MIT License.
