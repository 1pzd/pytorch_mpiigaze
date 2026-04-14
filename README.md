## 项目文件/文件夹 完整说明
### 一、根目录核心文件（直接运行/配置）

| 文件名 | 作用 |
| :--- | :--- |
| `train.py` | **模型训练主脚本**，用于训练 MPIIGaze/MPIIFaceGaze 视线估计模型 |
| `evaluate.py` | **模型评估主脚本**，测试训练好的模型，计算视线预测误差 |
| `demo.py` | **实时演示脚本**，调用摄像头/视频，实时显示人脸+视线方向 |
| `requirements.txt` | 项目依赖清单，一键安装所有需要的Python库 |
| `README.md` | 项目说明文档，基础介绍、使用方法 |
| `.gitignore` | Git忽略文件配置，排除不需要上传的文件（模型权重、数据集等） |

---

### 二、核心文件夹详解
#### 📁 `configs/`
**作用**：项目**所有配置文件**（YAML格式），无需改代码，改这里即可调整参数
- `mpiigaze/`：MPIIGaze 数据集训练/评估配置（模型、学习率、批次大小等）
- `mpiifacegaze/`：MPIIFaceGaze 数据集训练/评估配置
- `demo_xxx.yaml`：Demo 实时演示配置文件

**以 MPIIGaze 配置为例：**

| 文件名 | 模型 | 阶段 | 数据范围 |
| :--- | :--- | :--- | :--- |
| `lenet_train.yaml` | LeNet | 训练 | 部分 / 默认数据 |
| `lenet_eval.yaml` | LeNet | 评估 | 部分 / 默认数据 |
| `lenet_train_using_all_data.yaml` | LeNet | 训练 | 全量数据 |
| `resnet_preact_train.yaml` | ResNet (Pre-Act) | 训练 | 部分 / 默认数据 |
| `resnet_preact_eval.yaml` | ResNet (Pre-Act) | 评估 | 部分 / 默认数据 |
| `resnet_preact_train_using_all_data.yaml` | ResNet (Pre-Act) | 训练 | 全量数据 |
| `alexnet_train.yaml` | **AlexNet** | 训练 | 部分 / 默认数据 |
| `alexnet_eval.yaml` | **AlexNet** | 评估 | 部分 / 默认数据 |

#### 📁 `data/`
**作用**：存放项目运行所需的**静态资源文件**
- `calib/`：相机标定参数（用于图像矫正、3D坐标计算）
- `dlib/`：dlib 人脸68关键点检测预训练模型（Demo必需）
- `models/`：训练完成的模型权重文件（自动保存于此）

#### 📁 `gaze_estimation/`
**作用**：✅ **项目核心代码包**（整个项目的算法灵魂）
- 模型定义、数据加载、视线估计算法、头部姿态估计、可视化工具
- 损失函数、优化器、配置管理、数据预处理等全部核心逻辑
- **自定义模型说明**：在 `gaze_estimation/models/mpiigaze/` 目录下可以添加自定义模型代码（例如 `alexnet.py`），通过在 YAML 配置文件中指定 `model.name` 即可实现动态调用。

#### 📁 `tools/`
**作用**：数据预处理/辅助工具脚本
- `preprocess_xxx.py`：将原始数据集转换为项目可用格式
- `capture_video.py`：录制视频，用于Demo测试

#### 📁 `scripts/`
**作用**：一键自动化脚本（Shell脚本）
- 自动下载数据集、预训练模型
- 批量训练/评估所有测试样本，无需手动敲命令

#### 📁 `figures/`
**作用**：项目效果图存放
- 训练曲线、模型精度对比图、Demo演示截图

---

## 零基础训练与运行指南

### 第一步：准备数据与模型（必做）
**1. 下载 MPIIGaze 数据集并预处理**
（说明：本项目依赖 `.h5` 格式的数据集，需要先下载原始数据并用脚本转换）
运行脚本自动下载人脸检测模型和数据集：
```bash
# 下载dlib人脸关键点模型（Demo必需）
sh scripts/download_dlib_model.sh

# 下载MPIIGaze数据集（如果你还没有下载的话）
sh scripts/download_mpiigaze_dataset.sh
```
> Windows 系统如果无法运行 sh 脚本，可以打开脚本文件，手动复制里面的链接到浏览器下载，并放到对应的 `data/` 文件夹下。

**预处理数据**：使用 `tools/` 目录下的脚本将原始数据集处理为 h5 格式，放到指定的输出目录（通常配置在 yaml 文件的 `dataset.dataset_dir`，比如 `output/MPIIGaze.h5`）。

---

### 第二步：训练视线估计模型

所有训练的参数（学习率、模型选择、数据路径等）都在 `configs/` 下的 yaml 文件里。你只需要指定想用的配置即可。

**示例 1：使用经典的 LeNet 模型训练**
```bash
python train.py --config configs/mpiigaze/resnet_preact_train.yaml
```

**示例 2：使用最新添加的 AlexNet 模型训练**
我们已经在代码中加入了针对 MPIIGaze 数据集适配的 AlexNet 模型，直接运行：
```bash
python train.py --config configs/mpiifacegaze/resnet_simple_14_train.yaml
```

**训练过程说明：**
- 运行后，终端会打印出当前的网络配置并开始逐个 Epoch 训练。
- 训练完成后，模型权重文件（checkpoint）会自动保存到 `experiments/mpiigaze/` 或 `data/models/` 对应的子文件夹中。
- 你可以使用 Tensorboard 观察训练曲线。

---

### 第三步：评估模型精度

测试训练好的模型，计算预测的平均角度误差（误差越小越精准）。你需要使用对应的 eval 配置文件，并在配置文件中指定刚才训练好的模型权重路径（`test.checkpoint`）。

以评估 AlexNet 为例：
```bash
python evaluate.py --config configs/mpiigaze/alexnet_eval.yaml
python evaluate.py --config configs/mpiifacegaze/alexnet_eval.yaml
python evaluate.py --config configs/mpiigaze/resnet_preact_eval.yaml
python evaluate.py --config configs/mpiifacegaze/resnet_simple_14_eval.yaml
```
终端会输出当前模型在测试集上的平均角度误差。

