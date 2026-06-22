# cup-vs-earphone-classifie
# 图像分类系统：水杯 vs 耳机

基于 PyTorch 和 ResNet50 的图像分类项目，实现二分类任务的训练与推理。

---

# 项目简介

本项目是一个完整的图像分类工具，能够区分“水杯”和“耳机”两类物体。包含从 **数据准备 → 模型训练 → 模型评估 → 单图推理** 的完整闭环。

---
# 技术栈

- Python 3.14
- PyTorch + Torchvision
- ResNet50（迁移学习）
- OpenCV（图像处理）
- 测试准确率：**100%**

---

##  项目结构
cup_vs_earphone_classifier/
├── README.md # 项目说明
├── requirements.txt # 依赖清单
├── train.py # 训练脚本
├── predict.py # 推理脚本（支持结果可视化）
├── models.py # 模型定义
├── data/ # 数据集
│ ├── train/
│ │ ├── cup/
│ │ └── earphone/
│ └── val/
│ ├── cup/
│ └── earphone/
├── checkpoints/ # 训练好的模型权重
│ └── best_model_resnet50.pth
├── evaluation_results/ # 评估结果（混淆矩阵、分类报告）
└── prediction_demo.jpg # 推理结果展示图

---

## 快速开始

### 1. 安装依赖
    ```bash
     pip install -r requirements.txt

### 2.训练模型
如果你有自己的数据集，可以重新训练：

bash
python train.py
训练完成后，模型权重会保存在 checkpoints/ 目录下。

### 3. 识别新图片
bash
python predict.py
按提示输入图片路径（如 test.jpg），程序会输出识别结果和置信度，并自动生成带标注的结果图 result.jpg。

## 训练结果
模型	测试准确率
ResNet50	100%
训练过程通过迁移学习，基于 ImageNet 预训练权重进行微调，使用 Adam 优化器 + 权重衰减，学习率调度采用 ReduceLROnPlateau。
[推理演示](prediction_demo.jpg)

## 致谢
本项目基于开源项目 [基于深度学习的植物叶片识别方法研究] 进行本地化适配与二次开发
