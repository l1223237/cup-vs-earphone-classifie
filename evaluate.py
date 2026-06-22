import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
import time
import matplotlib.font_manager # 导入用于字体调试

# 从项目模块导入
# 确保 models.py 和 data_loader.py 在同一目录下或 Python 路径中
try:
    from models import create_resnet50, create_efficientnet_b0, create_efficientnet_b1 # 需要模型定义
    from data_loader import create_dataloaders # 需要加载测试数据
except ImportError as e:
    print(f"错误：无法导入项目模块 (models, data_loader)。请确保它们存在且在 Python 路径中。 {e}")
    exit()
# data_preprocessing 会被 data_loader 间接调用

# --- 配置 ---
MODEL_NAME = "resnet50" # 或者 "efficientnet_b0", "efficientnet_b1" - 需要与训练时保存的模型一致
NUM_CLASSES = 2
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILENAME = f"best_model_{MODEL_NAME}.pth" # 假设这是训练脚本保存的文件名
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
DATA_DIR = "my-data"
BATCH_SIZE = 512 # 评估时可以使用稍大的批次大小
NUM_WORKERS = 16 # 设为 0 避免多进程问题，除非你确定要使用
OUTPUT_DIR = "evaluation_results" # 保存结果的目录

preferred_fonts = [
    'WenQuanYi Micro Hei',      # 常见 Linux 中文字体
    'Noto Sans CJK SC',         # 较新的 Linux/跨平台 CJK 字体 (简体中文变体)
    'Source Han Sans CN',       # Adobe/Google CJK 字体 (简体中文)
    'SimHei',                   # Windows/Linux (若安装)
    'Droid Sans Fallback',      # Android/Linux 回退字体
    'PingFang SC',              # macOS 常见
    'Heiti SC',                 # macOS 备选
    'STHeiti',                  # macOS 备选
    'Heiti TC',                 # macOS (繁体)
    'Microsoft YaHei',          # Windows (若安装)
    'Arial Unicode MS'          # 广泛但可能需要安装
]

# 设置 Matplotlib 支持中文和 DPI
try:
    # 尝试设置字体列表
    plt.rcParams['font.sans-serif'] = preferred_fonts
    plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
    plt.rcParams['figure.dpi'] = 300 # DPI 设为 300

    # 验证实际使用的字体 (可选，用于调试)
    try:
        # 获取 Matplotlib 找到并使用的第一个字体
        actual_font = matplotlib.font_manager.FontProperties(family=plt.rcParams['font.sans-serif'][0]).get_name()
        print(f"Matplotlib 尝试使用字体: {actual_font}")
    except IndexError:
        print("警告：字体列表为空或 Matplotlib 未找到任何指定字体。")
    except Exception as font_exc:
        print(f"警告：检查字体时出错: {font_exc}") # 更详细的字体错误

except Exception as e:
    print(f"警告：设置中文字体时出错: {e}。已尝试 {preferred_fonts}。标签可能无法正确显示。")
    print("请确保您的 Linux 系统已安装中文字体 (如 wqy-microhei, noto-fonts-cjk) 并清除了 matplotlib 缓存。")
    # print("回退到 Matplotlib 默认字体。")
    # plt.rcParams['font.sans-serif'] = plt.rcParamsDefault['font.sans-serif']


def evaluate_model(model, dataloader, device):
    """
    在给定的数据集上评估模型，并收集所有预测和标签。

    Args:
        model (torch.nn.Module): 加载了权重的模型。
        dataloader (DataLoader): 测试数据加载器。
        device (torch.device): 运行设备。

    Returns:
        tuple: (all_preds, all_labels) 包含所有预测标签和真实标签的 numpy 数组。
               如果出错则返回 (None, None)。
    """
    model.eval() # 设置为评估模式
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    try:
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        progress_bar.close()
        return np.array(all_preds), np.array(all_labels)
    except Exception as e:
        print(f"\n错误：评估过程中出错: {e}")
        progress_bar.close()
        return None, None


def plot_confusion_matrix(labels, preds, class_names, output_filename="confusion_matrix.png"):
    """
    计算并绘制混淆矩阵。

    Args:
        labels (np.array): 真实标签数组。
        preds (np.array): 预测标签数组。
        class_names (list): 类别名称列表，用于轴标签。
        output_filename (str): 保存混淆矩阵图像的文件路径。
    """
    if labels is None or preds is None:
        print("错误：无法生成混淆矩阵，因为标签或预测为空。")
        return

    print("\n正在生成混淆矩阵...")
    try:
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(20, 18)) # 调整大小以适应 48 个类别
        # 对于类别较多的情况，annot=False 效果更好，避免数字重叠
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'混淆矩阵 - {MODEL_NAME}', fontsize=16)
        plt.ylabel('真实类别', fontsize=12)
        plt.xlabel('预测类别', fontsize=12)
        plt.xticks(rotation=90, fontsize=8) # 旋转标签并减小字体
        plt.yticks(rotation=0, fontsize=8) # 减小字体
        plt.tight_layout() # 自动调整布局

        # 确保保存前输出目录存在
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename)
        print(f"混淆矩阵已保存至: {output_filename}")
    except Exception as e:
        print(f"错误：生成或保存混淆矩阵时出错: {e}")
    finally:
        plt.close() # 关闭图形，释放内存

if __name__ == '__main__':
    print("--- 开始模型评估 ---")
    print(f"模型: {MODEL_NAME}")
    print(f"检查点路径: {CHECKPOINT_PATH}")

    # --- 创建输出目录 ---
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"已创建评估结果目录: {OUTPUT_DIR}")
        except OSError as e:
            print(f"错误：无法创建输出目录 '{OUTPUT_DIR}': {e}")
            exit()

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载数据 ---
    print("\n--- 加载测试数据 ---")
    try:
        # 同时获取训练集，以获取 class_to_idx 和 classes
        train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
            data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        if test_loader is None or train_dataset is None:
            print("错误：从 create_dataloaders 返回了 None。评估中止。")
            exit()
        # 从训练数据集中获取类别名称
        class_names = train_dataset.classes
        print(f"获取到 {len(class_names)} 个类别名称。")
    except Exception as e:
        print(f"错误：加载数据时出错: {e}")
        exit()


    # --- 加载模型 ---
    print(f"\n--- 加载模型 ({MODEL_NAME}) ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误：找不到检查点文件 '{CHECKPOINT_PATH}'。请确保模型已训练并保存。")
        exit()

    try:
        # 初始化模型结构 (不需要预训练权重)
        if MODEL_NAME == "resnet50":
            model = create_resnet50(num_classes=NUM_CLASSES, pretrained=False)
        elif MODEL_NAME == "efficientnet_b0":
            model = create_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False)
        elif MODEL_NAME == "efficientnet_b1":
            model = create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=False)
        else:
            print(f"错误：不支持的模型名称 '{MODEL_NAME}'")
            exit()

        # 加载保存的状态字典
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # 兼容直接保存 state_dict 和保存包含 'state_dict' 的字典
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("从检查点字典中加载 'state_dict' 成功。")
        else:
            model.load_state_dict(checkpoint)
            print("直接加载状态字典成功。")

        model = model.to(device)
        # print("模型权重加载成功。") # 上面已经打印了

    except Exception as e:
        print(f"错误：加载模型或权重时出错: {e}")
        exit()

    # --- 执行评估 ---
    print("\n--- 在测试集上执行评估 ---")
    start_time = time.time()
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    eval_duration = time.time() - start_time

    if all_preds is None or all_labels is None:
        print("评估未能完成，无法生成报告和混淆矩阵。")
        exit()

    print(f"评估完成，耗时: {eval_duration:.2f} 秒")

    # --- 计算总体准确率 (可选补充) ---
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n--- 总体准确率 ---")
    print(f"Accuracy: {accuracy:.4f}")


    # --- 计算并打印详细指标 ---
    print("\n--- 分类报告 (精确率, 召回率, F1分数) ---")
    if class_names: # 确保 class_names 不是 None 或空列表
        try:
            # 设置 zero_division=0 避免在某些类别没有预测样本或没有真实样本时产生警告/错误
            report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, digits=4) # 增加小数位数
            print(report)
            # 可以将报告保存到文件
            report_filename = os.path.join(OUTPUT_DIR, f"classification_report_{MODEL_NAME}.txt")
            with open(report_filename, "w", encoding="utf-8") as f: # 使用 utf-8 编码
                f.write(f"模型: {MODEL_NAME}\n检查点: {CHECKPOINT_PATH}\n\n")
                f.write(f"总体准确率: {accuracy:.4f}\n\n") # 也保存总体准确率
                f.write(report)
            print(f"分类报告已保存至: {report_filename}")
        except Exception as e:
            print(f"错误：生成或保存分类报告时出错: {e}")
    else:
        print("警告：无法生成带类别名称的分类报告，因为类别名称列表为空。")
        # 可以选择生成不带名称的报告
        try:
            report = classification_report(all_labels, all_preds, zero_division=0, digits=4)
            print(report)
        except Exception as e:
            print(f"错误：生成无名称分类报告时出错: {e}")


    # --- 生成并保存混淆矩阵 ---
    if class_names: # 只有在有类别名称时才绘制带标签的混淆矩阵
        cm_filename = os.path.join(OUTPUT_DIR, f"confusion_matrix_{MODEL_NAME}.png")
        plot_confusion_matrix(all_labels, all_preds, class_names, output_filename=cm_filename)
    else:
        print("警告：无法生成带类别标签的混淆矩阵，因为类别名称列表为空。")

    print("\n--- 评估脚本执行完毕 ---")