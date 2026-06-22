import torch
import torch.nn as nn
import torchvision.models as models
try:
    from torchsummary import summary
    torchsummary_imported = True
except ImportError:
    torchsummary_imported = False
    print("警告: torchsummary 未安装，无法打印详细模型摘要。")
    print("运行 'pip install torchsummary' 来安装。")

# 数据集的类别数量
NUM_CLASSES = 2

def create_resnet50(num_classes=NUM_CLASSES, pretrained=True):
    """
    创建并配置一个 ResNet50 模型。

    Args:
        num_classes (int): 数据集的类别数量。
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    Returns:
        torch.nn.Module: 配置好的 ResNet50 模型。
    """
    print(f"创建 ResNet50 模型 (预训练: {pretrained})...")
    # 加载预训练的 ResNet50 模型
    # 使用推荐的权重 API
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # 获取原始全连接层的输入特征数量
    num_ftrs = model.fc.in_features

    # 替换原来的全连接层 (通常是1000类输出) 为一个新的、
    # 具有我们数据集类别数量 (num_classes) 输出的全连接层。
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"  已将 ResNet50 的最终全连接层替换为 {num_classes} 个输出。")

    return model

def create_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    """
    创建并配置一个 EfficientNet-B0 模型。

    Args:
        num_classes (int): 数据集的类别数量。
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    Returns:
        torch.nn.Module: 配置好的 EfficientNet-B0 模型。
    """
    print(f"创建 EfficientNet-B0 模型 (预训练: {pretrained})...")
    # 加载预训练的 EfficientNet-B0 模型
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # 获取原始分类器最后一个线性层的输入特征数量
    # EfficientNet 的分类器通常是一个包含 Dropout 和 Linear 的 Sequential
    num_ftrs = model.classifier[1].in_features

    # 替换原来的线性层
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    print(f"  已将 EfficientNet-B0 的最终分类器层替换为 {num_classes} 个输出。")

    return model

def create_efficientnet_b1(num_classes=NUM_CLASSES, pretrained=True):
    """
    创建并配置一个 EfficientNet-B1 模型。

    Args:
        num_classes (int): 数据集的类别数量。
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    Returns:
        torch.nn.Module: 配置好的 EfficientNet-B1 模型。
    """
    print(f"创建 EfficientNet-B1 模型 (预训练: {pretrained})...")
    # 加载预训练的 EfficientNet-B1 模型
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.efficientnet_b1(weights=weights)

    # 获取原始分类器最后一个线性层的输入特征数量
    num_ftrs = model.classifier[1].in_features

    # 替换原来的线性层
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    print(f"  已将 EfficientNet-B1 的最终分类器层替换为 {num_classes} 个输出。")

    return model

if __name__ == '__main__':
    print(f"为 {NUM_CLASSES} 个类别创建模型并统计参数...\n")
    # 定义输入图像的尺寸 (Channels, Height, Width)
    # 需要与我们预处理中定义的 TARGET_IMG_SIZE 一致 (data_preprocessing.py)
    INPUT_SHAPE = (3, 224, 224)

    # 确定设备 (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}\n")

    # --- ResNet50 ---
    resnet_model = create_resnet50().to(device)
    print("\nResNet50 模型摘要:")
    if torchsummary_imported:
        try:
            summary(resnet_model, input_size=INPUT_SHAPE, device=device.type)
        except Exception as e:
            print(f"  生成 torchsummary 时出错: {e}")
            print("  回退: 模型最后层结构:")
            print(resnet_model.fc)
    else:
        print("  torchsummary 未安装，跳过详细摘要。")
        print("  模型最后层结构:")
        print(resnet_model.fc)

    print("\n" + "="*30 + "\n")

    # --- EfficientNet-B0 ---
    effnet_b0_model = create_efficientnet_b0().to(device)
    print("\nEfficientNet-B0 模型摘要:")
    if torchsummary_imported:
        try:
            summary(effnet_b0_model, input_size=INPUT_SHAPE, device=device.type)
        except Exception as e:
            print(f"  生成 torchsummary 时出错: {e}")
            print("  回退: 模型分类器结构:")
            print(effnet_b0_model.classifier)
    else:
        print("  torchsummary 未安装，跳过详细摘要。")
        print("  模型分类器结构:")
        print(effnet_b0_model.classifier)

    print("\n" + "="*30 + "\n")

    # --- EfficientNet-B1 ---
    effnet_b1_model = create_efficientnet_b1().to(device)
    print("\nEfficientNet-B1 模型摘要:")
    if torchsummary_imported:
        try:
            summary(effnet_b1_model, input_size=INPUT_SHAPE, device=device.type)
        except Exception as e:
            print(f"  生成 torchsummary 时出错: {e}")
            print("  回退: 模型分类器结构:")
            print(effnet_b1_model.classifier)
    else:
        print("  torchsummary 未安装，跳过详细摘要。")
        print("  模型分类器结构:")
        print(effnet_b1_model.classifier)

    print("\n模型参数统计完成。") 