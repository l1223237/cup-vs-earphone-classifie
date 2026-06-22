import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

# 从我们之前创建的文件中导入预处理函数
from data_preprocessing import get_basic_transforms, get_train_transforms

# 数据集根目录 (包含 train, val, test 子目录)
DATASET_ROOT = "split_dataset"

# 定义默认的批处理大小
DEFAULT_BATCH_SIZE = 32

def create_dataloaders(data_dir=DATASET_ROOT, batch_size=DEFAULT_BATCH_SIZE, num_workers=0):
    """
    创建训练、验证和测试集的 PyTorch DataLoaders。

    Args:
        data_dir (str): 包含 'train', 'val', 'test' 子目录的数据集根目录。
        batch_size (int): DataLoader 使用的批处理大小。
        num_workers (int): 用于数据加载的子进程数量。0 表示在主进程中加载。

    Returns:
        tuple: 包含三个 DataLoader (train, val, test) 和训练数据集对象的元组。
               (train_loader, val_loader, test_loader, train_dataset)
               如果数据目录或子目录不存在，则返回 (None, None, None, None)。
    """
    print(f"开始创建 DataLoaders...")
    print(f"  数据目录: {data_dir}")
    print(f"  批处理大小: {batch_size}")
    print(f"  工作进程数: {num_workers}")

    # 检查根目录是否存在
    if not os.path.isdir(data_dir):
        print(f"错误：数据根目录 '{data_dir}' 不存在。")
        return None, None, None, None

    # 定义各个划分的目录路径
    image_datasets_paths = {
        split: os.path.join(data_dir, split)
        for split in ['train', 'val', 'test']
    }

    # 检查 train, val, test 子目录是否存在
    for split, path in image_datasets_paths.items():
        if not os.path.isdir(path):
            print(f"错误：子目录 '{path}' 不存在。请先运行数据集划分脚本。")
            return None, None, None, None

    # 获取转换流程
    # 现在为训练集使用带增强的转换
    transforms = {
        'train': get_train_transforms(),
        'val': get_basic_transforms(),
        'test': get_basic_transforms()
    }

    # 使用 ImageFolder 创建数据集
    try:
        image_datasets = {
            split: ImageFolder(path, transforms[split])
            for split, path in image_datasets_paths.items()
        }
        print("成功创建 ImageFolder 数据集。")
        print(f"  训练集样本数: {len(image_datasets['train'])}")
        print(f"  验证集样本数: {len(image_datasets['val'])}")
        print(f"  测试集样本数: {len(image_datasets['test'])}")
        print(f"  检测到的类别: {len(image_datasets['train'].classes)}")

        # 提取类别名称列表以备后用
        class_names = image_datasets['train'].classes

    except Exception as e:
        print(f"错误：创建 ImageFolder 数据集时出错: {e}")
        return None, None, None, None

    # 创建 DataLoaders
    try:
        dataloaders = {
            split: DataLoader(
                image_datasets[split],
                batch_size=batch_size,
                # 训练集需要打乱顺序，验证/测试集不需要
                shuffle=(split == 'train'),
                num_workers=num_workers,
                # 如果使用 GPU，pin_memory=True 可以加速数据传输
                pin_memory=torch.cuda.is_available()
            )
            for split in ['train', 'val', 'test']
        }
        print("成功创建 DataLoaders。")
    except Exception as e:
        print(f"错误：创建 DataLoader 时出错: {e}")
        return None, None, None, None


    return dataloaders['train'], dataloaders['val'], dataloaders['test'], image_datasets['train']

# 新增函数：计算类别权重
def calculate_class_weights(train_dataset):
    """
    根据训练集中的类别样本数量计算类别权重。

    Args:
        train_dataset (torchvision.datasets.ImageFolder): 训练集数据集对象。

    Returns:
        torch.Tensor: 包含每个类别权重的张量，可用于 CrossEntropyLoss。
                      如果数据集为空或无法获取类别信息，则返回 None。
    """
    print("\n开始计算类别权重...")
    try:
        # targets/labels are stored in dataset.targets for ImageFolder
        targets = train_dataset.targets
        class_counts = Counter(targets) # 统计每个类别索引出现的次数
        num_classes = len(train_dataset.classes)
        total_samples = len(targets)

        if num_classes == 0 or total_samples == 0:
            print("错误：无法从数据集中获取类别或样本信息。")
            return None

        print(f"  训练集总样本数: {total_samples}")
        print(f"  类别数: {num_classes}")

        # 计算权重: weight[c] = total_samples / (num_classes * count[c])
        # 确保按类别索引 0, 1, 2,... 的顺序生成权重
        weights = []
        for i in sorted(class_counts.keys()): # 按类别索引排序
            count = class_counts[i]
            if count == 0:
                # 处理计数为0的类别（理论上不应发生，除非数据加载错误）
                print(f"警告：类别索引 {i} 的样本数为 0，将分配权重 0。")
                weights.append(0.0)
            else:
                weight = total_samples / (num_classes * count)
                weights.append(weight)
                # 可选: 打印每个类别的权重
                # print(f"  类别 {train_dataset.classes[i]} (索引 {i}): 样本数={count}, 权重={weight:.4f}")

        # 转换为 PyTorch 张量
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print("类别权重计算完成。")
        # print(f"  权重张量: {class_weights_tensor}") # 完整打印可能很长
        return class_weights_tensor

    except AttributeError:
        print("错误：提供的数据集对象似乎不是标准的 ImageFolder 或缺少 'targets'/'classes' 属性。")
        return None
    except Exception as e:
        print(f"错误：计算类别权重时出错: {e}")
        return None

if __name__ == '__main__':
    # 示例用法
    batch_size = 32
    # 尝试设置 num_workers=0 以提高初始兼容性
    num_workers = 0
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(batch_size=batch_size, num_workers=num_workers)

    if train_loader and val_loader and test_loader:
        print("\nDataLoader 创建成功!")

        # 打印一些信息来验证
        print(f"类别名称 ({len(train_dataset.classes)} 类): {train_dataset.classes[:10]}...")
        print(f"类别到索引的映射 (部分): {list(train_dataset.class_to_idx.items())[:5]}")

        # --- 新增：计算并打印类别权重 ---
        class_weights = calculate_class_weights(train_dataset)
        if class_weights is not None:
            print(f"\n计算得到的类别权重 (前10个):")
            print(class_weights[:10])
            # 注意：实际训练时，需要将此权重张量移到与模型和数据相同的设备上
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # class_weights = class_weights.to(device)
            # print(f" (示例：权重张量应移至设备 {device})")
        else:
            print("\n未能计算类别权重。")
        # --- 结束新增部分 ---

        # 检查一个批次的数据
        try:
            # 从训练数据加载器中获取一个批次
            images, labels = next(iter(train_loader))
            print(f"\n从 train_loader 获取了一个批次:")
            print(f"  图像批次形状: {images.shape}") # 应该是 [batch_size, channels, height, width]
            print(f"  标签批次形状: {labels.shape}")   # 应该是 [batch_size]
            print(f"  一个图像张量的均值: {images[0].mean():.4f}") # 检查值是否在合理范围内 (标准化后接近0)
            print(f"  一个批次中的部分标签: {labels[:10]}") # 打印前10个标签索引
        except Exception as e:
            print(f"\n尝试从 DataLoader 获取批次时出错: {e}")
            if num_workers > 0:
                 print("这可能是由于 num_workers > 0 在某些系统/配置下的问题，尝试设置 num_workers=0。")

    else:
        print("\nDataLoader 创建失败。请检查错误信息。") 