import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import torch # Need torch for RandomErasing

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600

# 定义图像的目标尺寸
# 许多预训练模型使用 224x224 或 256x256
TARGET_IMG_SIZE = 224

# 定义用于标准化的 ImageNet 均值和标准差
# 这些值是根据 ImageNet 数据集计算得出的，广泛用于迁移学习
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_basic_transforms(target_size=TARGET_IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    获取基础的图像预处理流程（用于验证/测试集）。

    Args:
        target_size (int): 调整后的图像目标边长。
        mean (list): 用于标准化的均值。
        std (list): 用于标准化的标准差。

    Returns:
        transforms.Compose: 包含一系列预处理操作的对象。
    """
    preprocess_transforms = transforms.Compose([
        # 1. 调整图像大小：将图像调整为指定的目标尺寸 (target_size x target_size)。
        #    对于非方形图像，它会调整较短边到 target_size，并保持宽高比，
        #    但通常我们会结合 CenterCrop 或直接 Resize 到方形。这里我们直接 Resize 到方形。
        transforms.Resize((target_size, target_size)),

        # 2. 转换为张量：将 PIL Image 或 numpy.ndarray (H x W x C)
        #    转换为 torch.FloatTensor (C x H x W)，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]。
        transforms.ToTensor(),

        # 3. 标准化：使用给定的均值和标准差对张量图像进行标准化。
        #    公式: output[channel] = (input[channel] - mean[channel]) / std[channel]
        #    这有助于模型更快地收敛，并适应预训练模型的输入分布。
        transforms.Normalize(mean=mean, std=std)
    ])
    return preprocess_transforms

# 新增函数：为训练集获取带数据增强的转换
def get_train_transforms(target_size=TARGET_IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    获取包含基础和高级数据增强（包括随机擦除）的训练集图像预处理流程。
    """
    train_transforms = transforms.Compose([
        # --- PIL-based Augmentations ---
        # 1. 随机裁剪并调整大小
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        # 2. 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 3. 随机旋转
        transforms.RandomRotation(degrees=15),
        # 4. 随机仿射变换
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # 5. 颜色抖动
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # --- Tensor-based Augmentations ---
        # 6. 转换为张量
        transforms.ToTensor(),
        # 7. 标准化
        transforms.Normalize(mean=mean, std=std),
        # 8. 随机擦除 (Tensor -> Tensor) - 新增
        #    p=0.5: 50%的概率应用擦除
        #    scale=(0.02, 0.2): 擦除区域面积占原图的比例范围
        #    ratio=(0.3, 3.3): 擦除区域的宽高比范围
        #    value=0: 用 0 填充擦除区域 (黑色)，也可以是 'random' 或指定值
        #    inplace=False: 不修改原始张量
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
    ])
    return train_transforms

# 新增函数：可视化数据增强效果
def visualize_augmentations(dataset_path, num_examples=5, output_filename="data_augmentation_example.png"):
    """
    加载一张随机图像，并展示应用训练集数据增强（包括擦除）后的多个不同结果。
    注意：由于擦除作用于张量，且通常在标准化后，这里的可视化仅为近似效果。
    """
    print(f"\n开始生成数据增强可视化示例 ({output_filename})...")
    try:
        # 随机选择一个类别
        valid_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        if not valid_classes:
            print(f"错误：在 '{dataset_path}' 中找不到有效的类别目录。")
            return
        class_name = random.choice(valid_classes)
        class_dir = os.path.join(dataset_path, class_name)

        # 随机选择一张图像
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and not f.startswith('.')]
        if not image_files:
            print(f"错误：类别 '{class_name}' 中没有图像文件。")
            return
        image_file = random.choice(image_files)
        original_img_path = os.path.join(class_dir, image_file)

        # 加载原始图像
        original_img = Image.open(original_img_path).convert('RGB') # 确保是 RGB
        print(f"  选择图像: {original_img_path}")

        # 定义用于可视化的增强转换 (应用到 Tensor，不含标准化)
        # 将 RandomErasing 应用于 [0, 1] 范围的 Tensor
        augmentation_transforms_for_vis = transforms.Compose([
            # PIL based
            transforms.RandomResizedCrop(TARGET_IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # To Tensor
            transforms.ToTensor(),
            # Tensor based (Random Erasing)
            # p=1.0 确保可视化时总能看到效果
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
            # Back to PIL for display
            transforms.ToPILImage()
        ])

        # 创建画布
        # 1 (原图) + num_examples (增强后)
        cols = num_examples + 1
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2.5, 3)) # 调整画布大小

        # 显示原图
        axes[0].imshow(original_img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))) # 也调整下大小方便对比
        axes[0].set_title(f"原始图像\n({class_name})", fontsize=10)
        axes[0].axis('off')

        # 显示增强后的图像
        for i in range(num_examples):
            # 应用转换以生成增强图像 (PIL -> Tensor -> PIL)
            augmented_img_pil = augmentation_transforms_for_vis(original_img)
            axes[i+1].imshow(augmented_img_pil)
            axes[i+1].set_title(f"增强后 {i+1}", fontsize=10)
            axes[i+1].axis('off')

        fig.suptitle("数据增强效果示例 (含随机擦除)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局以适应标题

        # 保存图像
        plt.savefig(output_filename)
        print(f"  可视化结果已保存为: {output_filename}")

    except FileNotFoundError:
         print(f"错误: 找不到原始图像文件 '{original_img_path}' 或目录。")
    except Exception as e:
        print(f"错误: 生成数据增强可视化时出错: {e}")

if __name__ == '__main__':
    # 示例：获取并打印转换流程
    basic_transforms = get_basic_transforms()
    print("定义的基础图像预处理流程 (用于验证/测试):")
    print(basic_transforms)
    print("\n" + "="*30 + "\n")
    train_transforms_instance = get_train_transforms()
    print("更新后的训练图像预处理流程 (带随机擦除):")
    print(train_transforms_instance)

    # 示例：可视化增强效果
    # 注意：这里使用原始数据集路径，因为 split_dataset/train 可能文件较少
    # 如果原始数据集不在项目根目录，需要调整路径
    original_dataset_dir = "Leaves of 48 Plants Dataset"
    visualize_augmentations(original_dataset_dir, num_examples=5)

    # 注意：这个脚本本身不处理任何图像，只是定义了转换流程。
    # 这些转换将在后续创建 PyTorch Dataset 和 DataLoader 时使用。 