import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 导入学习率调度器
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import copy # 用于深拷贝模型状态以保存最佳模型
from tqdm import tqdm # For nice progress bars

# 从我们项目中的其他模块导入
from models import create_resnet50, create_efficientnet_b0, create_efficientnet_b1 # 根据需要选择模型
from data_loader import create_dataloaders, calculate_class_weights
# 注意：data_preprocessing 在 data_loader 中被间接使用

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练模型一个周期。

    Args:
        model (torch.nn.Module): 要训练的模型。
        dataloader (DataLoader): 训练数据加载器。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 运行设备 (e.g., 'cuda' or 'cpu')。

    Returns:
        tuple: 包含平均训练损失和平均训练准确率的元组。
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(dataloader, desc="Train Epoch", leave=False)

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # 更新进度条显示
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.4f}", lr=f"{current_lr:.1e}")

    progress_bar.close()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    在验证集上评估模型。

    Args:
        model (torch.nn.Module): 要评估的模型。
        dataloader (DataLoader): 验证数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 运行设备。

    Returns:
        tuple: 包含平均验证损失和平均验证准确率的元组。
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validate", leave=False)

    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.4f}")

    progress_bar.close()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc

# --- 主训练逻辑将在下方 if __name__ == '__main__': 中实现 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    MODEL_NAME = "resnet50"  # 或 "efficientnet_b0", "efficientnet_b1"
    NUM_CLASSES = 2         # 确保与 models.py 和数据一致
    BATCH_SIZE = 8          # 根据你的硬件调整
    EPOCHS = 10               # 增加周期数以便观察效果
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4       # 新增：权重衰减系数
    LR_SCHEDULER_PATIENCE = 5 # 新增：LR调度器耐心
    LR_SCHEDULER_FACTOR = 0.1 # 新增：LR降低因子
    EARLY_STOPPING_PATIENCE = 10 # 新增：早停耐心
    DATA_DIR = "my-data"
    CHECKPOINT_DIR = "checkpoints" # 新增：模型保存目录
    # 设置 num_workers=0 确保初始兼容性，之后可以根据系统性能调整
    NUM_WORKERS = 0

    print("--- 开始训练配置 ---")
    print(f"模型: {MODEL_NAME}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"周期数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"权重衰减: {WEIGHT_DECAY}")
    print(f"LR调度器耐心: {LR_SCHEDULER_PATIENCE}, 因子: {LR_SCHEDULER_FACTOR}")
    print(f"早停耐心: {EARLY_STOPPING_PATIENCE}")
    print(f"数据目录: {DATA_DIR}")
    print(f"检查点目录: {CHECKPOINT_DIR}")

    # --- 创建检查点目录 ---
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"已创建检查点目录: {CHECKPOINT_DIR}")

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据加载 ---
    print("\n--- 加载数据 ---")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    if not train_loader or not val_loader:
        print("错误：无法创建 DataLoaders，训练中止。")
        exit() # 或者 raise Exception

    # --- 类别权重 (处理不平衡) ---
    print("\n--- 计算类别权重 ---")
    class_weights = calculate_class_weights(train_dataset)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"类别权重已计算并移至 {device}")
    else:
        print("警告：无法计算类别权重，将不使用加权损失。")

    # --- 模型选择与初始化 ---
    print(f"\n--- 初始化模型 ({MODEL_NAME}) ---")
    if MODEL_NAME == "resnet50":
        model = create_resnet50(num_classes=NUM_CLASSES)
    elif MODEL_NAME == "efficientnet_b0":
        model = create_efficientnet_b0(num_classes=NUM_CLASSES)
    elif MODEL_NAME == "efficientnet_b1":
        model = create_efficientnet_b1(num_classes=NUM_CLASSES)
    else:
        print(f"错误：不支持的模型名称 '{MODEL_NAME}'")
        exit() # Or raise

    model = model.to(device)

    # --- 损失函数 ---
    print("\n--- 定义损失函数 ---")
    # 使用类别权重 (如果成功计算)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"损失函数: CrossEntropyLoss {'(带权重)' if class_weights is not None else ''}")

    # --- 优化器 (加入权重衰减) ---
    print("\n--- 定义优化器 ---")
    # Adam 是一个常用的优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"优化器: Adam, 学习率: {LEARNING_RATE}, 权重衰减: {WEIGHT_DECAY}")

    # --- 学习率调度器 ---
    print("\n--- 定义学习率调度器 ---")
    # 当验证损失在 'patience' 个周期内不再下降时，降低学习率
    # verbose=True 会在降低学习率时打印信息
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR,
                                  patience=LR_SCHEDULER_PATIENCE)
    print(f"调度器: ReduceLROnPlateau (监测验证损失, 耐心={LR_SCHEDULER_PATIENCE}, 因子={LR_SCHEDULER_FACTOR})")


    # --- 训练循环 (加入模型保存和早停) ---
    print("\n--- 开始训练循环 ---")
    start_time = time.time()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None # 用于保存最佳模型状态

    # 用于记录历史指标，方便后续绘图或分析
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录历史数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"  耗时: {epoch_duration:.2f} 秒")

        # --- 学习率调度 ---
        # 基于验证损失调整学习率
        scheduler.step(val_loss)

        # --- 模型保存与早停逻辑 ---
        # 基于验证损失判断是否保存最佳模型
        if val_loss < best_val_loss:
            print(f"  验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}。保存模型...")
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 使用 deepcopy 确保保存的是当前状态，而不是后续可能变化的 model 引用
            best_model_state = copy.deepcopy(model.state_dict())
            # 定义保存路径
            save_path = os.path.join(CHECKPOINT_DIR, f"best_model_{MODEL_NAME}.pth")
            try:
                torch.save(best_model_state, save_path)
                print(f"  最佳模型已保存至: {save_path}")
            except Exception as e:
                print(f"  错误：保存模型时出错: {e}")
        else:
            epochs_no_improve += 1
            print(f"  验证损失未改善 ({val_loss:.4f} vs best {best_val_loss:.4f})。连续未改善周期数: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # 检查是否触发早停
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n验证损失连续 {EARLY_STOPPING_PATIENCE} 个周期未改善，触发早停！")
            break # 退出训练循环

    total_training_time = time.time() - start_time
    print("\n--- 训练完成 ---")
    print(f"总耗时: {total_training_time // 60:.0f} 分 {total_training_time % 60:.0f} 秒")
    print(f"最佳验证损失: {best_val_loss:.4f}")

    # --- 最终模型测试逻辑 --- 
    if best_model_state is not None and test_loader is not None:
        print("\n--- 加载最佳模型进行最终测试 ---")
        try:
            model.load_state_dict(best_model_state)
            print("最佳模型状态已加载。")

            # 使用 validate 函数在测试集上评估
            # 注意: test_loader 在 create_dataloaders 中被返回，但之前可能未被使用
            test_loss, test_acc = validate(model, test_loader, criterion, device)
            print("\n--- 最终测试集结果 ---")
            print(f"  测试损失: {test_loss:.4f}")
            print(f"  测试准确率: {test_acc:.4f}")

        except Exception as e:
            print(f"错误：在测试最佳模型时出错: {e}")

    elif test_loader is None:
         print("\n警告：无法加载测试数据加载器，跳过最终测试。")
    else: # best_model_state is None
        print("\n警告：未能保存任何最佳模型状态（可能训练未完成或未改善），跳过最终测试。")

    # --- (注释掉之前的 TODO) ---
    # # --- (TODO: 在这里添加最终模型测试逻辑，加载最佳模型: model.load_state_dict(best_model_state)) ---
    # else:
    #     print("\n未能找到最佳模型状态或测试数据加载器进行测试。") 