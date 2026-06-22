import os
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

def count_images_in_folder(folder_path):
    """统计文件夹内所有图片的数量"""
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    return count

def analyze_dataset(data_dir="my-data"):
    """分析数据集：统计train和val中各类别的图片数量"""
    results = {}
    
    for split in ["train", "val"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"⚠️ 目录不存在: {split_path}")
            continue
        
        # 获取所有类别文件夹
        classes = [d for d in os.listdir(split_path) 
                   if os.path.isdir(os.path.join(split_path, d))]
        
        results[split] = {}
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            count = count_images_in_folder(cls_path)
            results[split][cls] = count
            print(f"{split}/{cls}: {count} 张图片")
    
    return results

def plot_distribution(results):
    """绘制类别分布柱状图"""
    # 合并train和val的数据
    all_classes = set()
    for split_data in results.values():
        all_classes.update(split_data.keys())
    
    class_names = sorted(list(all_classes))
    train_counts = [results.get("train", {}).get(cls, 0) for cls in class_names]
    val_counts = [results.get("val", {}).get(cls, 0) for cls in class_names]
    
    # 绘图
    x = range(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], train_counts, width, label='训练集 (train)', color='steelblue')
    ax.bar([i + width/2 for i in x], val_counts, width, label='验证集 (val)', color='coral')
    
    ax.set_xlabel('类别')
    ax.set_ylabel('图片数量')
    ax.set_title('数据集类别分布')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    # 添加数值标签
    for i, (t, v) in enumerate(zip(train_counts, val_counts)):
        if t > 0:
            ax.text(i - width/2, t + 0.5, str(t), ha='center', va='bottom', fontsize=10)
        if v > 0:
            ax.text(i + width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print(" 柱状图已保存为: class_distribution.png")
    plt.show()

if __name__ == "__main__":
    results = analyze_dataset("my-data")
    if results:
        plot_distribution(results)
    else:
        print(" 未找到有效数据，请检查 my-data 文件夹结构是否正确。")