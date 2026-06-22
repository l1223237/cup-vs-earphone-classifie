import torch
from torchvision import transforms
from PIL import Image
from models import create_resnet50
import cv2
import numpy as np
import os

# --- 配置 ---
NUM_CLASSES = 2
MODEL_PATH = "checkpoints/best_model_resnet50.pth"
classes = ["cup", "earphone"]

# --- 加载模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet50(num_classes=NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f" 模型已加载到 {device}")

# --- 图像预处理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 识别函数 ---
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prob = torch.softmax(outputs, 1)
    class_idx = predicted.item()
    confidence = prob[0][class_idx].item()
    return classes[class_idx], confidence

# --- 识别并生成带标签条的图片 ---
def predict_and_visualize(image_path, save_path="result.jpg"):
    label, conf = predict(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    h, w = img.shape[:2]

    # 在底部画一个半透明标签条
    bar_height = 60
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 200, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # 在标签条上写识别结果（居中显示）
    text = f"{label}  ({conf:.2%})"
    font_scale = 1.2
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 15
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f" 结果图已保存到：{save_path}")
    return label, conf

# --- 主程序 ---
if __name__ == "__main__":
    img_path = input("请输入图片路径（例如：test.jpg）：").strip()
    try:
        label, conf = predict_and_visualize(img_path, "prediction_demo.jpg")
        print(f" 识别结果：{label}")
        print(f"📊 置信度：{conf:.2%}")
    except Exception as e:
        print(f" 出错：{e}")