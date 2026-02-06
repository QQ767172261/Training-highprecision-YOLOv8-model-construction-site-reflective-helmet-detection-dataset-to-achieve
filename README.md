训练高精度的 YOLOv8 模型工地反光衣安全帽检测数据集，实现对施工人员是否佩戴安全帽、是否穿着反光衣的智能监控 工地安全帽 反光衣的检测识别
@[toc]

**数据集描述：
智慧工地数据集，包括安全帽、反光衣、未戴安全帽、未穿反光衣四类标注标签总共有22453张，有xml和txt两种格式文件**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/39bc59da84954b0587065acff324f1fe.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f0549c1e58b241e5a89dce8d96cc1c5c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7892ef7f90104720ac91b6f5f13a52df.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e78961e20d1843e3a647066d74681f8a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/93e10c39c3dc4ae1a019d0bae65ab6ea.png)
 **智慧工地安全行为检测数据集** 非常适合用于训练高精度的 YOLOv8 模型，实现对施工人员是否佩戴安全帽、是否穿着反光衣的智能监控。

我们将从 **环境搭建 → 数据准备 → 模型训练 → 推理评估 → 完整主函数调用** 全流程指导你，使用 **Ultralytics YOLOv8** 实现：

✅ 安全帽检测（戴 / 未戴）  
✅ 反光衣检测（穿 / 未穿）  
✅ 支持 XML 和 TXT 格式自动处理  
✅ 提供完整可运行代码（含主函数）

---

## 🧰 一、环境搭建（CUDA + Anaconda + Python + 依赖）

### 1. 检查 CUDA 驱动

```bash
nvidia-smi
```

- 确保显示 GPU 和驱动版本（如 535+）
- 推荐 CUDA 11.8 或 12.1

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

---

### 3. 创建虚拟环境

```bash
# 创建环境
conda create -n smart_construction python=3.9

# 激活环境
conda activate smart_construction
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（CUDA 11.8 示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8（Ultralytics）
pip install ultralytics

# 安装 OpenCV、图像处理、XML 解析等
pip install opencv-python numpy matplotlib lxml scikit-learn tqdm pillow
```

✅ 验证安装：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True
```

---

## 📂 二、数据集结构分析

```
smart_site_dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── labels_xml/      # Pascal VOC 格式 .xml
│   ├── img001.xml
│   └── ...
├── labels_yolo/     # YOLO 格式 .txt
│   ├── img001.txt
│   └── ...
```

> ✅ 你有 **22453 张图像**，包含 4 类：
- `helmet`：戴安全帽
- `vest`：穿反光衣
- `no_helmet`：未戴安全帽
- `no_vest`：未穿反光衣

---

## 🔄 三、统一数据格式（推荐使用 YOLO `.txt`）

如果已有 `.txt` 文件，可直接使用。  
如果只有 `.xml`，需转换为 YOLO 格式。

### 3.1 VOC XML → YOLO TXT 转换脚本

```python
# convert_xml_to_yolo.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 类别定义（必须与你的标注一致）
classes = ['helmet', 'vest', 'no_helmet', 'no_vest']
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(xml_file, output_dir, img_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        if size is None: return
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in class_to_id:
                print(f"⚠️ 忽略类别: {cls_name}")
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bb = convert_box((w, h), (xmin, xmax, ymin, ymax))

            lines.append(f"{class_to_id[cls_name]} {' '.join(f'{x:.6f}' for x in bb)}")

        # 保存为 .txt
        txt_file = Path(output_dir) / (Path(xml_file).stem + '.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        print(f"❌ 处理 {xml_file} 失败: {e}")

# 路径配置（请根据实际路径修改）
image_dir = "smart_site_dataset/images"
xml_dir = "smart_site_dataset/labels_xml"
output_dir = "smart_site_dataset/labels"

os.makedirs(output_dir, exist_ok=True)

for xml_file in Path(xml_dir).glob("*.xml"):
    convert_annotation(xml_file, output_dir, image_dir)

print("✅ 所有 XML 已转换为 YOLO 格式！")
```

---

## 🔁 四、划分训练/验证/测试集

```python
# split_dataset.py
import os
from sklearn.model_selection import train_test_split
from glob import glob

all_images = glob("smart_site_dataset/images/*.jpg")
image_names = [os.path.basename(img) for img in all_images]

train_names, test_names = train_test_split(image_names, test_size=0.2, random_state=42)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42)

def write_list(names, path):
    with open(path, 'w') as f:
        f.write('\n'.join(names))

write_list(train_names, "smart_site_dataset/train.txt")
write_list(val_names, "smart_site_dataset/val.txt")
write_list(test_names, "smart_site_dataset/test.txt")

print("✅ 数据集划分完成：train/val/test")
```

---

## 📄 五、创建 `data.yaml`

```yaml
# smart_construction.yaml
train: smart_site_dataset/train.txt
val: smart_site_dataset/val.txt
test: smart_site_dataset/test.txt

nc: 4
names: ['helmet', 'vest', 'no_helmet', 'no_vest']
```

---

## 🚀 六、训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO

def train_model():
    # 加载预训练模型（推荐使用 yolov8m 或 yolov8l）
    model = YOLO('yolov8m.pt')

    # 开始训练
    results = model.train(
        data='smart_construction.yaml',
        epochs=120,
        imgsz=640,           # 可尝试 1280 提升小目标检测
        batch=16,            # 根据显存调整（16G 显存可用 16）
        name='safety_detection',
        project='runs/smart_site',
        device=0,            # 使用 GPU 0
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        patience=15,         # 早停
        augment=True,        # 启用 Mosaic、HSV 增强
        close_mosaic=10,     # 最后10轮关闭 Mosaic
        cache='disk'         # 缓存图像加速训练
    )
    print("✅ 训练完成！")
    return results

if __name__ == "__main__":
    train_model()
```

> 📌 模型保存路径：`runs/smart_site/safety_detection/weights/best.pt`

---

## 🔍 七、推理与预测代码（`infer.py`）

```python
# infer.py
from ultralytics import YOLO
import cv2

def predict_image(model_path, image_path, conf=0.5):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('Safety Detection', im)
        cv2.waitKey(0)
        cv2.imwrite('result.jpg', im)
    return results

if __name__ == "__main__":
    predict_image(
        model_path='runs/smart_site/safety_detection/weights/best.pt',
        image_path='test.jpg'
    )
```

---

## 📊 八、评估代码（`evaluate.py`）

```python
# evaluate.py
from ultralytics import YOLO

def evaluate_model():
    model = YOLO('runs/smart_site/safety_detection/weights/best.pt')

    metrics = model.val(
        data='smart_construction.yaml',
        split='val',
        batch=16,
        imgsz=640,
        device=0,
        save_json=True,
        conf=0.001  # 低置信度评估 mAP
    )

    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")

    return metrics

if __name__ == "__main__":
    evaluate_model()
```

---

## 🏁 九、主函数整合（`main.py`）

```python
# main.py
import os
import sys

def main():
    print("🚀 智慧工地安全检测系统启动...")

    # 步骤1：数据转换（可选）
    # if not os.path.exists("smart_site_dataset/labels"):
    #     print("🔄 正在转换 XML 到 YOLO 格式...")
    #     os.system("python convert_xml_to_yolo.py")

    # 步骤2：划分数据集
    if not os.path.exists("smart_site_dataset/train.txt"):
        print("🔄 正在划分训练/验证集...")
        os.system("python split_dataset.py")

    # 步骤3：训练模型
    print("🏋️ 正在训练模型...")
    os.system("python train.py")

    # 步骤4：评估模型
    print("📊 正在评估模型...")
    os.system("python evaluate.py")

    # 步骤5：推理示例
    print("🔍 正在进行推理测试...")
    os.system("python infer.py")

    print("✅ 全流程完成！")

if __name__ == "__main__":
    main()
```

---

以上文字及代码仅供参考学习使用。
