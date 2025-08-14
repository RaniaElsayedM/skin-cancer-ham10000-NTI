# Skin Cancer MNIST: HAM10000 - NTI Project

## 📌 Overview
This project is part of the **NTI training program**.  
Our main objective is to detect and classify skin lesions using the **HAM10000** dataset.  
We applied three main computer vision tasks:
1. **Segmentation** to isolate lesion areas.
2. **Object Detection** to locate lesions.
3. **Classification** to identify lesion type.
Early detection of skin cancer can significantly improve patient outcomes, and this project demonstrates how AI can assist dermatologists in diagnosis.

---

## 📂 Dataset
- **Name:** HAM10000
- **Images:** 10,015 dermatoscopic images
- **Classes:** 7 lesion categories :
- Melanocytic nevi (nv)                  6705
- Basal cell carcinoma (bcc)             6682
- Melanoma (mel)                         6678
- Vascular lesions (vasc)                6674
- Dermatofibroma (df)                    6670
- Benign keratosis-like lesions (bkl)    6594
- Actinic keratoses (akiec)              6540
- **Sources:** [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) ,
[HAM10000 Lesion Segmentations](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations) 

---

## 🛠 Workflow
### 0️⃣ Data Preprocessing
- **Input:** RAW dermatoscopic images (HAM10000)
- **Steps:** resizing, normalization, data augmentation (flip/rotate/brightness), train/val/test split

---

### 1️⃣ Segmentation (U-Net)
- **Input:** Original dermatoscopic image
- **Output:** Binary mask of lesion
- **Metric:** **Dice coefficient**, **IoU**

---

### 2️⃣ Object Detection (YOLOv8)
- **Task:** Detect lesion bounding boxes
- **Output:** Image with bounding box + confidence score
- **Metric:** **mAP@0.5**

---

### 3️⃣ Classification (EfficientNet / ResNet)
- **Task:** Classify lesion type (7 classes)
- **Output:** Predicted class + probability
- **Metrics:** **Accuracy**, **F1-score**, **Confusion Matrix**

---

### 4️⃣ Evaluation & Visualization
- **Measure:** accuracy, F1-score, Dice coefficient, mAP.

---

## 📊 Results
| Task            | Model          | Metric           | Score  |
|-----------------|----------------|------------------|--------|
| Segmentation    | U-Net          | Dice Coefficient | 0.93   |
| Detection       | YOLOv8s         | mAP50           | 0.99   |
| Classification  | ResNet50/VGG16 | Accuracy         | 0.96   |

---
## 👥 Team Members
- **Rania Elsayed** — Worked on **Segmentation (U-Net)**
- **Jasmine Mohamed** — Worked on **Object Detection (YOLOv8)**
- **Nesma Nasser** — Worked on **Classification (ResNet50)** 
- **Bassant Elsayed** — Worked on **Classification (VGG16 / MobileNetv2)** 





