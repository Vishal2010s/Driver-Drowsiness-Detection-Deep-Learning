# 🧠 Driver Drowsiness Detection using Eye Closure and Yawning Analysis with Deep Learning

> A Computer Vision based and Deep Learning system that detects early signs of driver fatigue using facial features (eye closure and yawning) by comparing a Custom CNN with MobileNetV2 Transfer Learning.The system classifies driver state into 3 fatigue levels — Alert, Mild Fatigue, Severe Fatigue — and tracks fatigue progression over time.

---
## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Use Cases](#-business-use-cases)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Workflow (11 Steps)](#-project-workflow-11-steps)
- [Conclusion](#-Conclusion)

---

## 📌 Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide. Reduced alertness, slower reaction time, and impaired decision-making significantly increase accident risk.

This project proposes a **non-intrusive vision-based Driver Monitoring System (DMS)** that detects fatigue by analyzing two physiological indicators:

- 👁 **Eye closure** (eyes open vs. closed)
- 🥱 **Yawning behavior** (yawn vs. no yawn)

The system classifies facial images into four categories and further converts these predictions into three fatigue levels using a rule-based decision fusion algorithm.

---
## 💼 Business Use Cases

| # | Use Case | Description |
|---|---|---|
| 1 | **Accident Prevention** | Real-time fatigue monitoring + timely warnings to reduce drowsy-driving accidents |
| 2 | **Fleet Management** | Logistics companies can monitor driver fatigue for compliance, reduce damage/claims |
| 3 | **ADAS Integration** | Embeddable in Advanced Driver Assistance Systems for real-time alerts |
| 4 | **Insurance Risk Assessment** | Insurance providers can assess driver risk profiles more accurately |
| 5 | **Autonomous Vehicle Support** | Safety layer for monitoring human readiness during manual-to-autonomous transitions |

---

# 🛠 Technologies Used

| Category | Tools |
|-----------|-------|
| Programming Language | Python 3.10+ |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Image Processing | PIL |
| Data Manipulation | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning Metrics | Scikit-learn |
| Development Platform | Google Colab (T4 GPU recommended) |

---
## ⚙️ Installation

### Option 1: Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `src/Driver_Drowsiness_Detection_raw.ipynb`
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells (`Ctrl+F9`)

No local installation needed — all dependencies are pre-installed in Colab.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

# 📂 Dataset

The dataset consists of real human facial images focusing on eye and mouth regions.

| Property | Value |
|---|---|
| Total images | ~2911|
| Classes | 4 (Eyes Open, Eyes Closed, Yawn, No Yawn) |
| Image size | Variable (resized to 224×224) |
| Source | MRL Eye Dataset + custom yawn/no-yawn images |

### Download

📂 [Download Dataset (Google Drive)](https://drive.google.com/file/d/1Rj0EYgDy0sy7jc7-KXvQiHwpqA0Dzj__/view?usp=sharing)

After downloading:
1. Unzip the archive
2. Place the resulting folder in your Google Drive (e.g., `/content/drive/MyDrive/driver_drowsiness/archive/`)
3. The notebook will auto-detect class names and split into train/val/test

### Split Strategy

| Split | Ratio | Purpose |
|---|---|---|
| train_split | 80% | Model training |
| validation | 10% | Model validation |
| test | 10% | Final evaluation |

The notebook handles splitting automatically using stratified sampling.

The dataset is organized into:

```text
dataset/
│
├── train_split/
├── validation/
└── test/
```

---

# 🔄 Project Workflow

```text
Dataset Collection
        │
        ▼
Dataset Organization
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
Custom CNN
        │
        ├─────────────┐
        ▼             ▼
MobileNetV2      Model Training
        │             │
        └──────► Evaluation
                     │
                     ▼
             Decision Fusion Logic
                     │
                     ▼
       Driver Fatigue Progression Curve
                     │
                     ▼
          Performance Comparison
```

---


---

# 📁 Repository Structure

```text
Driver-Drowsiness-Detection/

├── src/
│     Driver_Drowsiness_Detection_raw.ipynb
│
├── outputs/
│     step4_preprocessing/
│     Step 7_train/
│     step9_decision_fusion/
│     step11_performance_analysis/
│     
├── models/
│     best_custom_cnn.keras
│     best_mobilenetv2.keras
│
├── images/
|      CNN_Drive fatigue progesion curve.png/
|      Mobilenet_Drive fatigue progesion curve/
|      Model_Test Accuracy coparision
|
├── docs/
|     Driver_Drowsiness_Detection_Project_Report        
│
├── requirements.txt
│
├── LICENSE
│
└── README.md
```
---
## 🗺️ Project Workflow (11 Steps)

| Step | Section | What It Does |
|---|---|---|
| 1 | Problem Understanding | Identify driver drowsiness as a road safety issue |
| 2 | Dataset Collection | Collect facial images for 4 classes |
| 3 | Dataset Organization | Split into train/val/test |
| 4 | Data Preprocessing | Resize 224×224, normalize, augment |
| **5** | **Model Selection** | **Custom CNN vs MobileNetV2** |
| **6** | **Model Development** | **Eye State + Mouth State interpretation** |
| **7** | **Model Training** | **Frozen base + Adam, 2-stage** |
| **8** | **Model Evaluation** | **Accuracy, confusion matrix, precision/recall** |
| **9** | **3-Level Fatigue Fusion** | **4-class → Alert/Mild/Severe** |
| **10** | **Fatigue Progression Curve** | **Time-series simulation** |
| **11** | **Performance Analysis** | **Limitations, error cases, per-class** |

---

# 🏗️ Model Architecture

This project implements and compares two deep learning architectures for driver drowsiness detection:

1. **Custom Convolutional Neural Network (CNN)**
2. **MobileNetV2 Transfer Learning**

Both models classify facial images into four categories:

- 👁️ Open
- 😴 Closed
- 🙂 no_yawn
- 🥱 yawn

The predictions are then converted into three fatigue levels using a rule-based Decision Fusion module.

---

## 1️⃣ Custom CNN Architecture

The Custom CNN was designed from scratch to learn discriminative facial features directly from the dataset.

### Architecture

```text
Input (224×224×3)
        ↓
Conv2D(32) → BatchNorm → MaxPool
        ↓
Conv2D(64) → BatchNorm → MaxPool
        ↓
Conv2D(128) → BatchNorm → MaxPool
        ↓
Conv2D(256) → BatchNorm → MaxPool
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, ReLU)
        ↓
Dropout(0.30)
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.20)
        ↓
Dense(4, Softmax)
```

- **Trainable Parameters:** ~488K

---

## 2️⃣ MobileNetV2 Architecture (Transfer Learning)

The second model uses **MobileNetV2 pretrained on ImageNet** for feature extraction.

Only the custom classification head is trained while keeping the convolutional backbone frozen.

### Architecture

## 2. MobileNetV2 (Transfer Learning)

```text
Input (224×224×3)
        ↓
MobileNetV2 (ImageNet, Frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, ReLU)
        ↓
Dropout(0.50)
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.30)
        ↓
Dense(4, Softmax)
```

- **Trainable Parameters:** Classification Head Only

---

# ⚙️ Training Strategy

## Model Training

Both the Custom CNN and MobileNetV2 models were trained using a common reusable training pipeline with TensorFlow/Keras callbacks for efficient learning and model checkpointing.

### Training Configuration

Both models were trained using a common training pipeline with reusable functions and identical optimization settings to ensure a fair performance comparison.

| Configuration | Custom CNN | MobileNetV2 |
|--------------|------------|-------------|
| Model Type | Custom CNN | Transfer Learning |
| Pretrained Weights | Not Applicable | ImageNet |
| Base Network | Built from scratch | MobileNetV2 |
| Base Model Frozen | Not Applicable | Yes |
| Input Image Size | 224 × 224 × 3 | 224 × 224 × 3 |
| Batch Size | 32 | 32 |
| Optimizer | Adam | Adam |
| Initial Learning Rate | 1e-4 | 1e-4 |
| Loss Function | Categorical Crossentropy | Categorical Crossentropy |
| Evaluation Metric | Accuracy | Accuracy |
| Epochs | 20 | 20 |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Training Strategy | Train entire network from scratch | Freeze MobileNetV2 feature extractor and train custom classifier head |
| Model Checkpoint | Save best validation model | Save best validation model |

### Callbacks Used

| Callback | Purpose |
|----------|---------|
| **✅ EarlyStopping** | Stops training when validation performance no longer improves to prevent overfitting. |
| **✅ ReduceLROnPlateau** | Reduces the learning rate when validation loss plateaus, helping the optimizer converge more effectively. |
| **✅ ModelCheckpoint** | Saves the best-performing model based on validation accuracy for later evaluation. |


### Output

During training, the following metrics were monitored for both models:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Learning Rate

The best-performing model weights were saved as:

- `best_custom_cnn.keras`
- `best_mobilenetv2.keras`

---

# 🔍 Data Augmentation

Training images were augmented to improve generalization.

Applied transformations include:

- Rotation
- Zoom
- Brightness Adjustment
- Horizontal Flip

### Custom CNN

- Rescale pixel values to **[0, 1]**

### MobileNetV2

- `preprocess_input()`
- Pixel values normalized to **[-1, 1]**

---

# 📊 Output Classes

Both models predict one of the following four facial states:

| Class | Description |
|--------|-------------|
| Open | Eyes Open |
| Closed | Eyes Closed |
| no_yawn | No Yawning |
| yawn | Yawning |

---

# 🧠 Decision Fusion

The four predicted facial states are converted into three fatigue stages.

| Predicted Class | Fatigue Score | Fatigue Stage |
|-----------------|--------------:|---------------|
| Open | 0 | Alert |
| no_yawn | 0 | Alert |
| yawn | 1 | Mild Fatigue |
| Closed | 2 | Severe Fatigue |

---
## 📊 Model Performance Comparison

| Metric | Custom CNN | MobileNetV2 |
|:-------------------|-----------:|------------:|
| **Test Accuracy (%)** | **69.3878** | **88.0952** |
| Closed Precision | 0.6897 | 1.0000 |
| Open Precision | 0.7833 | 0.9867 |
| No_yawn Precision | 0.7018 | 0.7241 |
| Yawn Precision | 0.6333 | 0.8305 |


# 📊 Model Comparison

| Metric | Custom CNN | MobileNetV2 |
|----------|-----------|-------------|
| Test Accuracy | Higher than baseline | Best Performing |
| Transfer Learning | No | Yes |
| Parameters | Fewer | More |
| Training Speed | Faster | Moderate |
| Generalization | Moderate | Excellent |

---
> **Observation:** MobileNetV2 consistently outperformed the Custom CNN across all evaluation metrics, achieving approximately **18.7% higher test accuracy** and superior precision for every class.

---
# 🏆 Conclusion

This project presents an end-to-end Driver Drowsiness Detection system using Deep Learning and Computer Vision. Two architectures—a Custom CNN and a MobileNetV2 Transfer Learning model—were developed and evaluated for classifying eye and mouth states.

Experimental results demonstrate that MobileNetV2 significantly outperforms the Custom CNN, achieving **88.09%** test accuracy while providing superior precision, generalization, and robustness. The addition of a rule-based Decision Fusion module further enhances the system by converting image-level predictions into meaningful fatigue stages (Alert, Mild Fatigue, Severe Fatigue).

The proposed approach is suitable for integration into Driver Monitoring Systems (DMS), Advanced Driver Assistance Systems (ADAS), fleet safety solutions, and smart mobility applications.

---


# 👨‍💻 Author

**S. Vishal**

Artificial Intelligence | Machine Learning | Computer Vision

---

# ⭐ If you found this project useful...

Please consider giving the repository a ⭐ on GitHub.
