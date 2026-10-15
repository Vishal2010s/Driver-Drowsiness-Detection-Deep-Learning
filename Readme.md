# 🧠 Driver Drowsiness Detection Using Eye Closure and Yawning Analysis

**Deep Learning • Custom CNN vs MobileNetV2 • Computer Vision • Streamlit**

[![Live Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://driver-drowsiness-detection-deep-learning-yzmrwuurppunh6fxp8ar.streamlit.app/)

> A computer-vision prototype that classifies **eye state** and **yawning** using two independently trained binary classifiers per architecture, compares a Custom CNN against MobileNetV2 transfer learning, and maps predictions to **Alert**, **Mild Fatigue**, or **Severe Fatigue** using a rule-based policy. The deployed Streamlit app includes saved training reports and single-image prediction.

**[▶ Open the live dashboard](https://driver-drowsiness-detection-deep-learning-yzmrwuurppunh6fxp8ar.streamlit.app/)**

## 📌 Contents

- [🧠 Driver Drowsiness Detection Using Eye Closure and Yawning Analysis](#-driver-drowsiness-detection-using-eye-closure-and-yawning-analysis)
  - [📌 Contents](#-contents)
  - [📌 Project Overview](#-project-overview)
  - [💼 Business Use Cases](#-business-use-cases)
  - [🛠 Technology Stack](#-technology-stack)
  - [📂 Dataset and Preprocessing](#-dataset-and-preprocessing)
  - [🏗 Model Architecture](#-model-architecture)
    - [Custom CNN](#custom-cnn)
    - [MobileNetV2 (transfer learning)](#mobilenetv2-transfer-learning)
  - [⚙️ Training Strategy](#️-training-strategy)
  - [📊 Final Evaluation Results](#-final-evaluation-results)
    - [Accuracy by task](#accuracy-by-task)
    - [Notable confusion-matrix findings](#notable-confusion-matrix-findings)
    - [Detailed classification metrics](#detailed-classification-metrics)
  - [🧠 Decision Fusion and Simulated Progression](#-decision-fusion-and-simulated-progression)
    - [Three-stage mapping](#three-stage-mapping)
    - [Constructed fatigue progression](#constructed-fatigue-progression)
  - [🖥 Streamlit Dashboard](#-streamlit-dashboard)
    - [Tab 1 — Training reports](#tab-1--training-reports)
    - [Tab 2 — Prediction](#tab-2--prediction)
  - [📁 Repository Structure](#-repository-structure)
  - [⚙️ Installation and Deployment](#️-installation-and-deployment)
    - [Option 1: Run on Google Colab (Recommended)](#option-1-run-on-google-colab-recommended)
    - [Option 2: Run Locally](#option-2-run-locally)
    - [Deploy with Streamlit Community Cloud](#deploy-with-streamlit-community-cloud)
    - [Retraining (optional)](#retraining-optional)
  - [⚠️ Limitations and Next Steps](#️-limitations-and-next-steps)
  - [🏁 Conclusion](#-conclusion)
  - [👨‍💻 Author](#-author)

## 📌 Project Overview

Driver fatigue can reduce alertness and impair reaction time. This project studies two visual cues—**eye closure** (`Closed` / `Open`) and **yawning** (`no_yawn` / `yawn`)—as image-classification tasks.

Rather than training one four-class classifier, the finalized pipeline trains **four separate binary models**:

| Visual task | Custom CNN | MobileNetV2 |
|---|---|---|
| Eye state | `eyes_cnn` | `eyes_mobilenet` |
| Yawning | `yawn_cnn` | `yawn_mobilenet` |

Each model produces a two-class Softmax output. The reporting workflow compares both architectures; the deployed **Prediction** tab loads the two **MobileNetV2** models.

## 💼 Business Use Cases

Potential applications, subject to additional validation and engineering, include driver-monitoring research, fleet-safety dashboards, ADAS research, and analysis of visual alertness indicators. **This educational prototype is not a certified real-time driving-safety system.** Its test metrics must not be interpreted as demonstrated road-safety effectiveness.

## 🛠 Technology Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Transfer learning | MobileNetV2, ImageNet weights |
| Image processing | Pillow; OpenCV Haar-cascade face/eye detection in the app |
| Data analysis | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn; Streamlit |
| Training environment | Google Colab or a compatible local Python environment |
| Deployment | GitHub and Streamlit Community Cloud |

## 📂 Dataset and Preprocessing

The source data contains four labeled image groups: `Closed`, `Open`, `no_yawn`, and `yawn`. The master pipeline validates images and checks for **exact duplicate file contents** using SHA-256 before splitting. It writes a split manifest and class counts for traceability.

The final saved training reports show:

| Task | Training images | Validation images | Test images |
|---|---:|---:|---:|
| Eyes (`Closed`, `Open`) | 1,163 | 144 | 148 |
| Yawn (`no_yawn`, `yawn`) | 1,164 | 144 | 148 |

These task totals come from the same four-class source, not two disjoint datasets; **2,911 source images** are represented overall. A seeded, approximately **80% / 10% / 10%** per-class split is used. The training code records `subject_disjoint: false`, so performance on unseen individuals is **not established**.

**Preprocessing**

- Resize to `224 × 224 × 3` RGB with **nearest-neighbor interpolation**.
- Custom CNN: divide image values by 255 (`[0, 1]`).
- MobileNetV2: use `mobilenet_v2.preprocess_input` (approximately `[-1, 1]`).
- Preserve the stored image orientation without automatic EXIF-based rotation, consistent with the master pipeline.

**Training augmentation**

| Transformation | Custom CNN | MobileNetV2 |
|---|---|---|
| Rotation | ±10° | ±20° |
| Zoom | 10% | 20% |
| Brightness factor | 0.9–1.1 | 0.8–1.2 |
| Horizontal flip | Enabled | Enabled |

**Dataset link retained from the original project README:** [Google Drive dataset](https://drive.google.com/file/d/1Rj0EYgDy0sy7jc7-KXvQiHwpqA0Dzj__/view?usp=sharing). Access and reuse permissions should be checked before redistribution.

After downloading:
1. Unzip the archive
2. Place the resulting folder in your Google Drive (e.g., `/content/drive/MyDrive/driver_drowsiness/archive/`)
3. The notebook will auto-detect class names and split into train/val/test


## 🏗 Model Architecture

### Custom CNN

```text
RGB input (224 × 224 × 3)
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
Dense(256, ReLU) → Dropout(0.30)
   ↓
Dense(128, ReLU) → Dropout(0.20)
   ↓
Dense(2, Softmax)
```

A separate instance is trained for eyes and for yawning, using categorical cross-entropy.

### MobileNetV2 (transfer learning)

```text
RGB input (224 × 224 × 3)
   ↓
MobileNetV2 backbone (ImageNet-pretrained)
   ↓
GlobalAveragePooling2D
   ↓
Dense(256, ReLU) → Dropout(0.50)
   ↓
Dense(128, ReLU) → Dropout(0.30)
   ↓
Dense(2, Softmax)
```

The backbone is first frozen while the head is trained. An optional fine-tuning stage unfreezes the final 30 backbone layers, keeping BatchNormalization layers frozen. The **baseline vs fine-tuned model is selected using validation loss**, not test accuracy.

## ⚙️ Training Strategy

| Setting | Final master configuration |
|---|---|
| Random seed | `42` |
| Image size | `224 × 224` |
| Batch size | `32` |
| Main training maximum | `30` epochs, with early stopping |
| Initial optimizer / learning rate | Adam / `1e-4` |
| Loss | Categorical cross-entropy |
| Fine-tuning maximum | `15` additional epochs for MobileNetV2 |
| Fine-tuning learning rate | `1e-5` |
| Checkpoint selection | Lowest validation loss |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

The master pipeline exports model checkpoints, training curves, confusion matrices, classification reports, prediction files, comparison tables, provenance/configuration metadata, and example images for model-loading checks. The dashboard reads these saved artifacts; it **does not retrain models**.

## 📊 Final Evaluation Results

Results below are taken from the finalized deployed training-report screenshots, **not** the older four-class evaluation in the original README.

### Accuracy by task

| Model | Eyes (148 tests) | Yawn (148 tests) | Sample-weighted accuracy (296 tests) |
|---|---:|---:|---:|
| **Custom CNN** | 74/148 = **50.00%** | 128/148 = **86.49%** | 202/296 = **68.24%** |
| **MobileNetV2** | 144/148 = **97.30%** | 132/148 = **89.19%** | 276/296 = **93.24%** |

**Interpretation:** MobileNetV2 achieved a **25.00 percentage-point** higher sample-weighted test accuracy than the Custom CNN across the two separately evaluated tasks. This aggregate is **not** a measured end-to-end accuracy for three-stage fatigue classification.

### Notable confusion-matrix findings

- **Eyes / Custom CNN:** All 148 test images were predicted as `Closed` (74 correct `Closed`, 74 incorrect `Open`), revealing a failure to distinguish eye states.
- **Eyes / MobileNetV2:** 71/74 `Closed` and 73/74 `Open` images were classified correctly.
- **Yawn / Custom CNN:** 65/74 `no_yawn` and 63/74 `yawn` images were classified correctly.
- **Yawn / MobileNetV2:** 73/74 `no_yawn` and 59/74 `yawn` images were classified correctly. The **yawn recall is lower** for MobileNetV2 despite its higher overall yawn accuracy.

### Detailed classification metrics

| Task / model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Eyes / CNN | Closed | 0.500 | 1.000 | 0.667 |
| Eyes / CNN | Open | 0.000 | 0.000 | 0.000 |
| Eyes / MobileNetV2 | Closed | 0.986 | 0.959 | 0.973 |
| Eyes / MobileNetV2 | Open | 0.961 | 0.986 | 0.973 |
| Yawn / CNN | no_yawn | 0.855 | 0.878 | 0.867 |
| Yawn / CNN | yawn | 0.875 | 0.851 | 0.863 |
| Yawn / MobileNetV2 | no_yawn | 0.830 | 0.986 | 0.901 |
| Yawn / MobileNetV2 | yawn | 0.983 | 0.797 | 0.881 |

The dashboard also displays training/validation curves and confusion matrices for each task and model.

## 🧠 Decision Fusion and Simulated Progression

### Three-stage mapping

For the **saved test-set report**, individual predicted class labels are mapped as follows:

| Predicted visual class | Score | Reported stage |
|---|---:|---|
| `Open` or `no_yawn` | 0 | Alert |
| `yawn` | 1 | Mild Fatigue |
| `Closed` | 2 | Severe Fatigue |

For **one uploaded portrait**, the Streamlit prediction workflow combines the two task outputs:

1. If eyes are classified `Closed` → **Severe Fatigue**.
2. If eyes are `Open` and yawning is `yawn` → **Mild Fatigue**.
3. If eyes are `Open` and yawning is `no_yawn` → **Alert**.
4. Otherwise → **Uncertain**.

These labels are **heuristic indicators from a single image**, not a clinical or validated real-time measure of sustained drowsiness.

### Constructed fatigue progression

The training report includes CNN and MobileNetV2 fatigue-progression plots. The master script **reorders saved test predictions by fatigue category** and groups them into **10 observations per interval** to construct illustrative progression curves. They are **not measurements from a real driving session**, and the x-axis intervals should not be understood as observed elapsed driving time.

## 🖥 Streamlit Dashboard

**Live demo:** https://driver-drowsiness-detection-deep-learning-yzmrwuurppunh6fxp8ar.streamlit.app/

### Tab 1 — Training reports

Displays:

- Dataset and preprocessing summary.
- Side-by-side CNN and MobileNetV2 training curves for `eyes` or `yawn`.
- Confusion matrices, precision, recall, and F1-scores.
- Decision-summary tables and **constructed** fatigue-progression plots.
- Sample-weighted test accuracy comparison and limitations.
- Optional robustness plots **only if corresponding artifacts exist**.

### Tab 2 — Prediction

The current `app.py` supports **uploaded portrait images** (`jpg`, `jpeg`, `png`, `webp`), not an active continuous webcam stream.

1. Upload a driver portrait.
2. The app tries an OpenCV Haar-cascade face/eye detection workflow and uses a fallback eye-crop region when detection is unsuccessful.
3. The cropped eye region is evaluated by the **eyes MobileNetV2 model**; the full portrait is evaluated by the **yawn MobileNetV2 model**.
4. Models apply the saved `224 × 224`, RGB, nearest-neighbor and MobileNet preprocessing contract.
5. Predictions with a top-class score below **70%** are labeled `Uncertain`.
6. The app shows eye state, yawning state, the rule-based fatigue stage, and both model confidence scores.

**Confidence scores are uncalibrated model outputs, not guaranteed probabilities of correctness.**

**Artifact integrity:** The app uses `metadata.json` to check model filenames, class order, shapes, model SHA-256 fingerprints, and inference results on saved example images before accepting models for prediction.

## 📁 Repository Structure

The following matches the visible GitHub layout and the uploaded bundle listing. Files omitted from the compact listing may also be present in individual per-model directories.

```text
Driver-Drowsiness-Detection/
├── docs/
├── src/
│   ├── Driver_Drowsiness_Detection_raw.ipynb
│   ├── master.py.py
│   └── master_final.ipynb
├── trained_bundle/
│   ├── eyes_cnn/
│   ├── eyes_mobilenet/
│   ├── yawn_cnn/
│   ├── yawn_mobilenet/
│   ├── best_eyes_cnn.keras
│   ├── best_eyes_mobilenet.keras
│   ├── best_yawn_cnn.keras
│   ├── best_yawn_mobilenet.keras
│   ├── fine_eyes_mobilenet.keras
│   ├── fine_yawn_mobilenet.keras
│   ├── metadata.json
│   ├── run_config.json
│   ├── environment.txt
│   ├── preprocessing_summary.csv
│   ├── model_comparison.csv
│   ├── model_accuracy_comparison.csv
│   ├── model_accuracy_comparison.png
│   ├── performance_by_class.csv
│   ├── performance_error_cases.csv
│   ├── decision_fusion_cnn.csv
│   ├── decision_fusion_mobilenet.csv
│   ├── decision_summary_cnn.csv
│   ├── decision_summary_mobilenet.csv
│   ├── fatigue_session_cnn.csv
│   ├── fatigue_session_mobilenet.csv
│   ├── fatigue_intervals_cnn.csv
│   ├── fatigue_intervals_mobilenet.csv
│   ├── fatigue_transitions_cnn.csv
│   ├── fatigue_transitions_mobilenet.csv
│   ├── fatigue_progression_cnn.png
│   ├── fatigue_progression_mobilenet.png
│   ├── eyes_cnn_example.png
│   ├── eyes_mobilenet_example.png
│   ├── yawn_cnn_example.png
│   ├── yawn_mobilenet_example.png
│   └── ... additional saved evaluation plots and metadata
├── app.py
├── requirements.txt
└── Readme.md
```

The per-model folders contain report artifacts such as `training_curves.png`, `classification_report.csv`, and `confusion_matrix.png`, which the dashboard references directly. The checked-in trained bundle is a deployment/reporting artifact; full training image splits do not need to be in the web app repository.

## ⚙️ Installation and Deployment

### Option 1: Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `src/Driver_Drowsiness_Detection_raw.ipynb`
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells (`Ctrl+F9`)

No local installation needed — all dependencies are pre-installed in Colab.

### Option 2: Run Locally

```bash
# Clone using the exact URL of your repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The dashboard expects `trained_bundle/` beside `app.py`. By default it loads that folder automatically; an alternate extracted bundle path can be supplied in the sidebar.

### Deploy with Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and all **required** `trained_bundle/` artifacts to GitHub.
2. Select the repository and deployment branch in Streamlit Community Cloud.
3. Set the main app file to **`app.py`**.
4. Confirm model and sample-image files referenced in `metadata.json` are present and that the saved hashes match.
5. Verify both **Training reports** and **Prediction** tabs after deployment.

The deployed inference workflow loads **two MobileNetV2 `.keras` checkpoints**, while report visualizations read the saved outputs of all four trained models. Installing compatible TensorFlow/Keras and OpenCV packages is important for Streamlit Cloud.

### Retraining (optional)

The `src/` training code expects a four-class source-image directory and writes an output run directory. In the provided master script, source and output paths default to Google Drive paths in Colab or local `dataset/` and `drowsiness_outputs/` paths outside Colab; configure these before running. Re-training can take substantial time and may require GPU resources. Generated results must be repackaged into `trained_bundle/` and kept consistent with `metadata.json` before redeployment.

## ⚠️ Limitations and Next Steps

- **No subject-independent evaluation:** the recorded split does not demonstrate generalization to unseen people or recording sessions.
- **Single-image predictions:** an isolated frame cannot measure sustained drowsiness or eye-closure duration (such as PERCLOS).
- **No live driving-session validation:** progression plots are constructed from test predictions, not sequential driving footage.
- **Crop and domain sensitivity:** Haar-cascade detection, fallback eye crops, lighting, eyewear, orientation and motion can affect performance.
- **CNN eye-model failure:** the eye CNN predicted only `Closed` on the held-out test set; additional debugging and validation are warranted.
- **Yawn recall trade-off:** MobileNetV2 missed 15 out of 74 true `yawn` test images.
- **Confidence calibration:** the 70% threshold is a demonstration setting, not a statistically validated operating point.

Future work includes subject-disjoint test design, temporal/video modeling, improved face/eye localization, probability calibration, robustness evaluation, and testing under realistic driving conditions.

## 🏁 Conclusion

The final project implements four binary deep learning models across two facial tasks, a consolidated training/evaluation pipeline, and a Streamlit dashboard for result exploration and image-based inference. MobileNetV2 achieved **93.24% sample-weighted accuracy across the separate eye and yawn test sets**, compared with **68.24%** for the Custom CNN. These scores do not evaluate the accuracy of the final three-stage fatigue mapping. The project demonstrates end-to-end preprocessing, training, transfer learning, evaluation, artifact verification, and cloud deployment while documenting the limits of image-based fatigue inference.

## 👨‍💻 Author

**S. Vishal**  
Artificial Intelligence | Machine Learning | Computer Vision

---

*If this project is useful, consider starring the GitHub repository.*
