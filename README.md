# 🌍 Task 6 — Nationality Detection Model

> **Internship Task** | Machine Learning | 4 Custom CNNs from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raghuveer3339/Nationality_Detection/blob/main/Task6_Nationality_Detection.ipynb)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-UTKFace%20%2B%20FER2013-green)](https://www.kaggle.com)

---

## 📌 Problem Statement

Build a machine learning model that:
- Detects a person's **nationality** from a face image
- Applies **conditional analysis** based on nationality:

| Nationality | Models Applied |
|---|---|
| 🇮🇳 **Indian** | Nationality + Emotion + Age + Dress Colour |
| 🇺🇸 **American** | Nationality + Emotion + Age |
| 🌍 **African** | Nationality + Emotion + Dress Colour |
| 🌐 **Others** | Nationality + Emotion only |

- Includes a **PyQt5 GUI** for local image upload and analysis

---

## 🧠 Approach — 4 Custom CNNs

All models are built **from scratch** using Keras — no pre-trained weights.

```
Input Face Image
       │
       ▼
┌─────────────────────────┐
│  Model 1: Nationality   │  128×128 · 4 classes
│  CNN (4 blocks)         │  American / African / Indian / Others
└────────────┬────────────┘
             │
    ┌────────┴────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌──────────────────┐            ┌─────────────────────┐
│ Model 2: Emotion │            │ Model 3: Age CNN     │
│ CNN (4 blocks)   │            │ (4 blocks)           │
│ FER2013 · 7 cls  │            │ UTKFace · 4 buckets  │
│ Always runs      │            │ Indian + American    │
└──────────────────┘            └─────────────────────┘
                                         │
                                ┌────────┴──────────┐
                                ▼                   ▼
                      ┌──────────────────┐  (African / Indian)
                      │ Model 4: Dress   │
                      │ Colour CNN       │
                      │ (4 blocks)       │
                      │ 8 colour classes │
                      └──────────────────┘
```

---

## 🏗️ Model Architectures

All 4 CNNs share the same base architecture — a reusable `build_cnn()` function:

```
Input (H×W×3)
→ Block 1: Conv2D(32)  + BatchNorm + ReLU + Conv2D(32)  + BatchNorm + ReLU + MaxPool + Dropout
→ Block 2: Conv2D(64)  + BatchNorm + ReLU + Conv2D(64)  + BatchNorm + ReLU + MaxPool + Dropout
→ Block 3: Conv2D(128) + BatchNorm + ReLU + Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout
→ Block 4: Conv2D(256) + BatchNorm + ReLU + Conv2D(256) + BatchNorm + ReLU + MaxPool + Dropout
→ GlobalAveragePooling
→ Dense(256) + Dropout
→ Dense(num_classes, Softmax)
```

### Model Specifications

| Model | Input | Classes | Dataset |
|---|---|---|---|
| Nationality CNN | 128×128×3 | 4 (American/African/Indian/Others) | UTKFace |
| Emotion CNN | 64×64×3 | 7 (Angry/Disgust/Fear/Happy/Neutral/Sad/Surprise) | FER2013 |
| Age CNN | 128×128×3 | 4 (Child/Teen/Adult/Senior) | UTKFace |
| Dress Colour CNN | 64×64×3 | 8 (Red/Blue/Green/Yellow/White/Black/Orange/Purple) | Clothing Colour |

---

## 📊 Datasets

### UTKFace (Nationality + Age)
- 20,000+ face images labelled with age, gender, and race
- Race codes used: `0=White(American)`, `1=Black(African)`, `2=Asian(Others)`, `3=Indian`
- Age buckets: Child (0–12) · Teen (13–19) · Adult (20–59) · Senior (60+)
- Download via Kaggle: `jangedoo/utkface-new`

### FER2013 (Emotion)
- 35,887 grayscale 48×48 face images
- 7 emotion classes: Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise
- Download via Kaggle: `msambare/fer2013`

### Clothing Colour Dataset (Dress Colour)
- 8 colour classes: Red · Blue · Green · Yellow · White · Black · Orange · Purple
- Download via Kaggle: `biaiscience/clothes-color-classification`

---

## 🔬 Training Details

| | Nationality | Emotion | Age | Dress Colour |
|---|---|---|---|---|
| Input size | 128×128 | 64×64 | 128×128 | 64×64 |
| Max/class | 1500 | 1000 | 1500 | 500 |
| Optimizer | Adam 1e-3 | Adam 1e-3 | Adam 1e-3 | Adam 1e-3 |
| Loss | Cat. CE | Cat. CE | Cat. CE | Cat. CE |
| Train/Test | 80/20 | 80/20 | 80/20 | 80/20 |
| Callbacks | EarlyStopping · ReduceLROnPlateau · ModelCheckpoint | ← same for all → | | |

### Data Augmentation
- **Nationality / Age:** rotation, shift, horizontal flip, zoom
- **Emotion:** rotation, shift, horizontal flip, zoom *(subtle — preserves facial features)*
- **Dress Colour:** rotation, shift, horizontal flip, zoom, brightness *(colour-safe)*

---

## 📁 Project Structure

```
Nationality_Detection/
│
├── Task6_Nationality_Detection.ipynb    # Main Colab notebook
├── task6_gui.py                         # PyQt5 local GUI (auto-saved to Drive)
└── README.md
```

**Google Drive folder** (auto-created by notebook):
```
Internship_Datasets/Task6_Nationality_Detection/
├── Datasets/
│   ├── UTKFace/
│   ├── FER2013/
│   └── ClothingColour/
├── Models/
│   ├── nationality_cnn.h5
│   ├── emotion_cnn.h5
│   ├── age_cnn.h5
│   └── dress_colour_cnn.h5
├── Plots/
│   ├── Nationality_CNN_training.png
│   ├── Nationality_CNN_confusion.png
│   ├── Emotion_CNN_training.png
│   ├── Emotion_CNN_confusion.png
│   ├── Age_CNN_training.png
│   ├── Age_CNN_confusion.png
│   ├── Colour_CNN_training.png
│   └── Colour_CNN_confusion.png
└── Outputs/
    ├── sample_utkface.png
    ├── pipeline_test_results.png
    └── model_performance_summary.png
```

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Click **Open in Colab** badge above
2. `Runtime → Change runtime type → T4 GPU`
3. Run cells top to bottom
4. Upload `kaggle.json` when prompted (Cell 4)
5. Mount Google Drive when prompted (Cell 3)

### Local Machine (GUI)
```bash
pip install tensorflow opencv-python PyQt5 numpy
python task6_gui.py
# Set MODEL_DIR inside the script to your .h5 files location
```

---

## 🖥️ GUI Features

**Colab:** Upload any face image → runs full conditional pipeline → displays results inline

**PyQt5 local GUI** (saved to Drive automatically):
- 📂 Upload face image
- ⚡ Detect nationality → apply conditional models
- 📊 Shows: Nationality · Emotion · Age (if applicable) · Dress Colour (if applicable)
- 🎨 Results displayed with confidence scores

---

## 📈 Visual Outputs

The notebook generates and saves to Drive:
- Training accuracy/loss curves for all 4 models
- Confusion matrices for all 4 models
- Pipeline inference results on 8 test images
- Model performance summary bar chart

---

## 📦 Requirements

```
tensorflow>=2.21.0
opencv-python-headless
numpy
scikit-learn
matplotlib
seaborn
kaggle
pillow
PyQt5         # for local GUI only
```

---

## 👤 Author

**Raghuveer** | Internship Project — Task 6
Built with Python · TensorFlow · OpenCV · PyQt5
