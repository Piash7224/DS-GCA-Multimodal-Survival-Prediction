
# DS-GCA: Multimodal Survival Prediction with Gated Cross-Attention

A multimodal deep learning framework for breast cancer survival prediction by fusing histopathological visual features and clinical data through gated cross-attention mechanisms.

---

## 🚀 Key Highlights

- Multimodal fusion of histopathology images + clinical data  
- Dual-Stream Gated Cross-Attention (DS-GCA) architecture  
- Achieved **C-Index ~0.74** on TCGA-BRCA cohort  
- Consistently outperforms single-modality and early fusion baselines  
- Includes ablation study and gating behavior analysis  

---

## 🧠 Why This Project Matters

Accurate survival prediction is critical for:
- Patient risk stratification  
- Personalized treatment planning  
- Clinical decision support systems  

This project explores how multimodal deep learning can improve predictive performance beyond traditional single-modality approaches.

---

## 📌 Overview

DS-GCA integrates:

- **Visual Branch**: Swin Transformer-based Attention-based Multiple Instance Learning (ABMIL)  
- **Clinical Branch**: Dense clinical metadata encoder with cross-validation  
- **Fusion Module**: Dual-stream Gated Cross-Attention (GCA)  

---

## ⚠️ Research Note

This repository contains the implementation of ongoing research work.  
Detailed experimental analysis and formal validation will be presented in a forthcoming publication.

---

## 🗂️ Project Structure

```text
DS-GCA/
├── README.md
├── requirements.txt
├── main.py
│
├── data_preprocessing/
│   └── prepare_bcss.py
│
├── models/
│   ├── swin_training.py
│   ├── swin_inference.py
│   ├── visual_embedding.py
│   ├── clinical_encoder.py
│   └── ds_gca_fusion.py
│
├── analysis/
│   ├── comparative_km_eval.py
│   ├── gating_behavior.py
│   ├── ablation_study.py
````

---

## ⚙️ Key Components

### Data Preparation

* Converts BCSS into a 4-class patch dataset
* Patient-level splits (70/15/15)
* Class balancing and priority-based labeling

### Swin Transformer (Visual Branch)

* Swin-S architecture with Focal Loss
* MixUp/CutMix augmentation
* Weighted sampling for class imbalance

### Visual Embedding

* MIL-based patch aggregation
* Patient-level feature generation (N×512)
* Memory-efficient processing

### Clinical Encoder

* MLP with BatchNorm + GELU
* Cross-validation for robustness
* Cox loss for survival modeling

### DS-GCA Fusion

* Dual-stream cross-attention
* Learnable gating for modality weighting
* Robust multimodal feature interaction

---

## 🧩 Architecture

```text
Visual Branch:                    Clinical Branch:
  Patch Bag →                       Clinical Data →
  ABMIL Aggregation                 Dense Encoder →
  Projection (512→64)               Projection (128→64)
       ↓                                 ↓
       └── Gated Cross-Attention ──────┘
                     ↓
           Concatenate (128)
                     ↓
           Classifier → Risk Score
```

---

## 📊 Results

### Performance

* **Swin Classifier**: ~84% accuracy (4-class TME)
* **DS-GCA Model**: ~0.74 C-Index

### Outputs

* Kaplan-Meier survival curves
* Statistical reports (CSV)
* Confusion matrices
* Gating behavior visualizations

---

## 🧪 Analysis

### Kaplan-Meier Evaluation

* Risk stratification (median & quartile split)
* Log-rank statistical testing

### Gating Behavior

* Modality importance analysis
* Gate activation diagnostics

### Ablation Study

* Clinical-only
* Visual-only
* Early fusion
* GCA variants
* Full DS-GCA

---

## ⚡ Setup & Installation

### Clone Repository

```bash
git clone https://github.com/Piash7224/DS-GCA-Multimodal-Survival-Prediction.git
cd DS-GCA-Multimodal-Survival-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Full Pipeline

```bash
python main.py --stage all
```

### Run Individual Stages

```bash
python main.py --stage 1  # Data Preparation
python main.py --stage 2  # Swin Training
python main.py --stage 3  # Visual Embedding
python main.py --stage 4  # Clinical Encoder
python main.py --stage 5  # DS-GCA Fusion
```

---

## 🔧 Configuration

* Patch Size: 224×224
* Visual Embedding: 512-d
* Clinical Embedding: 128-d
* Fusion Dimension: 64-d
* Learning Rate: 1e-4
* Epochs: 100
* Cross-validation: 5-fold

---

## 📦 Dependencies

* PyTorch
* torchvision
* timm
* scikit-learn
* lifelines
* pandas
* numpy
* matplotlib
* seaborn

---

## 📄 License

MIT License

---

## 📬 Contact

**Mohammad Mahmud Hasan**
Email: [piashmahmud204@gmail.com]

```


