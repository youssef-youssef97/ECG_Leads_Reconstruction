# ECG Leads Reconstruction

This repository provides the implementation of a deep learning pipeline for reconstructing missing ECG leads from a reduced set of input leads.

---

## Overview

The goal of this project is to reconstruct full 12-lead ECG signals from a limited number of input leads (I, II, V2) using a hybrid approach that combines:

* Contrastive representations
* Clean signals

The method is designed to improve generalization across datasets and preserve clinically relevant morphological features (e.g., QRS complex).

---

## Method Pipeline

The overall pipeline consists of four main stages:

1. **Data Cleaning**
   Raw ECG signals are filtered and standardized.

2. **Segmentation**
   Signals are segmented into fixed-length windows.

3. **Contrastive Representation Learning**
   A representation encoder is trained to learn meaningful latent embeddings.

4. **ECG Reconstruction**
   A reconstruction model predicts missing leads from selected input leads.

---

## 📂 Project Structure

```
.
├── Cleaning_Two_Datasets.py
├── Segmentation.py
├── contrastive.py
├── extract_contrastive_reps_ptbxl.py
├── extract_contrastive_reps_ptb.py
├── Final_Model.py
├── KNN_plots.py
├── cross_dataset_testing.py
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## Usage


### 1. Data Cleaning

```bash
python Cleaning_Two_Datasets.py --ptbxl --ptb
```

### 2. Segmentation

```bash
python Segmentation.py
```

### 3. Contrastive Training

```bash
python contrastive.py \
--dataset ptbxl \
--splits train \
--out ./contrastive_out \
--epochs 60 \
--batch_classes 12 \
--batch_samples 6 \
--device cuda \
--label-threshold 80
```

### 4. Extract Representations

```bash
python extract_contrastive_reps_ptb.py \
--checkpoint ./contrastive_out/best_encoder.pt \
--out ./out/contrastive_reps_ptb \
--batch 256 \
--device cuda \
--leads 0 1 7
```
```bash
python extract_contrastive_reps_ptbxl.py \
--checkpoint ./contrastive_out/best_encoder.pt \
--out ./out/contrastive_reps \
--batch 256 \
--device cuda
```
### 5. Representation Analysis

```bash
python KNN_plots.py
```

### 6. Train Reconstruction Model

```bash
python Final_Model.py
```

### 7. Cross-Dataset Testing

```bash
python cross_dataset_testing.py
```


---

## 📈 Evaluation

Evaluation metrics include:

* RMSE (Root Mean Square Error)
* R² Score
* Pearson Correlation

Metrics are computed **per segment and averaged**, ensuring balanced evaluation across ECG samples.

---

## 📚 Datasets

This project uses:

* PTB-XL ECG Dataset
* PTB Diagnostic ECG Database


---

## Notes

* Large files (datasets, intermediate outputs, and trained models) are excluded using `.gitignore`.
* Paths may need to be adjusted depending on your local setup.

---


## License

This project is intended for research purposes.
