# Plant Disease Detection with EfficientNetV2 & Explainable AI (Grad-CAM)

A **production-grade multi-class image classification pipeline** for plant disease detection, built using **EfficientNetV2** with a strong focus on **data correctness, evaluation rigor, and model interpretability**.

The system classifies **15 disease and healthy leaf categories** across tomato, potato, and pepper plants using the PlantVillage dataset.

This project was intentionally designed to follow **real-world ML engineering practices**, not just achieve high accuracy.

---

## 🚀 Why This Project Stands Out

Most ML projects stop at training accuracy.  
This project goes further by addressing **real failure modes** seen in applied ML:

- Proper **stratified train–validation split** (no class leakage)
- Explicit **class imbalance handling**
- **Correct preprocessing alignment** between training and inference
- Robust **post-training evaluation** (classification report & confusion matrix)
- **Explainable AI (Grad-CAM)** to verify model decision logic
- Fully **reproducible notebook** (runs top → bottom after restart)

---

## 🧠 Problem Statement

Early and accurate plant disease detection is critical for reducing crop loss and improving food security.  
Traditional inspection methods are manual, time-consuming, and error-prone.

This project builds an automated vision-based system capable of:
- Distinguishing visually similar diseases
- Handling class imbalance
- Providing **visual explanations** for predictions

---

## 📊 Dataset

- **Dataset**: PlantVillage (Kaggle)
- **Total Classes**: 15
- **Plants Covered**:
  - Tomato
  - Potato
  - Pepper
- **Note**: Dataset is not included in this repository due to size constraints

---

## 🏗️ Model Architecture

- **Backbone**: EfficientNetV2-B0 (ImageNet pretrained)
- **Head**:
  - Global Average Pooling
  - Dropout (regularization)
  - Softmax classification layer
- **Loss Function**: Sparse categorical cross-entropy
- **Imbalance Handling**: Class-weighted loss

EfficientNetV2 was chosen for its **parameter efficiency and strong generalization**, making it suitable for real-world deployment.

---

## ⚙️ Training Strategy

- Stratified train–validation split to ensure **all classes appear in validation**
- Class weights computed from training labels to prevent majority-class bias
- Consistent preprocessing using `EfficientNetV2.preprocess_input`
- Training history persisted separately to support session restarts

---

## 📈 Performance

| Metric | Value |
|------|------|
| Accuracy | **97%** |
| Macro F1-Score | **0.96** |
| Weighted F1-Score | **0.97** |

**Why this matters**:
- High **macro F1** confirms strong performance across **minority classes**
- Close alignment between macro and weighted metrics indicates **balanced learning**
- Results are consistent with the confusion matrix (no class collapse)

---

## 🔍 Evaluation & Error Analysis

- Full **classification report** with per-class precision, recall, and F1
- **Confusion matrix** used to identify residual confusions between visually similar diseases
- No evidence of majority-class dominance or random guessing

This evaluation pipeline avoids common pitfalls such as shuffled-label misalignment and unstratified validation.

---

## 🧪 Explainability (Grad-CAM)

Grad-CAM was used to interpret model predictions by visualizing class-specific activation regions.

### Key Observations:
- The model focuses on **disease-affected leaf regions** (lesions, vein distortions)
- Background pixels are largely ignored
- Activation patterns are smooth and biologically meaningful

This confirms that predictions are driven by **relevant visual features**, not spurious correlations.

---

## 🎯 Key Takeaways

This project demonstrates end-to-end ML engineering practices, including
correct data handling, robust evaluation, and explainable predictions.
It reflects a focus on **model correctness, reliability, and trust**, not just raw accuracy.

