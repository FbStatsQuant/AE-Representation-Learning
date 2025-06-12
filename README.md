# Autoencoder for Representation Learning and Classification

This project implements a **PyTorch Autoencoder (AE)** trained on high-dimensional input data to extract **compact, nonlinear latent representations**. These learned embeddings are then used in a downstream **classification task**, and compared to traditional **PCA projections**.

---

## 📌 Highlights

- Autoencoder for feature compression and denoising
- 2D latent space for visualization and classification
- Compared against PCA baseline
- Trained on labeled dataset (e.g., MNIST-style)
- Performance tested using logistic regression / shallow MLP

---

## 🧠 Methodology

- The AE learns an encoder–decoder mapping:
  - **Encoder**: maps input to latent space
  - **Decoder**: reconstructs original input from latent code
- After training, the 2D latent vectors from the encoder are used as features for a classifier
- PCA is applied separately to the same input data (same latent dimension) for comparison

---

## 📈 PCA Projection

![PCA](bba6a635-2ee4-4d68-a840-1354d09aaac4.png)

> **PCA Projection Colored by True Labels**  
> This 2D projection shows class clusters under a linear PCA projection. Overlapping regions (e.g., labels 3–6) demonstrate the limitations of PCA in separating nonlinear structure.

---

## ⚙️ Run Instructions

### 1. Install dependencies:
```bash
pip install torch numpy matplotlib scikit-learn
