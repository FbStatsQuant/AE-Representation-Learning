# Autoencoders as Nonlinear PCA

This repository explores **autoencoders** as a nonlinear generalization of Principal Component Analysis. Two notebooks implement the same core ideas — 2D bottleneck compression, latent space visualization, and comparison against linear PCA — using different frameworks and datasets to demonstrate both PyTorch and TensorFlow/Keras workflows.

---

## Notebooks

### 1. `1_AE_MNIST_torch.ipynb` — PyTorch · MNIST
Autoencoder trained on MNIST (70,000 handwritten digit images, 28×28 pixels). The encoder compresses each image from 784 dimensions down to a 2D latent vector. The decoder reconstructs the original image from those 2 numbers.

**Highlights**
- Encoder/Decoder defined as separate `nn.Module` classes, composed into a full autoencoder
- Unsupervised training with MSE reconstruction loss — no labels used
- 2D latent space scatter plot colored by digit class, compared side-by-side against linear PCA
- Latent space walk: decodes a 15×15 grid of points across the learned manifold
- Digit interpolation: smooth transition between any two digits via linear interpolation in latent space

### 2. `2_AE_Digits_tensorflow.ipynb` — TensorFlow/Keras · Sklearn Digits
Autoencoder trained on sklearn's Digits dataset (1,797 handwritten digit images, 8×8 pixels). Same architecture concept, different framework and scale — demonstrates TensorFlow's Keras functional API and shows that meaningful nonlinear representations emerge even from very small datasets.

**Highlights**
- Encoder, decoder, and full autoencoder built as three separate named `Model` objects using the Keras functional API
- Training managed entirely via `model.fit()` with `EarlyStopping` and `ReduceLROnPlateau` callbacks
- Same visualization pipeline as the PyTorch notebook: latent space, PCA comparison, latent walk, interpolation
- Trains in seconds on CPU — no GPU required

---

## The Core Idea

A standard autoencoder with a linear encoder and decoder is mathematically equivalent to PCA — it learns the same principal subspace. Adding nonlinear activations allows the encoder to follow the **curved manifold** of the data rather than being constrained to a flat hyperplane.

```
Input x ──► Encoder f(x) ──► z ∈ ℝ²  ──► Decoder g(z) ──► x̂ ≈ x
              (nonlinear)   bottleneck    (nonlinear)
```

With a 2D bottleneck the latent space is directly visualizable. The key result in both notebooks is the side-by-side plot of the autoencoder's latent space against PCA's 2D projection — the nonlinear compression produces tighter, better-separated digit clusters because it can represent the curved structure of the data manifold that PCA cannot.

---

## Repository Structure

```
autoencoders/
├── 1_AE_MNIST_torch.ipynb           # PyTorch — MNIST (70k samples, 28x28)
├── 2_AE_Digits_tensorflow.ipynb     # TensorFlow — Sklearn Digits (1.8k samples, 8x8)
└── README.md
```

---

## Requirements

**PyTorch notebook**
```
torch
torchvision
scikit-learn
matplotlib
numpy
```

**TensorFlow notebook**
```
tensorflow
scikit-learn
matplotlib
numpy
```

Install all at once:
```bash
pip install torch torchvision tensorflow scikit-learn matplotlib numpy
```

Datasets are loaded automatically — MNIST via `torchvision.datasets`, Digits via `sklearn.datasets.load_digits()`. No manual downloads required.

---

## Results Summary

| Notebook | Framework | Dataset | Samples | Input dim | Bottleneck |
|---|---|---|---|---|---|
| `1_AE_MNIST_torch` | PyTorch | MNIST | 70,000 | 784 | 2D |
| `2_AE_Digits_tensorflow` | TensorFlow | Sklearn Digits | 1,797 | 64 | 2D |

Both notebooks reach similar qualitative conclusions: the nonlinear autoencoder produces a more structured 2D latent space than linear PCA, with cleaner cluster separation and smoother interpolation paths between classes.

---

## Related Lectures

- `lecture_5_autoencoder.md` — Mathematical foundations: the autoencoder framework, connection to PCA, bottleneck design, denoising autoencoders, strengths and weaknesses
