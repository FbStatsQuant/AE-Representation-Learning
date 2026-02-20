# Autoencoders and Nonlinear PCA

## Introduction

Principal Component Analysis finds the linear subspace that best preserves variance in high-dimensional data. It is optimal within the class of linear transformations — but real data rarely lives on a linear manifold. Handwritten digits, face images, and financial time series all exhibit curved, nonlinear structure that a linear projection cannot capture without significant information loss.

**Autoencoders** extend the PCA idea to nonlinear transformations. By training a neural network to compress data into a low-dimensional bottleneck and then reconstruct it, we learn a nonlinear embedding that preserves the structure of the data manifold far more faithfully than PCA. This lecture develops the mathematical foundations of autoencoders, their connection to PCA, and their key variants.

---

## 1. The Autoencoder Framework

### Setup

An autoencoder is a neural network trained to reproduce its input at the output. It consists of two components:

- **Encoder** $f_\phi : \mathbb{R}^d \to \mathbb{R}^r$ — maps input $x$ to a latent code $z$
- **Decoder** $g_\psi : \mathbb{R}^r \to \mathbb{R}^d$ — maps latent code $z$ back to a reconstruction $\hat{x}$

where $r \ll d$ is the **bottleneck dimension**.

The full forward pass is:

$$z = f_\phi(x), \qquad \hat{x} = g_\psi(z) = g_\psi(f_\phi(x))$$

### Objective

Training minimizes the **reconstruction loss** — the discrepancy between the input and its reconstruction:

$$\mathcal{L}(\phi, \psi) = \frac{1}{n} \sum_{i=1}^n \ell(x_i, g_\psi(f_\phi(x_i)))$$

For continuous inputs, the standard choice is **Mean Squared Error**:

$$\ell(x, \hat{x}) = \|x - \hat{x}\|_2^2$$

For binary inputs (e.g., binarized images), **Binary Cross-Entropy** is more appropriate:

$$\ell(x, \hat{x}) = -\sum_{j=1}^d \left[x_j \log \hat{x}_j + (1 - x_j)\log(1 - \hat{x}_j)\right]$$

### Critical Observation

**No labels are used.** The autoencoder is an **unsupervised** model — the supervision signal is the input itself. This makes autoencoders applicable to any dataset, regardless of whether labels exist.

---

## 2. Connection to PCA

### Linear Autoencoders are PCA

Consider an autoencoder with a single linear encoder and a single linear decoder:

$$z = W_e x + b_e, \qquad \hat{x} = W_d z + b_d$$

where $W_e \in \mathbb{R}^{r \times d}$ and $W_d \in \mathbb{R}^{d \times r}$.

**Theorem**: The global minimum of the MSE reconstruction loss for this architecture is achieved when the columns of $W_d$ span the same subspace as the top $r$ principal components of the data covariance matrix.

In other words, a linear autoencoder with an $r$-dimensional bottleneck learns exactly the same $r$-dimensional subspace as PCA — the one that maximizes explained variance.

### Why Nonlinearity Helps

PCA is constrained to find a **flat** $r$-dimensional subspace in $\mathbb{R}^d$. If the data lies on a curved manifold — a sphere, a helix, a nonlinear cluster structure — a flat subspace cannot faithfully represent it.

A nonlinear autoencoder with activation functions $\phi$ in the encoder:

$$z = f_\phi(x) = \phi_L(\cdots \phi_1(W_1 x + b_1) \cdots)$$

can learn a **curved** mapping that follows the data manifold, achieving lower reconstruction error with the same bottleneck dimension $r$.

### Geometric Interpretation

| Method | Compression | Manifold assumption |
|---|---|---|
| PCA | Linear projection onto flat subspace | Data lies near a hyperplane |
| Autoencoder | Nonlinear projection onto learned manifold | Data lies near a curved surface |

The 2D latent space of a trained autoencoder on MNIST shows digit clusters that are better separated and more structured than PCA's 2D projection — evidence that the data manifold is genuinely nonlinear.

---

## 3. Architecture Design

### Encoder

The encoder maps $x \in \mathbb{R}^d$ to $z \in \mathbb{R}^r$ through a sequence of nonlinear transformations:

$$h^{(1)} = \phi(W^{(1)} x + b^{(1)})$$
$$h^{(\ell)} = \phi(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}), \quad \ell = 2, \ldots, L$$
$$z = W^{(L+1)} h^{(L)} + b^{(L+1)}$$

**No activation on the bottleneck layer** — the latent code $z$ should be unconstrained to allow the encoder to use the full real line.

### Decoder

The decoder is symmetric to the encoder, mapping $z \in \mathbb{R}^r$ back to $\hat{x} \in \mathbb{R}^d$:

$$\hat{x} = g_\psi(z) = \sigma(W^{(1)}_d \phi(\cdots \phi(W^{(L)}_d z + b^{(L)}_d) \cdots) + b^{(1)}_d)$$

**Output activation**: sigmoid for inputs normalized to $[0, 1]$, identity for standardized inputs.

### Bottleneck

The bottleneck dimension $r$ controls the compression ratio:

$$\text{Compression ratio} = \frac{d}{r}$$

For MNIST with $d = 784$ and $r = 2$: the compression ratio is 392 — the network must encode each image into just 2 numbers.

Choosing $r$:
- Too small: underfitting — the bottleneck cannot represent the data structure
- Too large: trivial solution — the network memorizes rather than compresses
- Sweet spot: the intrinsic dimensionality of the data manifold

### Implementation in PyTorch

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)    # no activation on bottleneck
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()                  # output in [0, 1]
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


model     = Autoencoder(latent_dim=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop — labels are not used
for X, _ in train_loader:
    X     = X.view(X.size(0), -1).to(device)   # flatten to (B, 784)
    X_hat, z = model(X)
    loss  = criterion(X_hat, X)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 4. The Latent Space

### Visualization

With $r = 2$, the latent space can be plotted directly. Each test sample is encoded to a point $(z_1, z_2) \in \mathbb{R}^2$ and plotted with its class label as color.

A well-trained autoencoder produces a latent space where:
- **Same-class samples cluster together** — their shared structure maps to nearby codes
- **Different-class samples separate** — their distinct features map to different regions
- **Transitions are smooth** — nearby points in latent space decode to similar images

This structure emerges entirely without label supervision — the autoencoder discovers it by learning to reconstruct.

### Latent Space Walk

Decoding a regular grid of points $\{(z_1, z_2)\}$ covering the latent space produces a visual map of what the decoder generates across the learned manifold. Regions of the latent space that correspond to no training data typically decode to blurry or unrecognizable images — the decoder has not been trained on those codes.

### Interpolation

Given two samples $x_a$ and $x_b$, their latent codes are $z_a = f_\phi(x_a)$ and $z_b = f_\phi(x_b)$. A linear interpolation in latent space:

$$z(\alpha) = (1 - \alpha) z_a + \alpha z_b, \quad \alpha \in [0, 1]$$

produces a sequence of decoded images $g_\psi(z(\alpha))$ that transitions smoothly between $x_a$ and $x_b$. This is qualitatively different from interpolating in pixel space, which produces blurry averages with no coherent structure.

---

## 5. Undercomplete vs Overcomplete Autoencoders

### Undercomplete

$r < d$ — the bottleneck is smaller than the input. This is the standard setting. The compression forces the encoder to discard redundant information and retain only the most salient structure.

### Overcomplete

$r \geq d$ — the bottleneck is as large or larger than the input. Without additional constraints, the network can learn the identity function trivially:

$$f_\phi(x) = x, \quad g_\psi(z) = z$$

achieving zero reconstruction loss while learning nothing useful. Overcomplete autoencoders require **regularization** to prevent this:

- **Sparse autoencoder**: penalizes non-zero activations in $z$, forcing most units to be silent
- **Denoising autoencoder**: corrupts the input before encoding, forces the encoder to learn robust features
- **Contractive autoencoder**: penalizes the Frobenius norm of the encoder Jacobian, forcing the representation to be insensitive to small input perturbations

---

## 6. Denoising Autoencoders

A **denoising autoencoder** (DAE) trains to reconstruct clean input $x$ from a corrupted version $\tilde{x}$:

$$\mathcal{L}(\phi, \psi) = \frac{1}{n} \sum_{i=1}^n \|\tilde{x}_i - g_\psi(f_\phi(x_i))\|^2$$

wait — the loss compares the reconstruction to the **clean** input $x_i$, not the corrupted $\tilde{x}_i$. The network receives corrupted data but is penalized for failing to recover the original.

Common corruption strategies:

- **Gaussian noise**: $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Masking**: randomly set a fraction of input dimensions to zero
- **Salt-and-pepper**: randomly set pixels to 0 or 1

```python
def add_noise(x, noise_factor=0.3):
    noisy = x + noise_factor * torch.randn_like(x)
    return torch.clamp(noisy, 0.0, 1.0)

# In the training loop
X_noisy = add_noise(X)
X_hat, _ = model(X_noisy)
loss = criterion(X_hat, X)    # reconstruct clean from noisy
```

DAEs learn more robust representations than standard autoencoders and can be applied directly to image denoising tasks.

---

## 7. Strengths and Weaknesses

### Strengths

**Unsupervised**: No labels required. Autoencoders can be applied to any dataset and are especially valuable when labeled data is scarce or expensive.

**Nonlinear compression**: Unlike PCA, autoencoders can follow curved data manifolds, achieving substantially lower reconstruction error at the same bottleneck dimension on real-world data.

**Interpretable latent space**: With $r = 2$ or $r = 3$, the latent space is directly visualizable, providing geometric insight into the data structure that is unavailable from the raw high-dimensional input.

**Flexible architecture**: The encoder and decoder can be any differentiable architecture — fully connected, convolutional, recurrent — making autoencoders applicable to images, sequences, graphs, and tabular data.

**Generative capability**: The decoder $g_\psi$ is a generative model — given any $z \in \mathbb{R}^r$, it produces a plausible reconstruction. Latent space interpolation and sampling enable controlled generation.

### Weaknesses

**No density model**: The standard autoencoder does not define a probability distribution over the latent space. Sampling random $z$ from an arbitrary distribution and decoding will often produce unrecognizable outputs if $z$ falls outside the region covered by training codes. This is the key motivation for the Variational Autoencoder.

**Blurry reconstructions with MSE**: MSE loss averages over all plausible reconstructions, producing blurry outputs. This is particularly visible with small bottlenecks. Perceptual losses or adversarial training are used to sharpen reconstructions in practice.

**No guaranteed structure in latent space**: The latent code $z$ has no enforced distribution. Nearby points in pixel space may map to distant points in latent space, and the latent space may have discontinuities or holes where the decoder produces garbage.

**Compression-reconstruction tradeoff**: As $r$ decreases, reconstruction quality degrades. There is no principled way to choose $r$ without domain knowledge about the intrinsic dimensionality of the data.

**Sensitive to bottleneck size**: Too small and the model underfits; too large without regularization and the model overfits trivially. Hyperparameter sensitivity is higher than in supervised models where validation accuracy gives a clear signal.

---

## 8. Summary

1. **An autoencoder trains a neural network to compress input into a low-dimensional bottleneck and reconstruct it**, using the reconstruction error as the sole training signal — no labels required.

2. **A linear autoencoder is equivalent to PCA** — it learns the top $r$ principal components of the data. Adding nonlinear activations allows the encoder to follow curved data manifolds, yielding richer representations at the same bottleneck dimension.

3. **The bottleneck dimension** $r$ controls the compression ratio. With $r = 2$, the latent space is directly visualizable — digit classes cluster and separate in a way that reveals the nonlinear structure of the data manifold.

4. **Undercomplete autoencoders** ($r < d$) learn by compression. **Overcomplete autoencoders** ($r \geq d$) require additional regularization (sparsity, noise, contraction) to avoid learning the identity.

5. **Denoising autoencoders** corrupt the input before encoding and train to reconstruct the clean version, learning more robust and generalizable representations.

6. **The key limitation** of standard autoencoders is the absence of a structured latent space — there is no guarantee that the distribution of codes is regular or that decoding arbitrary $z$ produces meaningful outputs. This motivates the Variational Autoencoder, which imposes a prior distribution on $z$.

---

## References

- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*.
- Bourlard, H., & Kamp, Y. (1988). Auto-association by multilayer perceptrons and singular value decomposition. *Biological Cybernetics*.
- Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P. A. (2010). Stacked denoising autoencoders. *JMLR*.
- Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive auto-encoders. *ICML*.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.

---

**Previous Lecture**: Weighted Loss for Imbalanced Classification  
**Next Lecture**: Variational Autoencoder (VAE)
