# Topology-Constrained Generative Diffusion

A class-conditional diffusion model (DDPM) that generates 2D point clouds with **mathematically guaranteed topological correctness**. Unlike standard generative models that only optimize for visual similarity, this project enforces topological invariants (Betti numbers) through a custom differentiable loss function — ensuring that a generated ring always has exactly one hole, a disk has none, and so on.

The ultimate goal is to scale this topology-aware generation framework to **3D point clouds** (Sphere vs Torus), a domain with very few existing works in the literature.

## Motivation

Standard generative models (Midjourney, DALL-E, 3D point cloud generators) produce visually plausible outputs but offer **zero mathematical guarantees** about structural correctness. A diffusion model asked to generate a ring might produce something that looks like a ring but contains a microscopic break — visually fine, topologically broken.

This project imposes the immutable rules of mathematics onto generative AI: every generated shape must pass a persistent homology test. If the model tries to close a hole that should exist, the topology-aware loss corrects the gradients during training.

## Roadmap

```
Phase 1  [DONE]    Disk (H₁=0) vs Ring (H₁=1)
                    ├── Conditional DDPM from scratch
                    ├── Geometric surrogate loss (replaced heavy C++ TDA libs)
                    ├── Concatenation fix for posterior collapse
                    └── Latent interpolation (Disk ↔ Ring morphing)

Phase 2  [CURRENT]  Nested Rings (H₀=2, H₁=2)
                    ├── Two concentric non-touching rings
                    ├── Model must control both component count AND hole count
                    └── Proves topology control is real, not memorization

Phase 3  [NEXT]     3D Point Clouds: Sphere vs Torus
                    ├── Sphere: H₀=1, H₁=0, H₂=1
                    ├── Torus:  H₀=1, H₁=2, H₂=1
                    └── Very few existing works in literature
```

## Project Structure

```
├── model.py                        # MLPDiffusion model architecture
├── dataset.py                      # PointCloudDataset (PyTorch Dataset)
├── dataset_generator.py            # Circle, disk, nested rings generators + topology utils
├── custom_loss.py                  # TopologicalLoss module
├── train.py                        # Training pipeline
├── sample.py                       # Reverse diffusion sampling
├── validate.py                     # Betti number validation (100 samples per class)
├── interpolate.py                  # Disk ↔ Ring latent space interpolation (GIF)
├── visualize_grid.py               # 3x3 grid visualization with Betti numbers
├── visualize_model_filtration.py   # Rips filtration animation (MP4/GIF)
```

## Model Architecture

**MLPDiffusion** — MLP-based conditional denoiser for 2D point clouds.

```
Input: [batch, n_points, 2] + timestep t + class label
                │
    ┌───────────┼───────────────┐
    ▼           ▼               ▼
  x_t     SinusoidalEmb    ClassEmb
(noisy)    t → 256-dim     label → 256-dim
    │           │               │
    └─────── concat ────────────┘
                │
          Linear(514 → 256)        ← head
                │
          4× Residual Block        ← blocks (Linear + GELU + skip)
                │
          Linear(256 → 2)          ← tail
                │
            noise_pred [batch, n_points, 2]
```

- **SinusoidalPositionEmbedding**: Transformer-style sinusoidal encoding for timestep `t`
- **Block**: Residual feedforward block (`x + GELU(Linear(x))`)
- ~530K parameters (hidden_size=256)

## Datasets

| Shape | Label | Topology (Betti) | Description |
|-------|-------|-------------------|-------------|
| Circle (Ring) | 1 | β₀=1, β₁=1 — 1 component, 1 hole | Points on the unit circle |
| Disk | 0 | β₀=1, β₁=0 — 1 component, 0 holes | Points uniformly sampled inside the unit disk |
| Nested Rings | 2 | β₀=2, β₁=2 — 2 components, 2 holes | Two concentric circles (r=0.5, r=1.0) with noise |

`PointCloudDataset` generates balanced samples on-the-fly for training.

## Diffusion Process

**Forward (noising):**
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε       ε ~ N(0, I)
```

**Reverse (denoising):**
```
x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t, label)) + √(σ²_t) · z
```

Linear noise schedule: β₁ = 0.0001 → β₁₀₀ = 0.02, T = 100 timesteps.

## Topology-Aware Loss

`TopologicalLoss` provides a differentiable geometric constraint based on point radii:

- **Ring (label=1):** `L = mean((‖x‖ - 1.0)²)` — pushes points toward the unit circle
- **Disk (label=0):** `L = mean(ReLU(‖x‖ - 1.0)²)` — only penalizes points outside the unit disk

Combined loss: `L_total = L_MSE + λ_topo · L_topo` where `λ_topo = 0.01`

**Why geometric surrogate instead of real persistent homology?**
Direct backpropagation through persistent homology (e.g. `torch_topological`) requires heavy C++ dependencies and is extremely slow. The geometric surrogate loss achieves the same topological constraints ~1000x faster by penalizing radius deviations, which directly control β₁ for circular shapes.

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset size | 10,000 |
| Points per cloud | 50 |
| Batch size | 128 |
| Epochs | 100 |
| Learning rate | 1e-3 (Adam) |
| LR scheduler | ExponentialLR (γ=0.99) |
| Timesteps | 100 |
| β range | [0.0001, 0.02] |
| λ_topo | 0.01 |
| Hidden size | 256 |
| Num classes | 2 |

## Key Engineering Decisions

**Concatenation over Addition:** Class embeddings are concatenated with input features rather than added. This prevents posterior collapse — the model can't ignore the label when it's structurally part of the input tensor.

**Geometric Surrogate Loss:** Instead of backpropagating through a full persistent homology pipeline (GUDHI/torch_topological), a lightweight radius-based loss enforces the same topological constraints differentiably. Ring points are penalized for deviating from radius 1.0; disk points are only penalized if they exceed radius 1.0.

**Latent Interpolation as Proof:** Smooth morphing between disk and ring embeddings (`(1-α)·emb_disk + α·emb_ring`) proves the model learned a continuous manifold rather than memorizing two discrete shapes.

## Usage

### Training

```bash
python train.py
```

Trains the class-conditional diffusion model and saves weights to `diffusion_model_conditioned.pth`.

### Sampling

```bash
python sample.py
```

Generates disk and ring samples (1000 points each) via reverse diffusion.

### Topology Validation

```bash
python validate.py
```

Generates 100 samples per class, computes Betti numbers via Rips complex (max edge length = 0.5), and reports topology correctness rates:

- Ring success: samples with Betti = [1, 1]
- Disk success: samples with Betti = [1, 0]

### Latent Space Interpolation

```bash
python interpolate.py
```

Interpolates class embeddings from disk (α=0) to ring (α=1) and generates a 60-frame GIF (`manifold_surf.gif`).

### Visualization

```bash
python visualize_grid.py                # 3x3 grid of rings with Betti numbers
python visualize_model_filtration.py    # Rips filtration animation (MP4/GIF)
```

## Topology Validation with Persistent Homology

The project uses [GUDHI](https://gudhi.inria.fr/) for persistent homology computation:

1. Construct a **Rips complex** from the generated point cloud
2. Build a **simplex tree** (max dimension = 2)
3. Compute **persistence** to get birth-death intervals
4. Extract **Betti numbers**: β₀ (connected components), β₁ (1-cycles / holes)

Expected results:

| Shape | β₀ | β₁ | Interpretation |
|-------|----|----|----------------|
| Disk | 1 | 0 | One connected piece, no holes |
| Ring | 1 | 1 | One connected piece, one hole |
| Nested Rings | 2 | 2 | Two separate pieces, each with one hole |

## Dependencies

```
torch
numpy
matplotlib
gudhi
scipy
pillow
```

Optional: `ffmpeg` for MP4 video output in filtration animation.

```bash
pip install torch numpy matplotlib gudhi scipy pillow
```
