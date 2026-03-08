# Annotated Notes: I-JEPA (Image-based Joint-Embedding Predictive Architecture)

**Paper**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
**Authors**: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas
**Venue**: CVPR 2023
**Link**: [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243)

---

## 1. Overview

I-JEPA is the first concrete implementation of the JEPA concept outlined in LeCun's 2022 position paper. It learns visual representations by predicting the embeddings of masked image regions from the embeddings of visible regions -- entirely in latent space, never reconstructing pixels.

### Core Claim

A non-generative, non-contrastive self-supervised method can learn semantic image representations that:
- Match or exceed masked autoencoders (MAE) on semantic tasks.
- Match or exceed contrastive methods (DINO, iBOT) on low-level tasks.
- Are significantly more compute-efficient than both.

---

## 2. Architecture Details

### 2.1 Three Components

```
Image x
  |
  |--> [Context Encoder f_theta] --> context embeddings s_x
  |         (applied to visible/unmasked patches)
  |
  |--> [Target Encoder f_theta_bar] --> target embeddings s_y
  |         (applied to masked/target patches)
  |
  s_x --> [Predictor g_phi] --> predicted target embeddings s_y_hat

  Loss: L = ||s_y - s_y_hat||^2   (L2 in latent space)
```

#### Context Encoder (f_theta)

- Standard Vision Transformer (ViT).
- Receives only the **visible** (unmasked) patches of the image as input.
- Parameters updated by gradient descent.
- Processes patches with full self-attention among visible patches only.

#### Target Encoder (f_theta_bar)

- Same architecture as the context encoder.
- Receives the **full image** (all patches) but outputs are only read at the target (masked) positions.
- Parameters are NOT updated by gradient descent. Instead, they are an Exponential Moving Average (EMA) of the context encoder parameters.
- This asymmetry is critical for preventing collapse.

#### Predictor (g_phi)

- A smaller transformer that takes the context encoder's output embeddings and predicts the target encoder's output at the masked positions.
- Takes as additional input: positional embeddings (mask tokens) indicating which positions need to be predicted.
- Parameters updated by gradient descent.
- Narrower than the main encoder (e.g., 384-dim vs. 1024-dim for ViT-H).

### 2.2 Information Flow

```
Input image (224x224)
    |
    v
Patchify (16x16 patches) --> 196 patches (14x14 grid)
    |
    |--> Apply context mask --> ~60 visible patches
    |         |
    |         v
    |    Context Encoder --> 60 context embeddings
    |         |
    |         v
    |    Predictor (+ positional mask tokens) --> predicted target embeddings
    |
    |--> Apply target mask --> ~56 target patches (4 blocks)
              |
              v
         Target Encoder --> 56 target embeddings
              |
              v
         L2 loss between predicted and actual target embeddings
```

---

## 3. Multi-Block Masking Strategy

This is arguably the most important design decision in I-JEPA. The masking strategy determines what the model must learn to predict.

### How It Works

1. **Target selection**: Sample 4 non-overlapping rectangular blocks from the image. Each block covers a contiguous region of patches (e.g., a 4x3 or 5x4 region). Together, the target blocks cover a substantial portion of the image.

2. **Context selection**: The context is everything NOT in the target blocks. Typically this leaves about 60% of patches visible. A single large complementary region, not multiple small scattered patches.

3. **Key constraint**: The target blocks are spatially large and semantically meaningful. This forces the model to learn high-level features, not just local texture.

### Why Multi-Block Masking Matters

| Masking Strategy | What the Model Learns | Problem |
|---|---|---|
| Random patches (MAE-style) | Local texture, low-level features | Biased toward pixel-level reconstruction cues |
| Single large block | Coarse global structure | Too easy if block is small; too ambiguous if large |
| **Multiple large blocks** (I-JEPA) | **Semantic structure, object parts, spatial relationships** | Goldilocks: hard enough to require understanding, tractable enough to learn |

The multi-block strategy means the model must understand:
- What objects are present (to predict their features in masked regions).
- How objects relate spatially (to predict features at specific locations).
- Scene-level semantics (to fill in multiple diverse regions coherently).

### Masking Hyperparameters

- Number of target blocks: 4
- Target block aspect ratio: sampled uniformly from [0.75, 1.5]
- Target block scale: each block covers 0.15-0.2 of the image area
- Total target coverage: roughly 50-60% of patches
- Context: remaining patches (roughly 40-50%)

---

## 4. EMA (Exponential Moving Average) for Target Encoder Stability

### Why EMA?

If both encoders were trained by gradient descent, the system could collapse -- both encoders could converge to outputting constant vectors, making the prediction loss trivially zero.

The EMA target encoder solves this:

```
theta_bar = tau * theta_bar + (1 - tau) * theta
```

Where tau is the momentum coefficient (e.g., 0.996 to 0.999, following a cosine schedule).

### Properties

- The target encoder provides a **slowly moving target** for the predictor to chase.
- Because it changes slowly, the predictor must learn genuine predictive features rather than co-adapting with a rapidly changing target.
- This is the same principle used in MoCo, BYOL, and DINO.
- The EMA schedule typically starts with a lower momentum (more aggressive updates) and increases toward 1.0 (very slow updates) as training progresses.

### EMA Schedule

```
tau(t) = tau_base + (1 - tau_base) * (1 - cos(pi * t / T)) / 2
```

Starting tau: ~0.996
Final tau: ~0.999 or higher
This cosine schedule provides faster learning early and more stable representations later.

---

## 5. Training Details

### ViT Variants Evaluated

| Model | Params | Patch Size | Embed Dim | Heads | Layers | Predictor Dim |
|-------|--------|-----------|-----------|-------|--------|---------------|
| ViT-B/16 | 86M | 16x16 | 768 | 12 | 12 | 384 |
| ViT-L/16 | 307M | 16x16 | 1024 | 16 | 24 | 384 |
| ViT-H/16 | 632M | 16x16 | 1280 | 16 | 32 | 384 |
| ViT-H/14 | 632M | 14x14 | 1280 | 16 | 32 | 384 |

### ImageNet Training

- **Dataset**: ImageNet-1K (1.28M images, 1000 classes)
- **No labels used during pretraining** -- purely self-supervised.
- **Resolution**: 224x224 (with some experiments at 448x448)
- **Batch size**: 2048
- **Optimizer**: AdamW
- **Learning rate**: 1.5e-4 with cosine decay
- **Weight decay**: 0.05
- **Epochs**: 300-600 depending on model size
- **Augmentations**: Minimal -- only random resized crop and horizontal flip. NO color jitter, NO multi-crop, NO solarization. This is a key advantage: I-JEPA does not rely on hand-crafted augmentations.

### Compute Requirements

| Model | GPUs | Training Time | GPU-Hours |
|-------|------|--------------|-----------|
| ViT-B/16 | 16 A100s | ~72 hours | ~1,152 |
| ViT-L/16 | 32 A100s | ~72 hours | ~2,304 |
| ViT-H/16 | 32 A100s | ~96 hours | ~3,072 |
| ViT-H/14 | 64 A100s | ~120+ hours | ~7,680+ |

Key efficiency advantage: I-JEPA is faster than methods requiring multiple views/augmentations (DINO needs 2 global + 8 local crops) or pixel reconstruction (MAE needs a decoder).

---

## 6. Key Results and Benchmarks

### ImageNet Linear Probing (Top-1 Accuracy)

| Method | ViT-B | ViT-L | ViT-H |
|--------|-------|-------|-------|
| MAE | 68.0 | 75.8 | 77.2 |
| DINO | 78.2 | -- | -- |
| iBOT | 79.5 | -- | -- |
| **I-JEPA** | **74.5** | **77.0** | **80.2** |

I-JEPA with ViT-H/14 at 448 resolution reaches ~81.1%.

### ImageNet 1% Semi-Supervised

| Method | ViT-H |
|--------|-------|
| MAE | 61.5 |
| **I-JEPA** | **72.4** |

Massive advantage in low-data regimes -- I-JEPA's representations are more semantic.

### Transfer Learning (Linear Probing on Other Datasets)

I-JEPA shows strong transfer to:
- Places205
- iNaturalist 2018
- CIFAR-100
- Various fine-grained classification tasks

### Object Counting and Low-Level Tasks

I-JEPA also performs well on tasks requiring spatial understanding (object counting, depth estimation), unlike some contrastive methods that focus too heavily on global semantics.

---

## 7. How I-JEPA Differs from Other Methods

### I-JEPA vs. MAE (Masked Autoencoder)

| Aspect | MAE | I-JEPA |
|--------|-----|--------|
| **Prediction target** | Raw pixels (via decoder) | Latent embeddings |
| **Decoder** | Required (transforms latent to pixels) | Not needed |
| **Masking ratio** | Very high (75%) | Moderate (~50-60%) |
| **Masking pattern** | Random individual patches | Multi-block (large contiguous regions) |
| **Augmentations** | Random crop, flip | Random crop, flip (same) |
| **Learned features** | Biased toward low-level texture | More semantic, higher-level |
| **Linear probe performance** | Lower | Higher |
| **Compute for same quality** | Higher (needs decoder forward/backward) | Lower |

**Key takeaway**: MAE's pixel reconstruction objective forces it to spend capacity on low-level details. I-JEPA's latent prediction naturally abstracts away irrelevant detail.

### I-JEPA vs. CLIP

| Aspect | CLIP | I-JEPA |
|--------|------|--------|
| **Training data** | Image-text pairs (400M+) | Images only (1.28M) |
| **Supervision signal** | Language (text captions) | Self-supervised (masked prediction) |
| **Negatives required** | Yes (contrastive) | No |
| **Zero-shot capability** | Yes (via text prompts) | No (needs a head/probe) |
| **Data efficiency** | Low (needs massive paired data) | High (ImageNet-1K suffices) |
| **Representation quality** | Excellent for text-aligned semantics | Excellent for visual semantics |

### I-JEPA vs. SimCLR

| Aspect | SimCLR | I-JEPA |
|--------|--------|--------|
| **Approach** | Contrastive (positive/negative pairs) | Predictive (latent prediction) |
| **Augmentations** | Heavy (color jitter, Gaussian blur, etc.) | Minimal (crop + flip) |
| **Batch size sensitivity** | Very sensitive (needs large batches for negatives) | Not sensitive |
| **Collapse prevention** | Contrastive negatives | EMA target encoder |
| **Shortcut problem** | Can rely on augmentation-specific cues | No augmentation shortcuts |

### I-JEPA vs. DINO / DINOv2

| Aspect | DINO | I-JEPA |
|--------|------|--------|
| **Architecture** | Student-teacher with multi-crop | Context-target with multi-block masking |
| **Views** | Multiple augmented views (2 global + 8 local) | Single image, spatially masked |
| **Augmentations** | Heavy multi-crop + color augmentation | Minimal |
| **Prediction** | Global: student matches teacher on [CLS] | Spatial: predict latent features at specific positions |
| **Spatial understanding** | Weaker (global feature matching) | Stronger (must predict spatially) |
| **Compute** | Higher (multiple forward passes per image) | Lower (single forward pass) |

---

## 8. Significance for the JEPA Lineage

I-JEPA established several principles that carry forward to V-JEPA, V-JEPA 2, and VL-JEPA:

1. **Latent prediction works**: You can learn excellent representations without ever reconstructing pixels.
2. **EMA target encoders prevent collapse**: No need for contrastive negatives or explicit regularization like VICReg.
3. **Masking strategy matters more than augmentation strategy**: The spatial structure of what you mask determines what features emerge.
4. **Minimal augmentations are sufficient**: When the masking is right, hand-crafted augmentations add little.
5. **The predictor should be lightweight**: A small transformer predictor suffices; the heavy lifting is in the encoder.

### What I-JEPA Does NOT Do

- No video (temporal dimension) -- this is addressed by V-JEPA.
- No language integration -- this is addressed by VL-JEPA.
- No action conditioning -- this is addressed by V-JEPA 2-AC.
- No hierarchical multi-scale prediction -- still an open research direction.

---

## 9. Reproducibility Notes

The official codebase is available at: `github.com/facebookresearch/ijepa`

Key implementation details that matter for reproduction:
- The predictor uses **cross-attention** where mask tokens attend to context embeddings.
- Positional embeddings are sinusoidal (2D sin-cos) and shared between context and target encoders.
- The target encoder processes the FULL image but loss is only computed at masked positions.
- Gradient is NOT propagated through the target encoder (stop-gradient + EMA).
- The predictor is discarded after pretraining; only the context encoder (or target encoder) is used for downstream tasks.

---

## 10. Open Questions

- How does multi-block masking translate to video? (Answered: V-JEPA uses spatiotemporal tube masking)
- Can JEPA handle multi-modal inputs? (Answered: VL-JEPA adds language)
- Does the predictor quality plateau, or can it scale further?
- What is the optimal masking strategy for different downstream tasks?
- Can I-JEPA learn from uncurated data as effectively as from ImageNet?

---

*Last updated: 2026-03-07*
