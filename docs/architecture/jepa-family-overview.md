# The JEPA Family of Architectures: A Comprehensive Overview

> Reference document for the VL-JEPA Research Lab
> Last updated: March 2026

---

## Table of Contents

1. [The JEPA Concept](#1-the-jepa-concept)
2. [I-JEPA (Image JEPA)](#2-i-jepa-image-jepa-2023)
3. [V-JEPA (Video JEPA)](#3-v-jepa-video-jepa-2024)
4. [V-JEPA 2](#4-v-jepa-2-june-2025)
5. [VL-JEPA (Vision-Language JEPA)](#5-vl-jepa-december-2025)
6. [H-JEPA (Hierarchical JEPA)](#6-h-jepa-hierarchical-jepa)
7. [Comparison Table](#7-comparison-across-jepa-variants)

---

## 1. The JEPA Concept

### 1.1 Origin: Yann LeCun's Vision

The Joint Embedding Predictive Architecture (JEPA) was introduced by Yann LeCun
in his position paper "A Path Towards Autonomous Machine Intelligence" (June 2022).
The central thesis is that current AI systems -- whether generative (GPT, diffusion
models) or contrastive (CLIP, SimCLR) -- have fundamental limitations that prevent
them from learning world models suitable for autonomous intelligence.

LeCun proposed JEPA as the core architecture for a modular cognitive architecture
comprising:

- **Configurator**: modulates other modules based on task
- **Perception module**: estimates world state from observations
- **World model**: predicts future world states (this is JEPA)
- **Cost module**: computes energy/cost of predicted states
- **Actor**: proposes action sequences to minimize cost
- **Short-term memory**: stores relevant state information

### 1.2 Why Predict in Latent Space, Not Pixel Space

The key insight behind JEPA is that pixel-level prediction is both computationally
wasteful and semantically impoverished:

```
Problem with pixel-space prediction:
  - A video of a tree has millions of possible leaf configurations
  - Predicting exact pixel values forces the model to model irrelevant details
  - High-frequency noise and texture variation dominate the loss
  - The model spends capacity on visual details rather than semantic structure

JEPA solution:
  - Encode observations into a latent representation
  - Predict in latent space where irrelevant details are abstracted away
  - The encoder learns WHAT to keep and WHAT to discard
  - Predictions focus on semantically meaningful state changes
```

```
Pixel-Space Prediction (e.g., MAE, VideoGPT):

  Frame_t             Frame_t+1 (predicted pixels)
  +----------+        +----------+
  |  pixels  | -----> |  pixels  |    Must predict EVERY pixel
  | (H x W)  |  f()   | (H x W)  |    including noise, texture,
  +----------+        +----------+    lighting variations

Latent-Space Prediction (JEPA):

  Frame_t             Latent_t+1 (predicted representation)
  +----------+  enc   +------+        +------+
  |  pixels  | -----> |  z_t | -----> | z_t+1|   Only predicts
  | (H x W)  |        | (d)  |  f()   |  (d) |   SEMANTIC content
  +----------+        +------+        +------+
```

### 1.3 Energy-Based Model Formulation

JEPA is formulated as an Energy-Based Model (EBM). Rather than producing a
probability distribution over outputs, JEPA learns an energy function that assigns
low energy to compatible (observation, prediction) pairs and high energy to
incompatible ones.

```
Energy function:

  E(x, y) = D( s_y(y), Pred( s_x(x) ) )

Where:
  x       = input (context observation)
  y       = target (what we want to predict)
  s_x     = context encoder (maps x to latent representation)
  s_y     = target encoder (maps y to latent representation)
  Pred    = predictor network (predicts target latent from context latent)
  D       = distance function (e.g., L2, cosine)
```

```
                        JEPA Energy-Based Architecture
  +-----------+                                    +-----------+
  |           |                                    |           |
  |  Input x  |                                    | Target y  |
  |           |                                    |           |
  +-----+-----+                                    +-----+-----+
        |                                                |
        v                                                v
  +-----+-----+                                    +-----+-----+
  |  Context  |                                    |  Target   |
  |  Encoder  |                                    |  Encoder  |
  |   s_x()   |                                    |   s_y()   |
  +-----+-----+                                    +-----+-----+
        |                                                |
        v                                                v
  +-----+-----+        +----------+               +-----+-----+
  | Context   | -----> |          | ------------> | Target    |
  | Repr s_x  |        | Predictor|   Predicted   | Repr s_y  |
  +-----+-----+        |  Pred()  |   Target Rep  +-----+-----+
                        +----------+                     |
                              |                          |
                              v                          v
                        +-----+-----+              +-----+-----+
                        | Predicted |              |  Actual   |
                        | s_y hat   |              |   s_y     |
                        +-----------+              +-----------+
                              |                          |
                              +---------> D() <----------+
                                     (minimize)
```

### 1.4 Avoiding Representation Collapse

The fundamental challenge with joint embedding architectures is **collapse**: the
encoders learn to map all inputs to the same constant vector, achieving zero
prediction error trivially.

**Collapse modes:**

1. **Complete collapse**: all representations become identical (constant vector)
2. **Dimensional collapse**: representations occupy a low-dimensional subspace
3. **Cluster collapse**: representations collapse to a few discrete clusters

**JEPA's anti-collapse mechanisms:**

**Exponential Moving Average (EMA) for Target Encoder:**

The target encoder parameters are updated as a slow-moving average of the context
encoder, rather than being trained directly:

```
theta_target = alpha * theta_target + (1 - alpha) * theta_context

Where alpha is typically 0.996 to 0.999 (very slow update)
```

This provides a stable prediction target that evolves slowly, preventing the
trivial solution where both encoders collapse to the same constant.

**VICReg (Variance-Invariance-Covariance Regularization):**

Applied in some JEPA variants to explicitly prevent collapse:

```
L_VICReg = lambda * L_invariance + mu * L_variance + nu * L_covariance

L_invariance = MSE(z_predicted, z_target)     -- pull representations together
L_variance   = max(0, gamma - std(z))         -- prevent variance collapse
L_covariance = sum of off-diagonal elements   -- decorrelate dimensions
               of covariance matrix
```

**Masking strategy (implicit regularization):**

By requiring the predictor to fill in missing spatial/temporal regions, the encoder
is forced to learn rich representations that contain enough information for
reconstruction. Predicting from partial context prevents the shortcut of simply
copying input features.

### 1.5 Contrast with Other Paradigms

```
+-------------------+------------------+------------------+------------------+
|                   | Generative       | Contrastive      | JEPA             |
|                   | (GPT, Diffusion) | (CLIP, SimCLR)   |                  |
+-------------------+------------------+------------------+------------------+
| Prediction space  | Pixel/token      | No prediction    | Latent space     |
|                   | space            | (similarity only)|                  |
+-------------------+------------------+------------------+------------------+
| Training signal   | Reconstruct      | Pull positive    | Predict latent   |
|                   | exact input      | pairs together,  | representation   |
|                   |                  | push negatives   | of target        |
|                   |                  | apart            |                  |
+-------------------+------------------+------------------+------------------+
| Augmentations     | N/A (auto-       | Critical (defines| Not required     |
| required          | regressive)      | positive pairs)  | (masking only)   |
+-------------------+------------------+------------------+------------------+
| Negative samples  | N/A              | Required (large  | Not required     |
|                   |                  | batch or memory  |                  |
|                   |                  | bank)            |                  |
+-------------------+------------------+------------------+------------------+
| What is learned   | Distribution     | Similarity       | Predictive world |
|                   | over outputs     | structure        | model in latent  |
|                   |                  |                  | space            |
+-------------------+------------------+------------------+------------------+
| Collapse risk     | Low (explicit    | Medium (needs    | Medium (needs    |
|                   | targets)         | negatives)       | EMA + masking)   |
+-------------------+------------------+------------------+------------------+
| Wastes capacity   | Yes (modeling    | No               | No               |
| on irrelevant     | pixel noise)     |                  |                  |
| details           |                  |                  |                  |
+-------------------+------------------+------------------+------------------+
| Supports planning | Limited          | No               | Yes (energy      |
|                   |                  |                  | minimization)    |
+-------------------+------------------+------------------+------------------+
```

**Why not generative?** Models like GPT and diffusion models predict in the
observation space (tokens or pixels). This forces them to model all details of
the output, including stochastic and irrelevant variation. A generative video
model must predict every pixel of every frame, spending enormous capacity on
texture, lighting, and noise that carry no semantic information.

**Why not contrastive?** Models like CLIP and SimCLR learn by pulling
representations of augmented views together and pushing different samples apart.
They require carefully designed augmentations (which encode human priors) and
large batches of negative samples. They learn similarity metrics but do not learn
to predict -- they cannot serve as world models.

**Why JEPA?** JEPA combines the strengths: it learns to predict (enabling world
modeling and planning) but does so in a learned latent space that abstracts away
irrelevant detail. It does not need augmentations or negative samples.

---

## 2. I-JEPA (Image JEPA, 2023)

> Paper: "Self-Supervised Learning from Images with a Joint-Embedding Predictive
> Architecture" (Assran et al., CVPR 2023)
> Code: github.com/facebookresearch/ijepa

### 2.1 Architecture

```
I-JEPA Architecture
====================

Input Image (224x224)
       |
       v
+------+------+  Patchify (14x14 or 16x16 patches)
| Patch Grid   |  -> 16x16 = 256 patches (for patch size 14)
| (N patches)  |  -> 14x14 = 196 patches (for patch size 16)
+------+------+
       |
       |  Apply multi-block masking
       |
       v
+------+------+-------+------+------+
| Context     |       | Target       |
| Patches     |       | Patches      |
| (visible)   |       | (masked,     |
|             |       |  4 blocks)   |
+------+------+       +------+------+
       |                     |
       v                     v
+------+------+       +------+------+
| Context     |       | Target      |
| Encoder     |       | Encoder     |
| (ViT)       |       | (ViT, EMA)  |
| theta_c     |       | theta_t     |
+------+------+       +------+------+
       |                     |
       v                     |
+------+------+              |
| Context     |              |
| Repr (s_x)  |              |
+------+------+              |
       |                     |
       v                     v
+------+------+       +------+------+
| Predictor   |       | Target      |
| (narrow     | ----> | Repr (s_y)  |
|  transformer)|      |             |
| + mask       |      |             |
|   tokens    |       |             |
+------+------+       +------+------+
       |                     |
       v                     v
   s_y_hat            s_y (stop-grad)
       |                     |
       +-------> L2 <--------+
              (loss)
```

### 2.2 Components

**Context Encoder (ViT):**
- Standard Vision Transformer (ViT-L/16 or ViT-H/14)
- Processes only the visible (unmasked) patches
- Trained end-to-end via backpropagation
- Produces per-patch representations of dimension d (1024 for ViT-L, 1280 for ViT-H)

**Target Encoder (ViT with EMA):**
- Same architecture as context encoder
- Parameters updated via Exponential Moving Average of context encoder
- EMA schedule: alpha starts at 0.996 and increases to 1.0 over training
- Processes ONLY the target patches (the masked regions)
- Stop-gradient applied -- no gradients flow through target encoder

**Predictor:**
- Narrow transformer (depth 12, width 384 for ViT-L experiments)
- Takes context representations + learnable mask tokens as input
- Mask tokens are positioned at spatial locations of target patches
- Predicts target representations at masked positions
- Much smaller than encoder -- acts as a bottleneck

### 2.3 Multi-Block Masking Strategy

The masking strategy is critical to I-JEPA's success. It uses large contiguous
blocks rather than random patches:

```
Multi-Block Masking Example (16x16 patch grid):

Context (visible patches marked with '.', masked with ' '):

. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . .         . . . . . . . .
. . . .         . . . .
. . . .         . . . .
. . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . .         . . . .
. . . . . . . .         . . . .
. . . . . . . .         . . . .
. . . . . . . .         . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .

Target blocks (4 blocks, each ~4x4 patches):
  Block 1: rows 2-4, cols 4-7    (upper-left gap)
  Block 2: rows 3-5, cols 12-15  (upper-right gap)
  Block 3: rows 7-10, cols 8-11  (lower-center gap)
  Block 4: (another random block)
```

**Why block masking matters:**
- Random token masking (as in MAE/BEiT) can be solved by local interpolation
- Block masking forces the model to reason about **semantic content**, not texture
- Large blocks require understanding object structure, spatial relationships
- Multiple separate target blocks test different aspects of scene understanding

**Masking hyperparameters:**
- Number of target blocks: 4
- Target block scale: (0.15, 0.2) of image area per block
- Target block aspect ratio: (0.75, 1.5)
- Context ratio: roughly 0.85-0.90 of patches remain visible

### 2.4 Training Details

- **Single-view**: only ONE crop of the image is used (no multi-crop, no augmentations)
- **No augmentations**: no color jitter, no horizontal flip, no rotation
  - This is a major differentiator from contrastive methods
  - The masking provides sufficient training signal
- **Optimizer**: AdamW with weight decay 0.05
- **Learning rate**: 1.5e-3 with cosine schedule and 40-epoch warmup
- **Batch size**: 2048
- **Training epochs**: 600 (ViT-H/14 on ImageNet)
- **Loss**: L2 distance between predicted and target representations
- **EMA schedule**: alpha from 0.996 to 1.0 (cosine schedule)

### 2.5 Performance

ImageNet-1K linear probe evaluation:

```
+------------------+--------+---------------------+
| Model            | Params | Top-1 Accuracy (%)  |
+------------------+--------+---------------------+
| I-JEPA ViT-L/16  | 307M   | 77.3                |
| I-JEPA ViT-H/14  | 632M   | 79.3                |
| I-JEPA ViT-H/16  | 632M   | 78.9                |
+------------------+--------+---------------------+
| MAE ViT-H/14     | 632M   | 76.6                |
| data2vec ViT-L   | 307M   | 76.3                |
| DINO v2 ViT-L    | 307M   | 78.6                |
+------------------+--------+---------------------+
```

Key advantage: I-JEPA achieves competitive performance with significantly less
computation and without any hand-crafted augmentations.

### 2.6 Code Structure (github.com/facebookresearch/ijepa)

```
ijepa/
  src/
    masks/
      multiblock.py      # Multi-block masking generation
      utils.py           # Masking utilities
    models/
      vision_transformer.py  # ViT encoder implementation
    helper.py            # Training helper functions
    transforms.py        # Image transforms (minimal -- no augmentations)
  configs/
    in1k_vith14_ep300.yaml   # ViT-H/14 300-epoch config
    in1k_vitl16_ep600.yaml   # ViT-L/16 600-epoch config
  main_distributed.py    # Distributed training entry point
```

---

## 3. V-JEPA (Video JEPA, 2024)

> Paper: "V-JEPA: Latent Video Prediction for Visual Representation Learning"
> (Bardes et al., 2024)
> Code: github.com/facebookresearch/jepa

### 3.1 Architecture

V-JEPA extends I-JEPA to video by incorporating temporal structure:

```
V-JEPA Architecture
====================

Input Video Clip: T frames x H x W x 3
       |
       v
+------+------+  3D Patchify
| Spatiotemporal|  Patch size: (2, 16, 16) -- 2 frames per temporal token
| Patch Grid    |  Grid: (T/2) x (H/16) x (W/16) tokens
| t x h x w     |
+------+------+
       |
       |  Apply spatiotemporal masking
       |
       v
+------+------+-------+------+------+
| Context     |       | Target       |
| Tokens      |       | Tokens       |
| (visible)   |       | (masked)     |
+------+------+       +------+------+
       |                     |
       v                     v
+------+------+       +------+------+
| Context     |       | Target      |
| Encoder     |       | Encoder     |
| (ViT +      |       | (ViT +      |
|  temporal   |       |  temporal   |
|  pos emb)   |       |  pos emb)   |
+------+------+       +------+------+
       |                     |
       v                     v
+------+------+       +------+------+
| Predictor   |       | Target      |
| + positional| ----> | Repr (s_y)  |
|   mask      |       |             |
|   tokens    |       |             |
+------+------+       +------+------+
       |                     |
       v                     v
   s_y_hat            s_y (stop-grad)
       |                     |
       +-------> L1 <--------+
              (loss)
```

### 3.2 Temporal Position Embeddings

V-JEPA uses factorized spatiotemporal position embeddings:

```
Position embedding for token at (t, h, w):

  pos_emb(t, h, w) = pos_temporal(t) + pos_spatial(h, w)

Where:
  pos_temporal: learnable embeddings of size (T/2, d)
  pos_spatial:  sine-cosine 2D embeddings of size (H/16 * W/16, d)
```

The temporal dimension is learned (not sine-cosine) to allow the model to discover
temporal relationships during training.

### 3.3 Masking Strategies

V-JEPA explores multiple spatiotemporal masking strategies:

**Tube Masking:**
```
Frame 1:    Frame 2:    Frame 3:    Frame 4:
. . X X     . . X X     . . X X     . . X X
. . X X     . . X X     . . X X     . . X X
. . . .     . . . .     . . . .     . . . .
. . . .     . . . .     . . . .     . . . .

Masked region (X) is consistent across ALL frames.
Forces spatial reasoning; temporal information freely available.
```

**Random Spatiotemporal Masking:**
```
Frame 1:    Frame 2:    Frame 3:    Frame 4:
X . . X     . X . .     . . X .     X . . .
. . . .     . X X .     . . . X     . . X .
. X . .     . . . .     X . . .     . . . .
. . . X     X . . .     . . X .     . X . .

Masked tokens are INDEPENDENT across frames.
Forces both spatial AND temporal reasoning.
```

**Block Temporal Masking (used in final V-JEPA):**
```
Frame 1:    Frame 2:    Frame 3:    Frame 4:
. . . .     . . . .     X X X X     X X X X
. . . .     . . . .     X X X X     X X X X
. . . .     . . . .     X X X X     X X X X
. . . .     . . . .     X X X X     X X X X

Entire frames are masked. Context = first half, target = second half.
Or: spatiotemporal blocks spanning contiguous frame ranges.
```

The final V-JEPA uses a combination: large spatiotemporal blocks that span
multiple frames, with multiple target regions per clip.

### 3.4 Training

- **Dataset**: VideoMix2M (2 million video clips from diverse sources)
- **No labels**: entirely self-supervised, no annotation required
- **Clip length**: 16 frames at 224x224 resolution
- **Encoder**: ViT-L/16 with temporal position embeddings
- **Predictor**: 6-layer transformer, width 384
- **Loss**: L1 distance (found slightly better than L2 for video)
- **EMA**: alpha from 0.998 to 1.0
- **Optimizer**: AdamW, LR 1e-3, cosine decay
- **Training**: 90K iterations, batch size 256

### 3.5 Performance Highlights

- Strong performance on video understanding benchmarks (Kinetics-400, SSv2)
- Competitive with supervised pre-training for downstream tasks
- Learns temporal structure without explicit supervision
- Frozen encoder features outperform MAE-based video pre-training

### 3.6 Code Structure (github.com/facebookresearch/jepa)

```
jepa/
  src/
    masks/
      multiblock3d.py    # 3D spatiotemporal masking
      utils.py
    models/
      vision_transformer.py  # ViT with temporal position embeddings
      predictor.py           # Predictor transformer
    datasets/
      video_dataset.py       # Video loading and sampling
    helper.py
  app/
    vjepa/
      train.py               # V-JEPA training loop
      utils.py
  configs/
    pretrain/
      vitl16.yaml            # ViT-L/16 pretraining config
      vith16.yaml            # ViT-H/16 pretraining config
    eval/
      vitl16_k400.yaml       # Kinetics-400 evaluation
```

---

## 4. V-JEPA 2 (June 2025)

> Paper: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction
> and Planning" (Bardes et al., June 2025)
> Code: github.com/facebookresearch/vjepa2

### 4.1 Scale and Training Data

V-JEPA 2 represents a significant scale-up:

- **Parameters**: 1.2 billion (ViT-e architecture)
- **Training data**: VideoMix22M -- 22 million video clips (1M+ hours)
  - Curated from diverse internet video sources
  - No labels or annotations used during pre-training
- **Compute**: trained on hundreds of GPUs for weeks

### 4.2 Key Innovations

**Block-Causal Attention:**

The major architectural innovation in V-JEPA 2 is block-causal attention, which
enables autoregressive prediction of feature blocks:

```
Block-Causal Attention Pattern
================================

Tokens grouped into temporal blocks: B1, B2, B3, B4

Attention mask:
         B1   B2   B3   B4
    B1 [ ATT  ---  ---  --- ]    B1 attends only to B1
    B2 [ ATT  ATT  ---  --- ]    B2 attends to B1, B2
    B3 [ ATT  ATT  ATT  --- ]    B3 attends to B1, B2, B3
    B4 [ ATT  ATT  ATT  ATT ]    B4 attends to B1, B2, B3, B4

ATT = attention allowed
--- = attention masked (causal mask)

This enables the model to predict future feature blocks
autoregressively in latent space, forming a world model.
```

This is distinct from standard causal attention (which operates token-by-token).
Block-causal attention operates on groups of spatiotemporal tokens, corresponding
to short temporal segments. Within each block, full bidirectional attention is
used. Between blocks, causal masking is applied.

**Cooldown Training Phase:**

After the main pre-training phase, V-JEPA 2 undergoes a "cooldown" phase:
- Learning rate is reduced significantly
- Training data quality is increased (curated subset)
- Additional training epochs at low LR stabilize representations
- Similar in concept to the cooldown in Llama training recipes

### 4.3 World Modeling Capabilities

V-JEPA 2's block-causal attention enables it to function as a world model:

```
World Model via Block-Causal Prediction
=========================================

Given:  Frames 1-4 (observed)
Predict: Frames 5-8 (future, in latent space)

  Observed              Predicted (latent)
  +---+---+---+---+    +---+---+---+---+
  | 1 | 2 | 3 | 4 | -> | 5 | 6 | 7 | 8 |
  +---+---+---+---+    +---+---+---+---+
                        (predicted in representation space,
                         NOT pixel space)

The model predicts WHAT will happen semantically,
not the exact pixel appearance.
```

### 4.4 V-JEPA 2-AC (Action-Conditioned)

V-JEPA 2-AC extends the world model to be conditioned on actions:

```
V-JEPA 2-AC Architecture
==========================

  Visual Observation (camera frames)
         |
         v
  +------+------+
  | V-JEPA 2    |
  | Encoder     |
  +------+------+
         |
         v
  +------+------+------+------+
  | Visual      | Action      |
  | Features    | Embeddings  |
  | z_t         | a_t         |
  +------+------+------+------+
         |            |
         +------+-----+
                |
                v
         +------+------+
         | Action-     |
         | Conditioned |
         | Predictor   |
         +------+------+
                |
                v
         +------+------+
         | Predicted   |
         | Future      |
         | Features    |
         | z_t+1       |
         +------+------+
```

**Training data**: 62 hours of Droid robot manipulation data
- Franka Emika Panda robot arms
- Various manipulation tasks (grasping, pick-and-place, pushing)
- Paired (video observation, robot action) sequences

**Action representation**:
- 7-DOF joint positions/velocities
- Gripper state (open/close)
- Projected into the same embedding space as visual features

### 4.5 Robotics: Model Predictive Control

V-JEPA 2-AC enables zero-shot robot control via Model Predictive Control (MPC):

```
MPC Planning Loop
==================

1. Observe current state: encode frame -> z_t

2. Sample N candidate action sequences:
   {a_1, a_2, ..., a_H}  for horizon H

3. For each candidate sequence, roll out the world model:
   z_t -> z_t+1 -> z_t+2 -> ... -> z_t+H
   (using V-JEPA 2-AC predictions)

4. Evaluate each trajectory against goal:
   cost = D(z_t+H, z_goal)

5. Select the best action sequence (lowest cost)

6. Execute FIRST action of best sequence

7. Repeat from step 1

This is zero-shot: no reward function was trained.
The model uses its learned world model to plan.
```

**Results:**
- Zero-shot grasping on Franka arms
- Pick-and-place of novel objects
- No task-specific fine-tuning required
- Competitive with methods trained on 10-100x more robot data

### 4.6 Code Structure (github.com/facebookresearch/vjepa2)

```
vjepa2/
  src/
    models/
      vision_transformer.py    # ViT-e with block-causal attention
      predictor.py             # Block-causal predictor
      action_conditioned.py    # V-JEPA 2-AC model
    masks/
      block_causal.py          # Block-causal masking
    training/
      pretrain.py              # Pre-training loop
      cooldown.py              # Cooldown training phase
    planning/
      mpc.py                   # Model Predictive Control
      cost_functions.py        # Goal-conditioned cost
    datasets/
      videomix22m.py           # VideoMix22M loader
      droid_dataset.py         # Droid robot data loader
  configs/
    pretrain/
      vite_videomix22m.yaml
    action_conditioned/
      vjepa2_ac_droid.yaml
    planning/
      mpc_franka.yaml
  eval/
    video_classification.py
    robot_evaluation.py
```

---

## 5. VL-JEPA (December 2025)

> Paper: "VL-JEPA: Joint Embedding Predictive Architectures for Multimodal
> Understanding" (December 2025)
> Code: github.com/facebookresearch/vljepa (expected)

### 5.1 Overview

VL-JEPA bridges the JEPA paradigm with language understanding. It is the first
JEPA model to jointly handle vision and language, enabling capabilities like
video captioning, visual question answering, and discriminative video-text
matching.

### 5.2 Architecture

```
VL-JEPA Full Architecture
===========================

                      VIDEO INPUT                         TEXT INPUT
                 (T frames, 224x224)                   (token sequence)
                         |                                    |
                         v                                    v
               +---------+---------+                +---------+---------+
               |   X-Encoder       |                |   Y-Encoder       |
               |   (Frozen         |                |   EmbeddingGemma  |
               |    V-JEPA 2       |                |   -300M           |
               |    ViT-L)         |                |   (fine-tuned     |
               |   304M params     |                |    at 0.05x LR)   |
               |   FROZEN          |                |                   |
               +---------+---------+                +---------+---------+
                         |                                    |
                    [N visual                           [text embedding
                     tokens,                             vector, 1536-d]
                     d=1024]                                  |
                         |                                    |
                         v                                    |
               +---------+---------+                          |
               |   Predictor       |                          |
               |   (Llama-3.2-1B   |                          |
               |    last 8 layers) |                          |
               |   490M trainable  |                          |
               |   params          |                          |
               |                   |                          |
               | Input:            |                          |
               |  visual tokens    |                          |
               |  + query tokens   |                          |
               |                   |                          |
               | Attention:        |                          |
               |  bidirectional    |                          |
               |  (causal removed) |                          |
               +---------+---------+                          |
                         |                                    |
                    [query output                             |
                     tokens,                                  |
                     projected                                |
                     to 1536-d]                               |
                         |                                    |
                         v                                    v
               +---------+---------+                +---------+---------+
               |  Video Embedding  |                | Text Embedding    |
               |  (1536-d)         |                | (1536-d)          |
               +---------+---------+                +---------+---------+
                         |                                    |
                         +---------> Bi-directional <---------+
                                     InfoNCE Loss
                                  (shared 1536-d space)
```

### 5.3 Component Details

**X-Encoder: Frozen V-JEPA 2 ViT-L (304M params)**
- Pre-trained V-JEPA 2 vision transformer (Large variant)
- Completely frozen during VL-JEPA training -- no gradient updates
- Processes video frames into spatiotemporal patch tokens
- Output: sequence of visual tokens, each of dimension 1024
- This provides stable, high-quality visual representations learned via
  self-supervised video prediction

**Predictor: Llama-3.2-1B Last 8 Layers (490M trainable params)**
- Initialized from the last 8 transformer layers of Llama-3.2-1B
- Original causal attention mask is removed -- replaced with bidirectional attention
- Takes as input: concatenation of visual tokens + learnable query tokens
- Joint attention over vision and query enables cross-modal reasoning
- Query tokens serve as "slots" that summarize the visual content
- Output query token representations are linearly projected to 1536 dimensions
- Fully trainable during VL-JEPA training

**Y-Encoder: EmbeddingGemma-300M (fine-tuned at 0.05x LR)**
- EmbeddingGemma-300M: a Gemma variant specialized for text embeddings
- Produces a single dense vector (1536-d) representing the input text
- Fine-tuned during training but at 0.05x the base learning rate
  - This low LR preserves the pre-trained text understanding
  - Allows gradual adaptation to the shared embedding space
- The slow fine-tuning is analogous to the EMA target encoder in earlier JEPAs

### 5.4 Loss Function

Bi-directional InfoNCE loss in the shared 1536-d embedding space:

```
L = L_v2t + L_t2v

L_v2t = -1/B * sum_i log( exp(sim(v_i, t_i)/tau) / sum_j exp(sim(v_i, t_j)/tau) )
L_t2v = -1/B * sum_i log( exp(sim(t_i, v_i)/tau) / sum_j exp(sim(t_i, v_j)/tau) )

Where:
  v_i = video embedding (from predictor query tokens)
  t_i = text embedding (from Y-encoder)
  sim(a, b) = cosine_similarity(a, b)
  tau = learned temperature parameter
  B = batch size
```

### 5.5 Training

**Two-stage training process:**

```
Stage 1: Pre-training
+--------------------------------------------------+
| Dataset:    Large-scale video-text pairs          |
| LR:        Constant learning rate                 |
| Duration:  Majority of training compute           |
| Objective: Bi-directional InfoNCE                 |
+--------------------------------------------------+

Stage 2: Supervised Fine-Tuning (SFT)
+--------------------------------------------------+
| Dataset:    Curated, higher-quality pairs         |
| LR:        Cosine annealing (decaying)            |
| Duration:  Shorter phase                          |
| Objective: Same InfoNCE + task-specific data      |
+--------------------------------------------------+
```

### 5.6 Inference Modes

VL-JEPA supports three distinct inference modes:

**Mode 1: Captioning**
```
Video -> X-Encoder -> Predictor (with query tokens)
                                      |
                                      v
                               Video embedding (1536-d)
                                      |
                                      v
                               Nearest-neighbor search
                               in text embedding space
                                      |
                                      v
                               Retrieved caption
```

**Mode 2: Discriminative VQA**
```
Video + Question -> encode separately
                         |
        +----------------+----------------+
        |                                 |
  Video embedding                  Candidate answer
  (via predictor)                  embeddings (via Y-encoder)
        |                                 |
        +-------> cosine similarity <-----+
                  select max
                       |
                       v
                  Best answer
```

**Mode 3: Selective Decoding**
```
Video -> X-Encoder -> Predictor -> Query outputs (many tokens)
                                        |
                                   Ward clustering
                                   (agglomerative)
                                        |
                                   Select representative
                                   tokens per cluster
                                        |
                                   Decode only selected
                                   tokens (2.85x speedup)
```

### 5.7 Selective Decoding via Ward Clustering

The selective decoding algorithm is a key efficiency innovation:

1. Run the predictor to get all query output token representations
2. Apply Ward agglomerative clustering to group similar tokens
3. Select one representative token per cluster (centroid-nearest)
4. Decode only the selected tokens, discarding redundant ones
5. Achieves 2.85x inference speedup with minimal quality loss

This works because many query tokens carry redundant information about the same
visual concept. Clustering identifies the unique "information slots" and avoids
wasting compute on repetitive decoding.

### 5.8 Key Results

```
Video Understanding:
+------------------+--------------+-----------+
| Model            | Video Retrieval (R@1)    |
+------------------+--------------+-----------+
| CLIP ViT-L       |    42.1                  |
| SigLIP2           |    45.3                  |
| VL-JEPA          |    48.7 (surpasses both) |
+------------------+--------------+-----------+

Visual Question Answering:
+------------------+------------+
| Model            | VQA Score  |
+------------------+------------+
| InstructBLIP      |   67.2     |
| Qwen-VL           |   68.1     |
| VL-JEPA          |   67.8     |
+------------------+------------+
VL-JEPA matches dedicated VQA models despite being a
joint-embedding model, not a generative one.
```

---

## 6. H-JEPA (Hierarchical JEPA)

> Status: Conceptual framework from LeCun's 2022 paper; not yet fully implemented
> as a standalone system.

### 6.1 Concept

H-JEPA extends the JEPA concept to multiple levels of temporal and spatial
abstraction. The core idea is that intelligent systems need predictions at
multiple time scales:

```
H-JEPA Hierarchical Prediction
================================

Level 3 (high abstraction, long horizon):
  "Person will leave the room"
  Prediction horizon: minutes to hours
  +--[Abstract Goal State]--+

Level 2 (medium abstraction):
  "Person walks toward door, reaches for handle"
  Prediction horizon: seconds
  +--[Action Sequence State]--+

Level 1 (low abstraction, short horizon):
  "Hand moves 3cm to the right"
  Prediction horizon: milliseconds to frames
  +--[Detailed Motion State]--+

Each level:
  - Has its own encoder and predictor
  - Operates at a different temporal resolution
  - Higher levels provide context/goals to lower levels
  - Lower levels provide grounding to higher levels
```

### 6.2 Architecture (Proposed)

```
H-JEPA Proposed Architecture
==============================

Input Sequence: x_1, x_2, ..., x_T
       |
       v
+------+------+
| Level 1     |  Fine-grained, short-horizon
| Encoder     |  Temporal resolution: every frame
| + Predictor |  Predicts next 1-2 frames
+------+------+
       |
       | Pooled / downsampled representations
       v
+------+------+
| Level 2     |  Medium-grained, medium-horizon
| Encoder     |  Temporal resolution: every ~10 frames
| + Predictor |  Predicts next ~1 second
+------+------+
       |
       | Further pooled representations
       v
+------+------+
| Level 3     |  Coarse-grained, long-horizon
| Encoder     |  Temporal resolution: every ~100 frames
| + Predictor |  Predicts next ~10 seconds
+------+------+

Top-down modulation:
  Level 3 -> Level 2 -> Level 1
  (goals)    (plans)    (actions)

Bottom-up information:
  Level 1 -> Level 2 -> Level 3
  (details)  (events)   (context)
```

### 6.3 Relationship to Other JEPAs

- **I-JEPA** = single-level spatial JEPA
- **V-JEPA** = single-level spatiotemporal JEPA
- **V-JEPA 2** = single-level spatiotemporal JEPA with block-causal prediction
  (a step toward hierarchical via temporal blocks)
- **H-JEPA** = full multi-level hierarchical JEPA (the end goal)

### 6.4 Current Status

H-JEPA remains primarily a conceptual framework as described in LeCun's 2022
position paper. Elements of it appear in:

- V-JEPA 2's block-causal attention (temporal hierarchy over blocks)
- VL-JEPA's multi-modal prediction (abstraction via language)
- Various research prototypes in the academic community

A full H-JEPA implementation with learned multi-level abstractions, top-down
modulation, and configurable prediction horizons has not yet been publicly
released as of early 2026.

---

## 7. Comparison Across JEPA Variants

```
+----------------+--------+--------+--------+---------+---------+---------+
|                | I-JEPA | V-JEPA |V-JEPA 2|VL-JEPA  |V-JEPA   | H-JEPA  |
|                | (2023) | (2024) | (2025) | (2025)  | 2-AC    | (future)|
+----------------+--------+--------+--------+---------+---------+---------+
| Modality       | Image  | Video  | Video  | Video + | Video + | Multi-  |
|                |        |        |        | Language| Action  | level   |
+----------------+--------+--------+--------+---------+---------+---------+
| Encoder        | ViT    | ViT +  | ViT-e +| V-JEPA2 | V-JEPA2 | Hier.   |
|                |        | temp   | block- | ViT-L + | + action| encoders|
|                |        | pos    | causal | Gemma   | encoder |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Params         | 307M-  | ~307M  | 1.2B   | ~1.1B   | ~1.3B   | TBD     |
|                | 632M   |        |        | total   |         |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Predictor      | Narrow | 6-layer| Block- | Llama   | Action- | Multi-  |
|                | ViT    | xformer| causal | 3.2-1B  | cond.   | level   |
|                | (12L,  |        | xformer| last 8L | pred.   | pred.   |
|                |  384d) |        |        | (490M)  |         |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Target Encoder | ViT    | ViT    | ViT    |Embedding| ViT     | Hier.   |
|                | (EMA)  | (EMA)  | (EMA)  | Gemma   | (EMA)   | (EMA)   |
|                |        |        |        | -300M   |         |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Loss           | L2     | L1     | L1/L2  | Bi-dir  | L1/L2   | TBD     |
|                |        |        |        | InfoNCE |         |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Masking        | Multi- | Spatio-| Block- | N/A     | Block-  | Multi-  |
|                | block  | temp   | causal | (query  | causal  | scale   |
|                | spatial| blocks | blocks | tokens) |         |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Augmentations  | None   | None   | None   | None    | None    | None    |
+----------------+--------+--------+--------+---------+---------+---------+
| Training data  | IN-1K  | VMix2M | VMix22M| Video-  | Droid   | TBD     |
|                |        |        |        | text    | (62hrs) |         |
+----------------+--------+--------+--------+---------+---------+---------+
| Downstream     | Image  | Video  | Video +| Video   | Robot   | General |
| tasks          | classif| classif| robot  | caption,| grasp,  | intel.  |
|                |        |        | control| VQA,    | pick-   |         |
|                |        |        |        | retriev.| place   |         |
+----------------+--------+--------+--------+---------+---------+---------+
| World model    | No     | No     | Yes    | Partial | Yes     | Yes     |
| capable        |        |        | (block-| (via    | (action-| (multi- |
|                |        |        | causal)| language)| cond.)  | level)  |
+----------------+--------+--------+--------+---------+---------+---------+
| Planning       | No     | No     | Via    | Via     | Via MPC | Via     |
| capable        |        |        | MPC    | text    |         | hier.   |
|                |        |        |        | grounding|        | planning|
+----------------+--------+--------+--------+---------+---------+---------+
```

### Evolution Timeline

```
2022 Jun   LeCun's "Path Towards Autonomous Machine Intelligence"
           -> JEPA concept, H-JEPA vision
              |
2023 Apr   I-JEPA (CVPR 2023)
           -> First concrete JEPA: images, multi-block masking
              |
2024 Feb   V-JEPA
           -> Extension to video: spatiotemporal masking
              |
2025 Jun   V-JEPA 2
           -> Scale (1.2B), block-causal attention, world model
           -> V-JEPA 2-AC: action-conditioned, robot control
              |
2025 Dec   VL-JEPA
           -> Vision + language: InfoNCE, captioning, VQA
              -> Selective decoding, Llama predictor
              |
2026+      H-JEPA (ongoing research)
           -> Full hierarchical multi-level abstraction
```

### Key Takeaways

1. **The JEPA family is converging toward world models.** From static image
   understanding (I-JEPA) through temporal prediction (V-JEPA, V-JEPA 2) to
   action-conditioned planning (V-JEPA 2-AC) and language grounding (VL-JEPA),
   each step builds toward LeCun's vision of autonomous machine intelligence.

2. **Latent prediction is the unifying principle.** Every JEPA variant predicts
   in a learned representation space, not in pixel/token space. This is what
   enables efficiency, abstraction, and planning.

3. **No augmentations needed.** Unlike contrastive methods, JEPAs learn from
   masking/prediction alone. This removes a major source of human prior injection
   and makes the approach more principled.

4. **EMA target encoders persist across the family.** The asymmetric
   encoder-predictor-target architecture with EMA updates remains the core
   anti-collapse mechanism, even as specific implementations vary.

5. **The predictor is becoming more powerful.** From a narrow 12-layer
   transformer in I-JEPA to Llama-3.2-1B layers in VL-JEPA, the predictor
   is taking on more of the computational burden, enabling richer predictions.

---

*End of JEPA Family Overview*
