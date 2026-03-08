# VL-JEPA: Technical Deep-Dive

> Implementation reference for the VL-JEPA Research Lab
> Last updated: March 2026
>
> This document is a detailed technical reference for VL-JEPA, written for
> someone implementing the architecture from scratch. Every tensor shape,
> hyperparameter, and design decision is documented.

---

## Table of Contents

1.  [Architecture Overview](#1-architecture-overview)
2.  [Complete Forward Pass Walkthrough](#2-complete-forward-pass-walkthrough)
3.  [X-Encoder: Frozen V-JEPA 2 ViT-L](#3-x-encoder-frozen-v-jepa-2-vit-l)
4.  [Predictor: Llama-3.2-1B Last 8 Layers](#4-predictor-llama-32-1b-last-8-layers)
5.  [Y-Encoder: EmbeddingGemma-300M](#5-y-encoder-embeddinggemma-300m)
6.  [Loss Function: Bi-Directional InfoNCE](#6-loss-function-bi-directional-infonce)
7.  [Training Procedure](#7-training-procedure)
8.  [Inference Modes](#8-inference-modes)
9.  [Selective Decoding Algorithm](#9-selective-decoding-algorithm)
10. [Implementation Considerations](#10-implementation-considerations)
11. [Key Hyperparameters Table](#11-key-hyperparameters-table)

---

## 1. Architecture Overview

VL-JEPA consists of three major components that operate in a shared 1536-dimensional
embedding space:

```
VL-JEPA System Diagram
========================

+================================================================+
|                        TRAINING                                 |
|                                                                 |
|  VIDEO                                TEXT                      |
|  (B, T, 3, 224, 224)                 (B, L_text)               |
|       |                                    |                    |
|       v                                    v                    |
|  +---------+                          +---------+               |
|  |X-Encoder|                          |Y-Encoder|               |
|  | V-JEPA2 |                          |Embedding|               |
|  | ViT-L   |                          | Gemma   |               |
|  | 304M    |                          | 300M    |               |
|  | FROZEN  |                          | 0.05xLR |               |
|  +----+----+                          +----+----+               |
|       |                                    |                    |
|  (B, N_vis, 1024)                    (B, 1536)                  |
|       |                                    |                    |
|       v                                    |                    |
|  +---------+                               |                    |
|  |Predictor|                               |                    |
|  |Llama 8L |                               |                    |
|  | 490M    |                               |                    |
|  |TRAINABLE|                               |                    |
|  +----+----+                               |                    |
|       |                                    |                    |
|  (B, N_q, 1536)                            |                    |
|       |                                    |                    |
|       v                                    v                    |
|  +---------+    Bi-directional        +---------+               |
|  | Video   | <--- InfoNCE Loss --->   |  Text   |               |
|  | Embed   |                          |  Embed  |               |
|  | (1536-d)|                          | (1536-d)|               |
|  +---------+                          +---------+               |
+================================================================+
```

**Parameter budget:**

```
Component            | Parameters | Trainable | Notes
---------------------|------------|-----------|------------------------
X-Encoder (ViT-L)   | 304M       | 0         | Completely frozen
Predictor (Llama 8L) | 490M       | 490M      | Fully trainable
Y-Encoder (Gemma)    | 300M       | 300M      | Fine-tuned at 0.05x LR
Linear projections   | ~6M        | ~6M       | Input/output projections
Query tokens         | ~0.1M     | ~0.1M     | Learnable parameters
---------------------|------------|-----------|------------------------
Total                | ~1.1B      | ~796M     |
```

---

## 2. Complete Forward Pass Walkthrough

This section traces the complete forward pass from raw video and text input
through to the loss computation, noting tensor shapes at every step.

### Notation

```
B       = batch size (e.g., 256)
T       = number of video frames (e.g., 16)
H, W    = frame spatial dimensions (224, 224)
P       = patch size (16)
t_p     = temporal patch size (2 frames per temporal token)
h       = H / P = 14
w       = W / P = 14
t       = T / t_p = 8
N_vis   = t * h * w = 8 * 14 * 14 = 1568 (total visual tokens)
d_enc   = encoder hidden dimension (1024 for ViT-L)
d_pred  = predictor hidden dimension (2048 for Llama-3.2-1B)
d_emb   = shared embedding dimension (1536)
N_q     = number of query tokens (e.g., 64 or 128)
L_text  = text sequence length (variable, padded to max)
d_text  = Y-encoder hidden dimension (1536 for EmbeddingGemma-300M)
```

### Step-by-Step Forward Pass

```
STEP 1: Video Patch Embedding (inside X-Encoder)
=================================================

Input:  video frames
        Shape: (B, T, 3, H, W) = (B, 16, 3, 224, 224)

Reshape to 3D patches:
        (B, T/t_p, t_p, 3, H/P, P, W/P, P)
        = (B, 8, 2, 3, 14, 16, 14, 16)

Flatten each 3D patch and project:
        patch_embed: Linear(t_p * P * P * 3, d_enc)
        = Linear(2 * 16 * 16 * 3, 1024) = Linear(1536, 1024)

        Shape: (B, t, h, w, d_enc)
        = (B, 8, 14, 14, 1024)

Flatten spatial-temporal grid:
        Shape: (B, N_vis, d_enc)
        = (B, 1568, 1024)


STEP 2: Add Positional Embeddings (inside X-Encoder)
=====================================================

Factorized spatiotemporal position embeddings:

  pos_spatial:  sine-cosine 2D, shape (h*w, d_enc) = (196, 1024)
  pos_temporal: learned, shape (t, d_enc) = (8, 1024)

  pos(t_i, h_j, w_k) = pos_temporal[t_i] + pos_spatial[h_j * w + w_k]

  Broadcast and add:
  tokens = tokens + pos_embeddings
  Shape: (B, 1568, 1024)


STEP 3: X-Encoder Transformer Blocks (inside X-Encoder)
========================================================

24 transformer blocks (ViT-L configuration):
  Each block:
    LayerNorm -> MultiHeadSelfAttention(heads=16) -> Residual
    LayerNorm -> MLP(1024 -> 4096 -> 1024)        -> Residual

  Input:  (B, 1568, 1024)
  Output: (B, 1568, 1024)    [same shape, refined representations]

  FROZEN: no gradients flow through these layers.


STEP 4: Project Visual Tokens for Predictor Input
==================================================

Linear projection from encoder space to predictor space:
  proj_in: Linear(d_enc, d_pred) = Linear(1024, 2048)

  Input:  (B, 1568, 1024)
  Output: (B, 1568, 2048)


STEP 5: Prepare Query Tokens
=============================

Learnable query tokens:
  query_tokens: Parameter(N_q, d_pred) = Parameter(64, 2048)

  Expand across batch:
  Shape: (B, N_q, d_pred) = (B, 64, 2048)


STEP 6: Concatenate Visual Tokens + Query Tokens
=================================================

Concatenate along sequence dimension:
  predictor_input = cat([visual_tokens, query_tokens], dim=1)

  Shape: (B, N_vis + N_q, d_pred)
  = (B, 1568 + 64, 2048)
  = (B, 1632, 2048)


STEP 7: Predictor Transformer Blocks (Llama-3.2-1B last 8 layers)
==================================================================

8 transformer blocks from Llama-3.2-1B:
  Each block:
    RMSNorm -> GroupedQueryAttention(heads=32, kv_heads=8) -> Residual
    RMSNorm -> SwiGLU-MLP(2048 -> 8192 -> 2048)           -> Residual

  IMPORTANT: Causal attention mask is REMOVED.
  All tokens attend to all other tokens (bidirectional).

  Attention pattern:
    Visual tokens <-> Visual tokens    (self-attention over vision)
    Visual tokens <-> Query tokens     (cross-attention: query reads vision)
    Query tokens  <-> Query tokens     (self-attention over queries)
    Query tokens  <-> Visual tokens    (cross-attention: vision reads query)

  Input:  (B, 1632, 2048)
  Output: (B, 1632, 2048)


STEP 8: Extract Query Token Outputs
====================================

Select only the query token positions from the predictor output:

  query_output = predictor_output[:, N_vis:, :]
  Shape: (B, N_q, d_pred) = (B, 64, 2048)


STEP 9: Project Query Outputs to Shared Embedding Space
========================================================

Linear projection + L2 normalization:
  proj_out: Linear(d_pred, d_emb) = Linear(2048, 1536)

  video_emb = proj_out(query_output)
  video_emb = L2_normalize(video_emb, dim=-1)

  Shape: (B, N_q, d_emb) = (B, 64, 1536)

Pool query tokens to a single video embedding:
  video_emb_pooled = mean(video_emb, dim=1)
  video_emb_pooled = L2_normalize(video_emb_pooled, dim=-1)

  Shape: (B, d_emb) = (B, 1536)


STEP 10: Y-Encoder Text Processing
====================================

Input: tokenized text
  Shape: (B, L_text)

EmbeddingGemma-300M forward pass:
  Token embedding -> 18 transformer layers -> pooled output

  Pooling: either [EOS] token representation or mean pooling
  Followed by output projection layer (built into EmbeddingGemma)

  text_emb = Y_encoder(text_tokens)
  text_emb = L2_normalize(text_emb, dim=-1)

  Shape: (B, d_emb) = (B, 1536)


STEP 11: Compute Bi-Directional InfoNCE Loss
=============================================

Compute similarity matrix:
  sim = video_emb_pooled @ text_emb.T  (matmul)
  sim = sim / tau                        (temperature scaling)

  Shape: (B, B)

Video-to-text loss:
  L_v2t = CrossEntropy(sim, targets=arange(B))

Text-to-video loss:
  L_t2v = CrossEntropy(sim.T, targets=arange(B))

Total loss:
  L = (L_v2t + L_t2v) / 2
```

### Forward Pass Summary Diagram

```
Video (B,16,3,224,224)          Text (B, L_text)
       |                              |
  [3D Patchify]                  [Tokenize]
       |                              |
  (B, 1568, 1536)                (B, L_text)
       |                              |
  [Linear 1536->1024]                |
       |                              |
  (B, 1568, 1024)                    |
       |                              |
  [+ pos embeddings]                 |
       |                              |
  (B, 1568, 1024)                    |
       |                              |
  [ViT-L x24 blocks] FROZEN         |
       |                              |
  (B, 1568, 1024)                    |
       |                              |
  [Linear 1024->2048]                |
       |                              |
  (B, 1568, 2048)                    |
       |                              |
  [+ query tokens]                   |
       |                              |
  (B, 1632, 2048)                    |
       |                              |
  [Llama 8L bidir] TRAINABLE    [EmbeddingGemma-300M] 0.05x LR
       |                              |
  (B, 1632, 2048)                (B, 1536)
       |                              |
  [Extract queries]                   |
       |                              |
  (B, 64, 2048)                      |
       |                              |
  [Linear 2048->1536]                |
       |                              |
  (B, 64, 1536)                      |
       |                              |
  [Mean pool + L2 norm]          [L2 norm]
       |                              |
  (B, 1536)                     (B, 1536)
       |                              |
       +----> InfoNCE Loss <----------+
```

---

## 3. X-Encoder: Frozen V-JEPA 2 ViT-L

### 3.1 Architecture Specification

The X-Encoder is a Vision Transformer Large (ViT-L) pre-trained with the V-JEPA 2
self-supervised objective. It is completely frozen during VL-JEPA training.

```
ViT-L Configuration
=====================

Parameter               | Value
------------------------|------------------
Hidden dimension (d)    | 1024
MLP dimension           | 4096  (4x hidden)
Number of heads         | 16
Head dimension          | 64    (d / heads)
Number of layers        | 24
Patch size (spatial)    | 16 x 16
Patch size (temporal)   | 2 frames
Input resolution        | 224 x 224
Frames per clip         | 16
Total parameters        | 304M
Activation              | GELU
Normalization           | LayerNorm (pre-norm)
Position embeddings     | Factorized spatiotemporal
```

### 3.2 Patch Embedding

The 3D patch embedding converts raw video frames into a sequence of tokens:

```
3D Patch Embedding
===================

Input video: (B, T, C, H, W) = (B, 16, 3, 224, 224)

Step 1: Reshape into 3D patches
  (B, T/t_p, t_p, C, H/P, P, W/P, P)
  = (B, 8, 2, 3, 14, 16, 14, 16)

Step 2: Flatten each patch
  Each patch: (t_p * C * P * P) = (2 * 3 * 16 * 16) = 1536 values

Step 3: Linear projection
  Linear(1536, 1024)  -- projects each patch to d_enc

  Output: (B, 8, 14, 14, 1024)

Step 4: Flatten spatiotemporal grid
  (B, 8*14*14, 1024) = (B, 1568, 1024)

Implementation note: This is typically done with a Conv3d layer:
  Conv3d(in_channels=3, out_channels=1024,
         kernel_size=(2, 16, 16), stride=(2, 16, 16))
  Input:  (B, 3, 16, 224, 224)  [channels first]
  Output: (B, 1024, 8, 14, 14)
  Reshape: (B, 1024, 1568) -> transpose -> (B, 1568, 1024)
```

### 3.3 Positional Encoding

```
Factorized Spatiotemporal Position Embeddings
===============================================

Spatial (sine-cosine, fixed):
  pos_spatial: (196, 1024)
  Generated using 2D sine-cosine encoding for a 14x14 grid
  Split: first 512 dims for height, last 512 dims for width
  Each uses interleaved sin/cos at different frequencies

Temporal (learned):
  pos_temporal: Parameter(8, 1024)
  Initialized with truncated normal (std=0.02)
  Learned during V-JEPA 2 pre-training (frozen in VL-JEPA)

Combined:
  For token at position (t_i, h_j, w_k):
    pos[t_i, h_j, w_k] = pos_temporal[t_i] + pos_spatial[h_j * 14 + w_k]

  Total positions: 8 * 196 = 1568
  Each position: 1024-dimensional vector
```

### 3.4 Transformer Blocks

Each of the 24 ViT-L transformer blocks:

```
Transformer Block (Pre-Norm)
==============================

Input: x, shape (B, N, 1024)

  # Self-attention
  x_norm = LayerNorm(x)                         # (B, N, 1024)
  q = W_q(x_norm)                               # (B, N, 1024)
  k = W_k(x_norm)                               # (B, N, 1024)
  v = W_v(x_norm)                               # (B, N, 1024)

  # Reshape for multi-head attention
  q = reshape(q, (B, N, 16, 64))                # 16 heads, dim 64 each
  k = reshape(k, (B, N, 16, 64))
  v = reshape(v, (B, N, 16, 64))

  attn = softmax(q @ k.T / sqrt(64)) @ v        # (B, 16, N, 64)
  attn = reshape(attn, (B, N, 1024))
  attn = W_out(attn)                             # (B, N, 1024)
  x = x + attn                                  # Residual connection

  # MLP
  x_norm = LayerNorm(x)                         # (B, N, 1024)
  mlp = GELU(W1(x_norm))                        # (B, N, 4096)
  mlp = W2(mlp)                                 # (B, N, 1024)
  x = x + mlp                                   # Residual connection

Output: x, shape (B, N, 1024)
```

### 3.5 Why Frozen?

The X-Encoder is frozen for several reasons:

1. **Representation quality**: V-JEPA 2 pre-training produces high-quality visual
   representations. Fine-tuning could degrade them, especially early in VL-JEPA
   training when gradients from the InfoNCE loss are noisy.

2. **Memory efficiency**: Freezing the 304M-parameter encoder eliminates the need
   to store optimizer states and gradients for those parameters. This saves
   approximately 304M * 8 bytes (Adam states) = ~2.4 GB per GPU.

3. **Training stability**: A frozen encoder provides a stable feature extraction
   backbone. The predictor and Y-encoder can adapt to produce compatible
   embeddings without the moving target of a fine-tuning encoder.

4. **Analogous to EMA target**: In the original JEPA formulation, the target
   encoder is updated slowly (EMA). A frozen encoder is the extreme case --
   zero update rate. The Y-encoder with 0.05x LR serves the "slowly adapting
   target" role instead.

---

## 4. Predictor: Llama-3.2-1B Last 8 Layers

### 4.1 Extraction from Llama-3.2-1B

Llama-3.2-1B has 16 transformer layers total. VL-JEPA uses the **last 8 layers**
(layers 8-15, zero-indexed):

```
Llama-3.2-1B Full Architecture (16 layers)
============================================

Layer 0  |  [Discarded]
Layer 1  |  [Discarded]
Layer 2  |  [Discarded]
Layer 3  |  [Discarded]
Layer 4  |  [Discarded]
Layer 5  |  [Discarded]
Layer 6  |  [Discarded]
Layer 7  |  [Discarded]
---------|---------------------------
Layer 8  |  [USED] -> Predictor Layer 0
Layer 9  |  [USED] -> Predictor Layer 1
Layer 10 |  [USED] -> Predictor Layer 2
Layer 11 |  [USED] -> Predictor Layer 3
Layer 12 |  [USED] -> Predictor Layer 4
Layer 13 |  [USED] -> Predictor Layer 5
Layer 14 |  [USED] -> Predictor Layer 6
Layer 15 |  [USED] -> Predictor Layer 7

Rationale: Later layers in LLMs capture higher-level abstractions
and are better suited for cross-modal reasoning tasks.
```

### 4.2 Llama Block Architecture

Each Llama-3.2-1B layer has this structure:

```
Llama Transformer Block
=========================

Parameter                     | Value
------------------------------|------------------
Hidden dimension              | 2048
MLP intermediate dimension    | 8192 (via SwiGLU)
Number of attention heads     | 32
Number of KV heads            | 8  (Grouped Query Attention)
Head dimension                | 64 (2048 / 32)
Normalization                 | RMSNorm (pre-norm)
Activation                    | SwiGLU (SiLU gate)
Position encoding             | RoPE (Rotary Position Embeddings)
Vocabulary size               | N/A (embedding layer discarded)
```

```
Llama Block Detail
===================

Input: x, shape (B, S, 2048)   where S = N_vis + N_q = 1632

  # Grouped Query Attention (GQA)
  x_norm = RMSNorm(x)                            # (B, S, 2048)

  q = W_q(x_norm)  -> (B, S, 32, 64)             # 32 query heads
  k = W_k(x_norm)  -> (B, S, 8, 64)              # 8 key heads
  v = W_v(x_norm)  -> (B, S, 8, 64)              # 8 value heads

  # GQA: each KV head serves 4 query heads (32/8 = 4)
  # Expand KV: (B, S, 8, 64) -> (B, S, 32, 64)
  k = repeat_interleave(k, repeats=4, dim=2)
  v = repeat_interleave(v, repeats=4, dim=2)

  # Apply RoPE to q and k
  q, k = apply_rotary_embeddings(q, k, freqs)

  # IMPORTANT: NO causal mask applied
  # Standard JEPA uses bidirectional attention here
  attn = softmax(q @ k.T / sqrt(64)) @ v          # (B, 32, S, 64)
  attn = reshape(attn, (B, S, 2048))
  attn = W_o(attn)                                 # (B, S, 2048)
  x = x + attn

  # SwiGLU MLP
  x_norm = RMSNorm(x)                             # (B, S, 2048)
  gate = SiLU(W_gate(x_norm))                     # (B, S, 8192)
  up   = W_up(x_norm)                             # (B, S, 8192)
  mlp  = gate * up                                # Element-wise (SwiGLU)
  mlp  = W_down(mlp)                              # (B, S, 2048)
  x = x + mlp

Output: x, shape (B, S, 2048)
```

### 4.3 Modifications from Original Llama

**Causal mask removal:**
```
Original Llama (autoregressive):
  Attention mask:
    [1, 0, 0, 0]    Token 1 sees only itself
    [1, 1, 0, 0]    Token 2 sees tokens 1-2
    [1, 1, 1, 0]    Token 3 sees tokens 1-3
    [1, 1, 1, 1]    Token 4 sees tokens 1-4

VL-JEPA Predictor (bidirectional):
  Attention mask:
    [1, 1, 1, 1]    Token 1 sees all tokens
    [1, 1, 1, 1]    Token 2 sees all tokens
    [1, 1, 1, 1]    Token 3 sees all tokens
    [1, 1, 1, 1]    Token 4 sees all tokens

Implementation: simply remove the causal_mask argument from the
attention computation, or pass an all-ones mask.
```

**RoPE adaptation:**

RoPE (Rotary Position Embeddings) was designed for 1D sequences. In VL-JEPA, the
input is a mix of 2D spatial tokens and learnable query tokens. The position IDs
are assigned sequentially across the concatenated sequence:

```
Position IDs:  [0, 1, 2, ..., 1567, 1568, 1569, ..., 1631]
                |<-- visual tokens -->|  |<-- query tokens -->|
                     (1568 tokens)            (64 tokens)

Note: The visual tokens already have spatial structure from the ViT
positional embeddings. The RoPE here provides additional sequential
position information within the predictor.
```

### 4.4 Input/Output Projections

```
Input Projection (encoder space -> predictor space):
  proj_in: Linear(1024, 2048)  with bias
  Applied to visual tokens from X-Encoder output

  visual_tokens: (B, 1568, 1024) -> (B, 1568, 2048)

Output Projection (predictor space -> embedding space):
  proj_out: Linear(2048, 1536)  with bias
  Applied to query token outputs from predictor

  query_outputs: (B, 64, 2048) -> (B, 64, 1536)
```

### 4.5 Query Tokens

```
Query Token Design
===================

query_tokens: nn.Parameter(N_q, d_pred) = Parameter(64, 2048)
Initialization: truncated normal with std = 0.02

Purpose:
  - Act as "information slots" that aggregate visual information
  - Analogous to the [CLS] token but with more capacity (64 slots)
  - Each query token can specialize to different visual aspects
  - After predictor processing, they contain cross-attended visual info

During forward pass:
  1. Expand: (64, 2048) -> (B, 64, 2048)  via broadcast
  2. Concatenate with visual tokens: (B, 1632, 2048)
  3. Process through 8 Llama layers with bidirectional attention
  4. Extract query positions: (B, 64, 2048)
  5. Project to embedding space: (B, 64, 1536)
  6. Pool to single vector: (B, 1536)
```

### 4.6 Parameter Count Breakdown

```
Per Llama layer:
  W_q:     2048 * 2048 = 4,194,304
  W_k:     2048 *  512 =   1,048,576
  W_v:     2048 *  512 =   1,048,576
  W_o:     2048 * 2048 = 4,194,304
  W_gate:  2048 * 8192 = 16,777,216
  W_up:    2048 * 8192 = 16,777,216
  W_down:  8192 * 2048 = 16,777,216
  RMSNorm (x2):          2 * 2048 = 4,096
  -----------------------------------------
  Per layer total:        ~60.8M

8 layers: ~486M
Projections + query tokens: ~4M
Total predictor: ~490M trainable parameters
```

---

## 5. Y-Encoder: EmbeddingGemma-300M

### 5.1 Architecture

EmbeddingGemma-300M is a Gemma model variant specifically designed for producing
dense text embeddings. It is a decoder-only transformer repurposed for embedding
generation.

```
EmbeddingGemma-300M Configuration
===================================

Parameter                  | Value
---------------------------|------------------
Hidden dimension           | 1536
MLP intermediate dimension | 6144
Number of heads            | 12
Head dimension             | 128  (1536 / 12)
Number of layers           | 18
Vocabulary size            | 256,000 (Gemma tokenizer)
Max sequence length        | 2048
Normalization              | RMSNorm
Activation                 | GELU
Total parameters           | ~300M
Output dimension           | 1536
```

### 5.2 How It Produces Text Embeddings

```
EmbeddingGemma Embedding Process
==================================

Input: text string (e.g., "A dog playing in a park")

Step 1: Tokenize
  tokens = gemma_tokenizer(text)
  Shape: (L_text,)  e.g., (8,)

Step 2: Token embedding lookup
  token_emb = embedding_table[tokens]
  Shape: (L_text, 1536)

Step 3: Forward through 18 transformer layers
  Each layer: RMSNorm -> MHA -> Residual -> RMSNorm -> MLP -> Residual
  Output: (L_text, 1536)

Step 4: Pooling
  Option A -- Last token pooling:
    text_emb = output[:, -1, :]     Shape: (1536,)
  Option B -- Mean pooling:
    text_emb = mean(output, dim=0)  Shape: (1536,)

  EmbeddingGemma typically uses last-token pooling with a special
  [EOS] token appended to mark the end of the sequence.

Step 5: Output projection (built-in)
  text_emb = output_proj(text_emb)
  Shape: (1536,)  -- already in the target dimension

Step 6: L2 normalization
  text_emb = text_emb / ||text_emb||_2
  Shape: (1536,)  -- unit norm
```

### 5.3 Fine-Tuning Strategy

The Y-Encoder is fine-tuned with a reduced learning rate:

```
Y-Encoder Learning Rate Strategy
==================================

Base learning rate (predictor, projections): LR_base
Y-Encoder learning rate:                    0.05 * LR_base

Example:
  If LR_base = 1e-4
  Then LR_y_encoder = 5e-6

This achieves two goals:

1. Preserves pre-trained text understanding:
   EmbeddingGemma was trained on massive text corpora.
   Aggressive fine-tuning would destroy this knowledge.

2. Gradually adapts to shared embedding space:
   The text embeddings slowly adjust to be compatible
   with the video embeddings from the predictor.

Analogy to EMA:
   In I-JEPA/V-JEPA, the target encoder uses EMA (slow update).
   Here, the Y-encoder uses low LR (slow update via gradient).
   Both serve the same purpose: stable, slowly-evolving target.
```

### 5.4 Why EmbeddingGemma-300M?

Design choices and alternatives:

```
+-------------------+--------+----------+----------------------------------+
| Text Encoder      | Params | Emb Dim  | Reason for / against             |
+-------------------+--------+----------+----------------------------------+
| CLIP text enc     | 63M    | 768      | Too small, limited text capacity  |
| BERT-base         | 110M   | 768      | Encoder-only, limited vocabulary  |
| BERT-large        | 340M   | 1024     | Reasonable but older architecture |
| SentenceT5-large  | 335M   | 768      | Good embeddings but lower dim     |
| EmbeddingGemma    | 300M   | 1536     | Best: modern, high-dim, designed  |
| -300M             |        |          | for dense embeddings             |
+-------------------+--------+----------+----------------------------------+
```

EmbeddingGemma-300M was chosen because:
- 1536-d output matches the desired shared embedding space dimension
- Modern architecture (Gemma family) with strong text understanding
- 256K vocabulary supports multilingual and technical text
- Specifically designed for producing high-quality dense embeddings
- 300M params balances capacity against training efficiency

---

## 6. Loss Function: Bi-Directional InfoNCE

### 6.1 Mathematical Formulation

The training objective is symmetric (bi-directional) InfoNCE, also known as
NT-Xent (Normalized Temperature-scaled Cross-Entropy):

```
Definitions:
  v_i = L2_normalized video embedding for sample i     (1536-d)
  t_i = L2_normalized text embedding for sample i      (1536-d)
  B   = batch size
  tau = learned temperature parameter (initialized ~0.07)

Similarity matrix:
  S_ij = (v_i . t_j) / tau     for all i, j in [1, B]
  Shape: (B, B)

  Since v and t are L2-normalized, v_i . t_j = cosine_similarity(v_i, t_j)

Video-to-Text loss (each video should match its paired text):

                    exp(S_ii)
  L_v2t = -1/B * SUM_i log ----------------
                            SUM_j exp(S_ij)

  Equivalent to: CrossEntropyLoss(S, targets=[0, 1, 2, ..., B-1])

Text-to-Video loss (each text should match its paired video):

                    exp(S_ii)
  L_t2v = -1/B * SUM_i log ----------------
                            SUM_j exp(S_ji)

  Equivalent to: CrossEntropyLoss(S^T, targets=[0, 1, 2, ..., B-1])

Total loss:
  L = (L_v2t + L_t2v) / 2
```

### 6.2 Temperature Parameter

```
Temperature (tau) Role:
========================

tau controls the sharpness of the softmax distribution:

  Small tau (e.g., 0.01):
    - Very sharp distribution
    - Model very confident about matches
    - Can lead to training instability (gradients explode)

  Large tau (e.g., 1.0):
    - Very flat distribution
    - Model treats all pairs as similarly likely
    - Slow learning, poor discrimination

  Optimal tau (typically 0.05-0.1):
    - Balances discrimination and stability
    - Usually learned during training (log-parameterized)

Implementation:
  log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))
  tau = log_tau.exp()  # Always positive

  # Clamp for stability
  tau = tau.clamp(min=0.01, max=1.0)
```

### 6.3 Anti-Collapse Properties

```
Why Bi-Directional InfoNCE Prevents Collapse
===============================================

1. Negative samples from the batch:
   - Each video is pushed away from B-1 non-matching texts
   - Each text is pushed away from B-1 non-matching videos
   - Collapse (all embeddings identical) would make the denominator
     equal to B * exp(1/tau) for every sample, giving random-chance
     loss of log(B) -- NOT a minimum

2. L2 normalization constrains the embedding space:
   - All embeddings lie on the unit hypersphere
   - Cannot collapse to a single point AND minimize loss
   - Must spread out to minimize loss for all samples

3. Bi-directionality:
   - V2T alone could collapse video embeddings if text is diverse
   - T2V alone could collapse text embeddings if video is diverse
   - Both together ensure NEITHER modality collapses

4. Learned temperature:
   - Tau adapts to the current embedding distribution
   - Prevents pathological gradient magnitudes

5. Large batch size:
   - More negatives per positive
   - Harder to "cheat" by finding a trivial separation
   - Effective batch size after gathering across GPUs: B_eff = B * N_gpus
```

### 6.4 Implementation

```python
# Pseudocode for bi-directional InfoNCE

def infonce_loss(video_emb, text_emb, temperature):
    """
    Args:
        video_emb: (B, 1536), L2-normalized
        text_emb:  (B, 1536), L2-normalized
        temperature: scalar (learned)
    Returns:
        loss: scalar
    """
    # Similarity matrix
    logits = video_emb @ text_emb.T / temperature  # (B, B)

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(B, device=logits.device)  # [0, 1, ..., B-1]

    # Bi-directional loss
    loss_v2t = F.cross_entropy(logits, labels)       # video-to-text
    loss_t2v = F.cross_entropy(logits.T, labels)     # text-to-video

    loss = (loss_v2t + loss_t2v) / 2.0
    return loss
```

---

## 7. Training Procedure

### 7.1 Two-Stage Training

```
Stage 1: Pre-training
=======================

Purpose: Learn initial alignment between video and text embeddings
Duration: Majority of total training compute (~80% of total steps)

Key characteristics:
  - Large, diverse video-text dataset
  - Constant learning rate (no decay during this stage)
  - Focus on broad coverage of visual-linguistic concepts
  - Gradient accumulation to achieve large effective batch size

Hyperparameters:
  Learning rate:        1e-4 (constant throughout Stage 1)
  Y-encoder LR:         5e-6 (0.05x base)
  Batch size:           256 per GPU
  Effective batch size: 256 * N_gpus (e.g., 256 * 64 = 16384)
  Weight decay:         0.05
  Optimizer:            AdamW
  Adam betas:           (0.9, 0.95)
  Adam epsilon:         1e-8
  Gradient clipping:    1.0 (max norm)
  Warmup:               2000 steps (linear warmup)
  Precision:            BF16 mixed precision


Stage 2: Supervised Fine-Tuning (SFT)
=======================================

Purpose: Refine alignment on higher-quality, curated data
Duration: ~20% of total training compute

Key characteristics:
  - Curated, higher-quality video-text pairs
  - Cosine annealing learning rate schedule
  - Potentially includes task-specific data (e.g., VQA pairs)
  - May include additional data filtering / quality scoring

Hyperparameters:
  Learning rate:        Starts at 1e-4, cosine decay to ~1e-6
  Y-encoder LR:         Proportionally reduced (0.05x base)
  Batch size:           Same as Stage 1
  Other hyperparameters: Same as Stage 1
  Duration:             ~20% of Stage 1 steps
```

### 7.2 Learning Rate Schedules

```
Stage 1 (Constant LR):

LR |
   |  ________________
   | /                \
   |/                  (abrupt transition to Stage 2)
   +-------------------+-------> steps
   0    warmup        S1_end

Stage 2 (Cosine Annealing):

LR |
   |____
   |    \
   |     \___
   |         \___
   |             \____
   +------------------+-------> steps
  S1_end             S2_end

Combined Schedule:

LR |
   |      ___________
   |     /           \
   |    /             \
   |   /               \___
   |  /                    \____
   | /                          \____
   +----+---------------------------+-> steps
   0  warmup  S1_end              S2_end
```

### 7.3 Data Pipeline

```
Training Data Pipeline
========================

1. Video loading:
   - Decode video at target FPS (e.g., 8 FPS)
   - Sample T=16 consecutive frames
   - Resize to 224x224 (shorter side resize + center crop)
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

2. Text processing:
   - Tokenize with Gemma tokenizer (256K vocab)
   - Truncate to max length (e.g., 77 or 128 tokens)
   - Pad to batch max length
   - Append [EOS] token for pooling

3. Batching:
   - Sample video-text pairs uniformly
   - Collate into batches of size B per GPU
   - All-gather text/video embeddings across GPUs for InfoNCE
     (effective negatives = B * N_gpus)

4. No augmentations:
   - Consistent with JEPA philosophy
   - No color jitter, random crop, or horizontal flip
   - Only spatial resize + center crop for consistency
```

### 7.4 Gradient Flow

```
Gradient Flow Diagram
======================

                    Loss
                     |
          +----------+----------+
          |                     |
    video_emb (grad)      text_emb (grad)
          |                     |
    [proj_out] (grad)     [Y-Encoder] (grad * 0.05)
          |
    [query outputs] (grad)
          |
    [Predictor 8 layers] (grad)
          |
    +-----+-----+
    |           |
 [proj_in]  [query_tokens]
 (grad)     (grad)
    |
 [X-Encoder output]
    |
 [STOP]  <-- frozen, no gradients

Summary:
  X-Encoder:  NO gradients (frozen)
  proj_in:    Full gradients
  Predictor:  Full gradients
  query_tokens: Full gradients
  proj_out:   Full gradients
  Y-Encoder:  Gradients scaled by 0.05x (via separate param group with low LR)
```

---

## 8. Inference Modes

### 8.1 Mode 1: Captioning

Captioning uses the learned embedding space for retrieval-based caption generation.

```
Captioning Inference Pipeline
===============================

Step 1: Encode video
  video -> X-Encoder -> Predictor -> pool query outputs
  Result: video_emb (1536-d)

Step 2: Build text embedding index (offline, one-time)
  For each caption c_i in the caption bank:
    text_emb_i = Y-Encoder(c_i)
  Store as matrix: TextBank (N_captions, 1536)

Step 3: Retrieve nearest caption
  similarities = video_emb @ TextBank.T    # (N_captions,)
  best_idx = argmax(similarities)
  caption = caption_bank[best_idx]

Step 4 (optional): Re-rank top-K
  Retrieve top-K candidates
  Re-score with more expensive methods if needed

Performance characteristics:
  - No autoregressive generation (no beam search, no sampling)
  - Inference time dominated by video encoding (X-Encoder + Predictor)
  - Text retrieval is a single matrix multiply (very fast)
  - Quality depends on caption bank coverage
```

### 8.2 Mode 2: Discriminative VQA

Visual Question Answering treats the task as embedding-space matching.

```
Discriminative VQA Pipeline
==============================

Given: video V, question Q, candidate answers {A_1, ..., A_K}

Step 1: Encode video (same as captioning)
  video_emb = encode_video(V)  # (1536,)

Step 2: Encode each candidate answer
  For each answer A_k:
    Combine question and answer: text_k = f"Question: {Q} Answer: {A_k}"
    text_emb_k = Y_Encoder(text_k)  # (1536,)

  Text embeddings matrix: (K, 1536)

Step 3: Score each candidate
  scores = video_emb @ text_embs.T  # (K,)

Step 4: Select best answer
  best_k = argmax(scores)
  answer = A_k[best_k]

Advantages:
  - No generative decoding needed
  - Deterministic (no sampling randomness)
  - Can evaluate all candidates in parallel
  - Works with any number of candidate answers

Limitations:
  - Requires candidate answer set (not open-ended generation)
  - Quality depends on how well the embedding space captures
    fine-grained question-answer semantics
```

### 8.3 Mode 3: Selective Decoding

Selective decoding enables efficient inference by reducing the number of tokens
that need to be processed. See Section 9 for the detailed algorithm.

```
Selective Decoding Pipeline
==============================

Step 1: Full predictor forward pass
  visual_tokens + query_tokens -> Predictor -> query_outputs
  query_outputs: (N_q, 1536) = (64, 1536)

Step 2: Cluster query outputs (Ward agglomerative clustering)
  clusters = ward_clustering(query_outputs, n_clusters=K)
  Where K = N_q / speedup_factor (e.g., 64 / 2.85 ~ 22)

Step 3: Select representative tokens
  For each cluster, select the token nearest to cluster centroid
  selected_tokens: (K, 1536)

Step 4: Use selected tokens for downstream task
  video_emb = mean_pool(selected_tokens)  # (1536,)
  Proceed with captioning or VQA using this reduced representation

Speedup: 2.85x with minimal quality degradation
```

---

## 9. Selective Decoding Algorithm

### 9.1 Motivation

After the predictor processes the concatenated visual + query tokens, the N_q
query output tokens contain the video's semantic information. However, many of
these tokens encode redundant or overlapping information. Selective decoding
identifies the unique "information slots" and discards redundant ones.

### 9.2 Ward Agglomerative Clustering

Ward's method is a hierarchical clustering algorithm that minimizes the total
within-cluster variance at each merge step.

```
Ward Agglomerative Clustering: Step-by-Step
=============================================

Input: N_q = 64 query token embeddings, each of dimension 1536
  tokens = {q_1, q_2, ..., q_64}   each q_i in R^1536

Initialize:
  Each token is its own cluster: C = {C_1, C_2, ..., C_64}
  Compute pairwise Ward distances between all pairs

Ward distance between clusters C_a and C_b:
  d_Ward(C_a, C_b) = (|C_a| * |C_b|) / (|C_a| + |C_b|) * ||mu_a - mu_b||^2

  Where:
    |C_a| = number of points in cluster a
    mu_a  = centroid of cluster a
    ||.|| = L2 norm

Iteration (repeat until desired number of clusters K reached):
  1. Find the pair (C_a, C_b) with minimum Ward distance
  2. Merge C_a and C_b into a new cluster C_new
  3. Update the distance matrix:
     For each remaining cluster C_c:
       d_Ward(C_new, C_c) computed using Lance-Williams formula:
         d(C_new, C_c) = ((|C_a|+|C_c|)*d(C_a,C_c)
                        + (|C_b|+|C_c|)*d(C_b,C_c)
                        - |C_c|*d(C_a,C_b))
                        / (|C_a| + |C_b| + |C_c|)

Stop when number of clusters = K

Example with K = 22 (for 2.85x speedup from 64 tokens):
  Start: 64 clusters (each is one token)
  Merge 1: 63 clusters
  Merge 2: 62 clusters
  ...
  Merge 42: 22 clusters  <-- stop here
```

### 9.3 Representative Token Selection

```
After clustering into K clusters:

For each cluster C_k (k = 1, ..., K):
  1. Compute cluster centroid:
     mu_k = (1/|C_k|) * SUM_{q_i in C_k} q_i

  2. Select representative token (nearest to centroid):
     rep_k = argmin_{q_i in C_k} ||q_i - mu_k||

  3. Use rep_k as the cluster's representative embedding

Result: K representative tokens out of original N_q
  selected: (K, 1536)  where K << N_q
```

### 9.4 Why Ward Clustering?

```
Comparison of clustering approaches:
+-------------------+-----------+--------------------+--------------------+
| Method            | Time      | Quality            | Deterministic?     |
+-------------------+-----------+--------------------+--------------------+
| K-Means           | O(N*K*I)  | Good, but random   | No (random init)   |
|                   |           | initialization      |                    |
+-------------------+-----------+--------------------+--------------------+
| Ward (agglom.)    | O(N^2*d)  | Excellent, variance-| Yes                |
|                   |           | minimizing          |                    |
+-------------------+-----------+--------------------+--------------------+
| Random selection  | O(K)      | Poor                | No                 |
+-------------------+-----------+--------------------+--------------------+
| Top-K by norm     | O(N*logK) | Biased toward high  | Yes                |
|                   |           | norm tokens         |                    |
+-------------------+-----------+--------------------+--------------------+

Ward advantages for VL-JEPA:
  - Deterministic: same input always gives same clusters
  - Variance-minimizing: preserves information diversity
  - No hyperparameter tuning (unlike K-Means iterations)
  - O(N^2) is fine since N_q = 64 (tiny)
  - Hierarchical: can explore different K values without re-running
```

### 9.5 Speedup Analysis

```
Without selective decoding:
  Process all N_q = 64 query token embeddings
  Downstream computation: proportional to N_q

With selective decoding (K = 22):
  Process only K = 22 representative tokens
  Speedup = N_q / K = 64 / 22 = 2.91x (approximately 2.85x reported)

Where the speedup comes from:
  - Video encoding (X-Encoder): UNCHANGED (still processes all patches)
  - Predictor forward pass: UNCHANGED (still processes all tokens)
  - Downstream operations: REDUCED by factor of N_q / K
    - Embedding pooling: 22 instead of 64 tokens
    - Similarity computation: 22-d instead of 64-d (marginal)
    - The main gain is in scenarios with per-token decoding

For captioning/VQA with simple pooling, the speedup is modest.
The 2.85x speedup is most significant when:
  - Each query token is decoded independently
  - Multiple rounds of similarity computation are needed
  - Token-level operations dominate inference time
```

---

## 10. Implementation Considerations

### 10.1 Memory Requirements

```
Memory Budget (per GPU, BF16 mixed precision)
===============================================

Component                        | Memory
---------------------------------|---------
X-Encoder parameters (frozen)    | 304M * 2B = 608 MB
X-Encoder activations            | ~2 GB (with gradient checkpointing)
  - Note: no optimizer states since frozen

Predictor parameters             | 490M * 2B = 980 MB
Predictor optimizer states       | 490M * 8B = 3.92 GB
  (AdamW: 2x param for moments)
Predictor activations            | ~3 GB (with gradient checkpointing)
Predictor gradients              | 490M * 2B = 980 MB

Y-Encoder parameters             | 300M * 2B = 600 MB
Y-Encoder optimizer states       | 300M * 8B = 2.4 GB
Y-Encoder activations            | ~1 GB (with gradient checkpointing)
Y-Encoder gradients              | 300M * 2B = 600 MB

Projections + query tokens       | ~12M * 2B = 24 MB
Projection optimizer states      | ~12M * 8B = 96 MB

Batch data (videos + text)       | ~1-2 GB (depends on B and T)

Embeddings for InfoNCE           | B * 1536 * 2 * 2B ~ small
---------------------------------|---------
Approximate total per GPU        | ~16-18 GB

With FSDP sharding across 8 GPUs: ~2-3 GB per GPU for params/optim
Plus activations: ~6-8 GB per GPU
Total per GPU with FSDP:         ~8-11 GB
```

### 10.2 Gradient Checkpointing

```
Gradient Checkpointing Strategy
=================================

Without checkpointing:
  All intermediate activations stored -> enormous memory
  24 ViT layers + 8 Llama layers + 18 Gemma layers = 50 layers
  Each layer stores multiple intermediate tensors

With checkpointing:
  X-Encoder:  Checkpoint every 4 layers (6 checkpoints)
              Recompute 3 layers per checkpoint on backward pass
              Memory saved: ~75% of X-Encoder activations
              NOTE: X-Encoder is frozen, but activations are still
              needed for the predictor's backward pass

  Predictor:  Checkpoint every 2 layers (4 checkpoints)
              Recompute 1 layer per checkpoint
              Memory saved: ~50% of Predictor activations

  Y-Encoder:  Checkpoint every 3 layers (6 checkpoints)
              Recompute 2 layers per checkpoint
              Memory saved: ~67% of Y-Encoder activations

Implementation (PyTorch):
  # Wrap each checkpointed segment
  from torch.utils.checkpoint import checkpoint

  def forward_with_checkpointing(layers, x, chunks=4):
      chunk_size = len(layers) // chunks
      for i in range(0, len(layers), chunk_size):
          chunk_layers = layers[i:i+chunk_size]
          x = checkpoint(
              lambda *args: sequential_forward(chunk_layers, *args),
              x,
              use_reentrant=False
          )
      return x
```

### 10.3 FSDP Strategy

```
Fully Sharded Data Parallel (FSDP) Configuration
===================================================

VL-JEPA uses FSDP to distribute model parameters and optimizer states
across multiple GPUs. The strategy differs per component:

X-Encoder (frozen):
  Sharding: FULL_SHARD
  - Parameters sharded across GPUs
  - All-gathered when needed for forward pass
  - No optimizer states (frozen)
  - Wrap policy: per-transformer-block

Predictor (trainable):
  Sharding: FULL_SHARD
  - Parameters, gradients, and optimizer states all sharded
  - All-gathered for forward/backward
  - Wrap policy: per-transformer-block
  - Mixed precision: BF16 compute, FP32 gradient accumulation

Y-Encoder (trainable, low LR):
  Sharding: FULL_SHARD
  - Same as predictor
  - Separate param group with 0.05x LR

FSDP configuration:
  sharding_strategy:   FULL_SHARD
  mixed_precision:     MixedPrecision(
                         param_dtype=torch.bfloat16,
                         reduce_dtype=torch.float32,
                         buffer_dtype=torch.bfloat16
                       )
  auto_wrap_policy:    ModuleWrapPolicy(
                         {TransformerBlock, LlamaDecoderLayer, GemmaBlock}
                       )
  backward_prefetch:   BACKWARD_PRE
  forward_prefetch:    True
  limit_all_gathers:   True
  use_orig_params:     True  (for per-param LR groups)
```

### 10.4 BF16 Mixed Precision

```
BF16 Mixed Precision Strategy
================================

BF16 (Brain Floating Point 16):
  - 1 sign bit, 8 exponent bits, 7 mantissa bits
  - Same dynamic range as FP32 (8 exponent bits)
  - Lower precision than FP16 (7 vs 10 mantissa bits)
  - But: no overflow issues (unlike FP16)
  - Natively supported on A100, H100 GPUs

What runs in BF16:
  - All forward pass computations (linear layers, attention)
  - X-Encoder parameters (stored as BF16)
  - Predictor parameters (stored as BF16, master copy in FP32)
  - Y-Encoder parameters (stored as BF16, master copy in FP32)

What runs in FP32:
  - Loss computation (InfoNCE with softmax -- numerically sensitive)
  - Gradient accumulation (important for small gradients from Y-Encoder)
  - Optimizer step (AdamW moment updates)
  - L2 normalization of embeddings
  - Temperature parameter

Implementation:
  # PyTorch autocast context manager
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      visual_tokens = x_encoder(video)      # BF16
      query_outputs = predictor(visual_tokens, query_tokens)  # BF16

  # Explicitly FP32 for numerically sensitive operations
  video_emb = F.normalize(query_outputs.float(), dim=-1)  # FP32
  text_emb = F.normalize(text_outputs.float(), dim=-1)    # FP32
  loss = infonce_loss(video_emb, text_emb, temperature)    # FP32
```

### 10.5 Distributed Training Considerations

```
Multi-GPU InfoNCE: Gathering Embeddings
=========================================

InfoNCE benefits from large effective batch sizes (more negatives).
With N GPUs, each computing B local samples:

  Effective batch size = B * N

Implementation:
  1. Each GPU computes local video_emb and text_emb: (B, 1536)
  2. All-gather across GPUs:
     all_video_emb: (B*N, 1536)
     all_text_emb:  (B*N, 1536)
  3. Compute similarity matrix: (B*N, B*N)
  4. Compute loss using the full matrix
  5. Backward: only gradients for local samples are needed

  # PyTorch distributed all-gather
  gathered_video = [torch.zeros_like(video_emb) for _ in range(world_size)]
  dist.all_gather(gathered_video, video_emb)
  all_video_emb = torch.cat(gathered_video, dim=0)

  # Important: make local embeddings require grad
  all_video_emb[rank * B : (rank + 1) * B] = video_emb

Typical setup:
  8 GPUs, B=256 per GPU -> effective batch = 2048
  64 GPUs, B=256 per GPU -> effective batch = 16384
```

### 10.6 Inference Optimization

```
Inference Optimizations
========================

1. X-Encoder compilation:
   - torch.compile() the frozen encoder
   - Fuses operations, eliminates overhead
   - One-time compilation cost, then cached

2. KV-cache for Predictor:
   - Not applicable (bidirectional attention, no autoregressive decoding)

3. Flash Attention:
   - Apply to all attention layers (ViT, Llama, Gemma)
   - Reduces memory from O(N^2) to O(N)
   - Speeds up attention by 2-4x

4. Token pruning (beyond selective decoding):
   - Can prune low-attention visual tokens after X-Encoder
   - Reduces predictor sequence length
   - Additional speedup on top of selective decoding

5. Quantization (post-training):
   - INT8 quantization of X-Encoder (frozen, easy to quantize)
   - INT8 for predictor weights during inference
   - Minimal quality loss for embedding models

6. Batched inference:
   - Process multiple videos simultaneously
   - Amortize GPU kernel launch overhead
   - Text encoding can be pre-computed and cached
```

---

## 11. Key Hyperparameters Table

### 11.1 Architecture Hyperparameters

```
+-----------------------------------+---------------------------+
| Hyperparameter                    | Value                     |
+-----------------------------------+---------------------------+
| X-Encoder architecture            | ViT-L/16 (V-JEPA 2)      |
| X-Encoder hidden dim              | 1024                      |
| X-Encoder layers                  | 24                        |
| X-Encoder heads                   | 16                        |
| X-Encoder MLP dim                 | 4096                      |
| X-Encoder parameters              | 304M (frozen)             |
+-----------------------------------+---------------------------+
| Predictor source model            | Llama-3.2-1B              |
| Predictor layers used             | Last 8 (of 16)            |
| Predictor hidden dim              | 2048                      |
| Predictor attention heads         | 32                        |
| Predictor KV heads (GQA)         | 8                         |
| Predictor MLP dim                 | 8192                      |
| Predictor attention type          | Bidirectional (no causal)  |
| Predictor parameters              | 490M (trainable)          |
+-----------------------------------+---------------------------+
| Y-Encoder architecture            | EmbeddingGemma-300M       |
| Y-Encoder hidden dim              | 1536                      |
| Y-Encoder layers                  | 18                        |
| Y-Encoder heads                   | 12                        |
| Y-Encoder parameters              | 300M (0.05x LR)          |
+-----------------------------------+---------------------------+
| Shared embedding dimension        | 1536                      |
| Number of query tokens (N_q)      | 64                        |
| Patch size (spatial)              | 16 x 16                   |
| Patch size (temporal)             | 2 frames                  |
| Input resolution                  | 224 x 224                 |
| Number of frames (T)             | 16                        |
| Total visual tokens (N_vis)       | 1568 (8 x 14 x 14)       |
+-----------------------------------+---------------------------+
```

### 11.2 Training Hyperparameters

```
+-----------------------------------+---------------------------+
| Hyperparameter                    | Value                     |
+-----------------------------------+---------------------------+
|             STAGE 1: PRE-TRAINING                             |
+-----------------------------------+---------------------------+
| Optimizer                         | AdamW                     |
| Base learning rate                | 1e-4                      |
| Y-Encoder learning rate           | 5e-6 (0.05x base)        |
| LR schedule                       | Constant (after warmup)   |
| Warmup steps                      | 2000 (linear)             |
| Weight decay                      | 0.05                      |
| Adam beta_1                       | 0.9                       |
| Adam beta_2                       | 0.95                      |
| Adam epsilon                      | 1e-8                      |
| Gradient clipping (max norm)      | 1.0                       |
| Batch size per GPU                | 256                       |
| Precision                         | BF16 mixed                |
| Temperature (tau) init            | 0.07 (learned)            |
| Temperature clamp range           | [0.01, 1.0]              |
+-----------------------------------+---------------------------+
|        STAGE 2: SUPERVISED FINE-TUNING                        |
+-----------------------------------+---------------------------+
| LR schedule                       | Cosine annealing          |
| Starting LR                       | 1e-4 (same as Stage 1)   |
| Final LR                          | ~1e-6                     |
| Y-Encoder LR                      | 0.05x base (cosine)      |
| Duration                          | ~20% of Stage 1 steps    |
| Data                              | Curated, higher quality   |
| Other hyperparameters             | Same as Stage 1           |
+-----------------------------------+---------------------------+
```

### 11.3 Data Hyperparameters

```
+-----------------------------------+---------------------------+
| Hyperparameter                    | Value                     |
+-----------------------------------+---------------------------+
| Video FPS (sampling rate)         | 8                         |
| Frames per clip                   | 16                        |
| Clip duration                     | 2 seconds                 |
| Spatial resolution                | 224 x 224                 |
| Spatial preprocessing             | Shorter side resize +     |
|                                   | center crop               |
| Pixel normalization mean          | [0.485, 0.456, 0.406]    |
| Pixel normalization std           | [0.229, 0.224, 0.225]    |
| Text max tokens                   | 77 or 128                 |
| Text tokenizer                    | Gemma (256K vocab)        |
| Data augmentations                | None                      |
+-----------------------------------+---------------------------+
```

### 11.4 Selective Decoding Hyperparameters

```
+-----------------------------------+---------------------------+
| Hyperparameter                    | Value                     |
+-----------------------------------+---------------------------+
| Clustering algorithm              | Ward agglomerative        |
| Input tokens (N_q)               | 64                        |
| Output clusters (K)              | ~22                       |
| Target speedup                    | 2.85x                     |
| Distance metric                   | Euclidean (L2)            |
| Linkage criterion                 | Ward (min variance)       |
| Representative selection          | Nearest to centroid       |
+-----------------------------------+---------------------------+
```

### 11.5 Infrastructure Hyperparameters

```
+-----------------------------------+---------------------------+
| Hyperparameter                    | Value                     |
+-----------------------------------+---------------------------+
| GPU type (recommended)            | NVIDIA A100 80GB or       |
|                                   | H100 80GB                 |
| Minimum GPUs                      | 8                         |
| Recommended GPUs                  | 64                        |
| FSDP sharding strategy            | FULL_SHARD                |
| Gradient checkpointing            | Enabled (all components)  |
| Flash Attention                   | Enabled (v2)              |
| Effective batch size              | B_local * N_gpus          |
| Communication backend             | NCCL                      |
| Number of data workers            | 8 per GPU                 |
| Pin memory                        | True                      |
+-----------------------------------+---------------------------+
```

---

## Appendix A: Pseudocode for Complete Training Loop

```python
# VL-JEPA Training Loop Pseudocode

# === Model Initialization ===
x_encoder = load_vjepa2_vit_l(pretrained=True)
x_encoder.eval()
x_encoder.requires_grad_(False)  # Freeze

predictor = load_llama32_1b_last_8_layers()
predictor.remove_causal_mask()
proj_in = nn.Linear(1024, 2048)
proj_out = nn.Linear(2048, 1536)
query_tokens = nn.Parameter(torch.randn(64, 2048) * 0.02)

y_encoder = load_embedding_gemma_300m(pretrained=True)

log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

# === Optimizer ===
optimizer = AdamW([
    {'params': predictor.parameters(), 'lr': 1e-4},
    {'params': [proj_in.weight, proj_in.bias,
                proj_out.weight, proj_out.bias,
                query_tokens, log_temperature], 'lr': 1e-4},
    {'params': y_encoder.parameters(), 'lr': 5e-6},  # 0.05x
], weight_decay=0.05, betas=(0.9, 0.95))

# === Training Loop ===
for step in range(total_steps):

    video, text_tokens = next(dataloader)
    # video: (B, 16, 3, 224, 224)
    # text_tokens: (B, L_text)

    with torch.no_grad():
        visual_tokens = x_encoder(video)
        # (B, 1568, 1024)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        # Project to predictor space
        visual_proj = proj_in(visual_tokens)
        # (B, 1568, 2048)

        # Expand and concat query tokens
        queries = query_tokens.unsqueeze(0).expand(B, -1, -1)
        # (B, 64, 2048)
        predictor_input = torch.cat([visual_proj, queries], dim=1)
        # (B, 1632, 2048)

        # Predictor forward (bidirectional)
        predictor_output = predictor(predictor_input)
        # (B, 1632, 2048)

        # Extract query outputs
        query_output = predictor_output[:, 1568:, :]
        # (B, 64, 2048)

        # Project to embedding space
        video_emb = proj_out(query_output)
        # (B, 64, 1536)

    # Pool and normalize (FP32)
    video_emb = video_emb.float().mean(dim=1)       # (B, 1536)
    video_emb = F.normalize(video_emb, dim=-1)       # (B, 1536)

    # Y-Encoder (BF16 forward, FP32 normalize)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        text_emb = y_encoder(text_tokens)
    text_emb = text_emb.float()
    text_emb = F.normalize(text_emb, dim=-1)         # (B, 1536)

    # All-gather for large effective batch
    all_video_emb = all_gather(video_emb)             # (B*N, 1536)
    all_text_emb = all_gather(text_emb)               # (B*N, 1536)

    # InfoNCE loss (FP32)
    tau = log_temperature.exp().clamp(0.01, 1.0)
    logits = all_video_emb @ all_text_emb.T / tau     # (B*N, B*N)
    labels = torch.arange(B * world_size)
    loss = (F.cross_entropy(logits, labels)
          + F.cross_entropy(logits.T, labels)) / 2

    # Backward and step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Update LR schedule
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    elif stage == 'pretrain':
        lr = base_lr  # constant
    else:  # SFT stage
        lr = cosine_annealing(step, base_lr, min_lr)
    set_lr(optimizer, lr)
```

---

## Appendix B: Key Architectural Decisions and Rationale

```
+---------------------------------------+------------------------------------------+
| Decision                              | Rationale                                |
+---------------------------------------+------------------------------------------+
| Freeze X-Encoder                      | Preserve V-JEPA 2 representation quality;|
|                                       | reduce memory and compute; stabilize     |
|                                       | training                                 |
+---------------------------------------+------------------------------------------+
| Use last 8 Llama layers               | Later layers capture higher abstractions;|
| (not first 8)                         | better for cross-modal reasoning than    |
|                                       | early token-mixing layers                |
+---------------------------------------+------------------------------------------+
| Remove causal mask in predictor       | Visual tokens have no natural ordering;  |
|                                       | bidirectional attention lets queries      |
|                                       | attend to all visual context freely      |
+---------------------------------------+------------------------------------------+
| 64 query tokens                       | More than 1 CLS token allows richer      |
|                                       | representation; 64 balances capacity     |
|                                       | vs. efficiency; enables selective         |
|                                       | decoding later                           |
+---------------------------------------+------------------------------------------+
| 1536-d shared space                   | Matches EmbeddingGemma native dim;       |
|                                       | no information loss on the text side;    |
|                                       | large enough for fine-grained matching   |
+---------------------------------------+------------------------------------------+
| Y-Encoder at 0.05x LR                | Preserves pre-trained text knowledge;    |
|                                       | gradual adaptation prevents catastrophic |
|                                       | forgetting; acts as "soft EMA" target    |
+---------------------------------------+------------------------------------------+
| InfoNCE instead of L2/L1             | Contrastive loss naturally handles the   |
|                                       | multi-modal alignment task; provides     |
|                                       | within-batch negatives; well-understood  |
|                                       | collapse properties                      |
+---------------------------------------+------------------------------------------+
| Two-stage training (constant->cosine) | Stage 1: broad exploration with stable   |
|                                       | LR; Stage 2: refinement with decaying LR |
|                                       | on curated data                          |
+---------------------------------------+------------------------------------------+
| Ward clustering for selective decode  | Deterministic, variance-minimizing;      |
|                                       | computationally trivial for N_q=64;      |
|                                       | hierarchical (can try different K)       |
+---------------------------------------+------------------------------------------+
| No data augmentations                 | Consistent with JEPA philosophy;         |
|                                       | masking/prediction provides sufficient   |
|                                       | training signal; avoids augmentation     |
|                                       | bias                                     |
+---------------------------------------+------------------------------------------+
```

---

*End of VL-JEPA Technical Deep-Dive*
