# Annotated Notes: VL-JEPA (Vision-Language JEPA)

**Paper**: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
**Authors**: Meta FAIR
**Date**: 2025
**Link**: [https://arxiv.org/abs/2512.10942](https://arxiv.org/abs/2512.10942)

---

## 1. Overview

VL-JEPA extends the JEPA paradigm to vision-language multimodal learning. Rather than using a generative decoder to map visual features to language (as in InstructBLIP, LLaVA, Qwen-VL), VL-JEPA trains a predictor to map visual features into a text embedding space via a joint-embedding predictive objective. This produces a model capable of captioning, VQA, video classification, and retrieval -- with a novel selective decoding mechanism that provides significant speedups.

### Core Claim

Vision-language alignment can be achieved through joint-embedding prediction (not generative pixel/token reconstruction), producing representations that are competitive with much larger generative VLMs while being more efficient at inference.

---

## 2. Full Architecture

### 2.1 System Diagram

```
VIDEO INPUT                          TEXT INPUT
    |                                     |
    v                                     v
[X-Encoder]                         [Y-Encoder]
Frozen V-JEPA 2 ViT-L              EmbeddingGemma-300M
304M params (frozen)                300M params (0.05x LR)
    |                                     |
    v                                     v
Visual features f_x               Text embeddings f_y
    |                                     |
    v                                     |
[Predictor]                               |
Llama-3.2-1B last 8 layers               |
490M params (full LR)                     |
    |                                     |
    v                                     v
Predicted text embeddings f_y_hat    Target embeddings f_y
    |                                     |
    +----------> Bi-directional InfoNCE <-+
                      Loss
```

### 2.2 Component Details

#### X-Encoder: Frozen V-JEPA 2 ViT-L (304M parameters)

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer Large |
| Parameters | 304M |
| Source | V-JEPA 2 pretrained weights |
| Training status | **Completely frozen** -- no gradient updates |
| Patch size | 16x16 spatial, 2 frames temporal |
| Embedding dimension | 1024 |
| Output | Sequence of visual feature tokens |

**Why frozen?** V-JEPA 2 already learned rich spatiotemporal representations from 1M+ hours of video. Freezing it:
- Preserves the quality of visual features.
- Dramatically reduces training compute (no gradient through the 304M visual encoder).
- Prevents catastrophic forgetting of visual knowledge.
- Makes the system modular -- the visual encoder can be swapped.

#### Predictor: Llama-3.2-1B Last 8 Layers (490M trainable parameters)

| Property | Value |
|----------|-------|
| Base model | Llama-3.2-1B |
| Layers used | Last 8 transformer layers only |
| Trainable parameters | ~490M |
| Input | Visual feature tokens from X-Encoder |
| Output | Predicted text embedding(s) |
| Learning rate | Full learning rate |
| Role | Transform visual features into text embedding space |

**Why last 8 layers of Llama?** Design rationale:
- The last layers of an LLM are where the most abstract, task-relevant representations are formed.
- The first layers handle lower-level token processing that is not needed here (input is already high-level visual features, not raw tokens).
- Using only 8 layers significantly reduces compute vs. a full 1B model.
- Llama's transformer architecture provides strong sequence modeling for the variable-length visual token sequences.

**Why Llama specifically?** The predictor needs to handle variable-length sequences of visual tokens and produce predictions in a language-model-compatible embedding space. Llama provides a well-pretrained transformer for this purpose.

#### Y-Encoder: EmbeddingGemma-300M (300M parameters, 0.05x LR)

| Property | Value |
|----------|-------|
| Architecture | EmbeddingGemma-300M |
| Parameters | 300M |
| Training status | Trained with 0.05x the predictor's learning rate |
| Input | Text (captions, questions, descriptions) |
| Output | Text embedding(s) |
| Role | Provides the prediction target in text embedding space |

**Why 0.05x LR?** The Y-Encoder serves as a slowly-moving target, similar to the EMA target encoder in I-JEPA. Training it with a very low learning rate:
- Provides a stable target for the predictor to learn against.
- Allows some adaptation to the task (unlike a fully frozen encoder).
- Prevents the target space from shifting too fast (which would destabilize training).
- This is an alternative to EMA -- instead of momentum-based updates, it uses a low learning rate to achieve similar stability.

**Why EmbeddingGemma?** EmbeddingGemma-300M is a text embedding model (not a generative model). It produces fixed-dimensional dense embeddings of text passages, which is exactly what JEPA needs as a prediction target.

### 2.3 Parameter Summary

| Component | Parameters | Trainable? | Learning Rate |
|-----------|-----------|------------|---------------|
| X-Encoder (V-JEPA 2 ViT-L) | 304M | No (frozen) | 0 |
| Predictor (Llama-3.2-1B last 8) | 490M | Yes | 1.0x (full) |
| Y-Encoder (EmbeddingGemma-300M) | 300M | Yes | 0.05x |
| **Total** | **~1.1B** | **~790M** | -- |

---

## 3. Bi-Directional InfoNCE Loss

### 3.1 Loss Formulation

VL-JEPA uses a bi-directional InfoNCE (Noise-Contrastive Estimation) loss:

```
L = L_v2t + L_t2v

L_v2t = -log( exp(sim(f_y_hat_i, f_y_i) / tau) / sum_j exp(sim(f_y_hat_i, f_y_j) / tau) )
  (video-to-text: for each video, find its matching text among all texts in the batch)

L_t2v = -log( exp(sim(f_y_hat_i, f_y_i) / tau) / sum_j exp(sim(f_y_hat_j, f_y_i) / tau) )
  (text-to-video: for each text, find its matching video among all videos in the batch)
```

Where:
- f_y_hat_i = Predictor(X-Encoder(video_i)) -- predicted text embedding from video
- f_y_i = Y-Encoder(text_i) -- actual text embedding
- sim(a, b) = cosine similarity
- tau = learnable temperature parameter

### 3.2 Why InfoNCE and Not L2?

A departure from I-JEPA / V-JEPA which use L2 loss in latent space:

- **Cross-modal alignment needs contrastive signal**: Within a single modality (image patches), L2 works because the encoder naturally produces compatible representations. Across modalities (video features vs. text embeddings), the spaces are fundamentally different and need an explicit alignment signal.
- **InfoNCE provides this alignment**: It explicitly trains the model to match correct video-text pairs and distinguish them from incorrect pairs.
- **Bi-directional**: Both directions (video-to-text and text-to-video) are optimized, ensuring the alignment is symmetric.

### 3.3 Relationship to CLIP

This loss is similar to CLIP's contrastive loss, but the architecture is fundamentally different:
- CLIP: Two encoders, direct similarity between encoded representations.
- VL-JEPA: Two encoders + a predictor. The predictor transforms visual features into text space before computing similarity. The predictor is the key differentiator -- it enables more complex mappings between modalities.

---

## 4. Two-Stage Training Procedure

### Stage 1: Pretraining

| Property | Value |
|----------|-------|
| Objective | Bi-directional InfoNCE |
| Data | Video-text pairs (captions, descriptions) |
| X-Encoder | Frozen |
| Predictor | Trained (full LR) |
| Y-Encoder | Trained (0.05x LR) |
| Focus | Learning vision-language alignment |

During pretraining, the model learns to predict text embeddings from visual features. The predictor learns the cross-modal mapping, while the Y-Encoder slowly adapts to provide good prediction targets.

### Stage 2: Supervised Fine-Tuning (SFT)

| Property | Value |
|----------|-------|
| Objective | Task-specific (e.g., classification, VQA) |
| Data | Labeled datasets for downstream tasks |
| Adaptation | Lightweight heads or prompt-based |
| Focus | Adapting aligned representations to specific tasks |

The SFT stage adapts the pretrained model to specific downstream tasks. Because the representations are already well-aligned, this stage requires relatively little data and compute.

---

## 5. Three Inference Modes

VL-JEPA supports three distinct inference modes, making it versatile across tasks.

### 5.1 Captioning Mode

```
Video --> X-Encoder --> Predictor --> predicted text embedding
                                          |
                                          v
                                    Text decoder (generates caption)
```

The predicted text embedding is used as a prompt/condition for text generation. The model maps video features to the text embedding space, and a decoder generates natural language from that embedding.

### 5.2 VQA (Visual Question Answering) Mode

```
Video + Question --> X-Encoder + Question encoding --> Predictor --> answer embedding
                                                                          |
                                                                          v
                                                                    Answer decoding
```

The question is incorporated into the prediction process, conditioning the output on both visual content and the question.

### 5.3 Selective Decoding Mode

This is the most novel inference mode and provides significant speedups.

```
Video --> X-Encoder --> Predictor --> predicted text embeddings
                                          |
                                          v
                                    Ward clustering on candidate responses
                                          |
                                          v
                                    Score cluster representatives with Y-Encoder
                                          |
                                          v
                                    Select best candidate
```

**How selective decoding works:**

1. Generate or enumerate candidate text responses.
2. Encode candidates with Y-Encoder to get text embeddings.
3. Apply Ward hierarchical clustering to group similar candidates.
4. Score cluster representatives against the predicted text embedding.
5. Select the best-scoring candidate.

**Why this is fast**: Instead of scoring every candidate individually, clustering reduces the number of comparisons. Ward clustering groups similar candidates, so only representative candidates need full scoring.

---

## 6. Selective Decoding Deep Dive

### 6.1 Ward Clustering

Ward's method is a hierarchical agglomerative clustering algorithm that minimizes the total within-cluster variance at each merge step.

Algorithm:
1. Start with each candidate as its own cluster.
2. At each step, merge the two clusters whose merger causes the smallest increase in total within-cluster variance.
3. Continue until the desired number of clusters is reached.

**Why Ward over other clustering?** Ward produces compact, well-separated clusters. For text embeddings, this means candidates with similar meaning are grouped together, allowing efficient elimination of whole groups.

### 6.2 Speedup: 2.85x

Selective decoding achieves a **2.85x inference speedup** compared to scoring all candidates individually.

The speedup comes from:
- Reducing the number of Y-Encoder forward passes (score cluster representatives, not all candidates).
- Hierarchical elimination: If a cluster representative scores poorly, all candidates in that cluster are eliminated.
- The clustering itself is cheap (operates on precomputed embeddings).

### 6.3 When to Use Selective Decoding

- **Classification with many classes**: Instead of scoring all 1000 ImageNet classes, cluster them and score representatives.
- **Multiple-choice VQA**: Cluster answer candidates and select.
- **Retrieval with large galleries**: Cluster the gallery and do hierarchical search.
- **NOT suitable for**: Open-ended generation (captioning, free-form VQA) where the candidate set is not predefined.

---

## 7. Results

### 7.1 Video Classification (8 Benchmarks)

**Average accuracy across 8 video classification benchmarks: 46.4%**

These benchmarks test diverse aspects of video understanding:
- Temporal reasoning (Something-Something v2)
- Action recognition (Kinetics-400, Kinetics-600)
- Fine-grained activity understanding (Epic Kitchens)
- Scene understanding
- Object interaction recognition

### 7.2 Video Retrieval (8 Benchmarks)

**Average recall across 8 video retrieval benchmarks: 58.4%**

Retrieval tasks test whether the model can match videos to text descriptions (and vice versa) in a shared embedding space. Strong retrieval performance validates the quality of the cross-modal alignment learned by the InfoNCE objective.

### 7.3 Detailed Benchmark Table

| Task Type | Metric | VL-JEPA | Notes |
|-----------|--------|---------|-------|
| Video Classification (avg 8) | Accuracy | 46.4% | Diverse temporal/spatial tasks |
| Video Retrieval (avg 8) | Recall | 58.4% | Text-video matching |
| Selective decoding speedup | Speedup | 2.85x | vs. exhaustive scoring |

### 7.4 Comparison with Baselines

| Model | Type | Classification | Retrieval | Notes |
|-------|------|---------------|-----------|-------|
| CLIP | Contrastive VL | Competitive | Strong | Requires massive image-text data |
| SigLIP2 | Contrastive VL | Strong | Strong | Improved CLIP variant |
| Perception Encoder | Contrastive VL | Strong | Strong | Meta's CLIP-scale model |
| InstructBLIP | Generative VLM | Strong on VQA | Weak on retrieval | Much larger, generative |
| Qwen-VL | Generative VLM | Strong on VQA | Moderate | Much larger, generative |
| **VL-JEPA** | **JEPA VL** | **46.4% avg** | **58.4% avg** | **Predictive, not generative** |

### 7.5 Key Comparisons

**VL-JEPA vs. CLIP/SigLIP2/Perception Encoder (contrastive models)**:
- VL-JEPA adds a predictor between the visual encoder and the contrastive loss, enabling more complex cross-modal mapping.
- VL-JEPA uses a frozen visual encoder (V-JEPA 2), while CLIP trains both encoders from scratch on paired data.
- VL-JEPA benefits from V-JEPA 2's self-supervised video pretraining, which provides temporal understanding that CLIP (trained on images) lacks.

**VL-JEPA vs. InstructBLIP/Qwen-VL (generative VLMs)**:
- VL-JEPA does NOT generate text token-by-token. It predicts text embeddings.
- VL-JEPA is much smaller (~1.1B total vs. 7B+ for generative VLMs).
- VL-JEPA is faster at inference for classification/retrieval (selective decoding).
- Generative VLMs are better at open-ended generation tasks.
- VL-JEPA's approach is more aligned with the JEPA philosophy of prediction in abstract space.

---

## 8. Architecture Dimensions and Counts

### 8.1 X-Encoder (V-JEPA 2 ViT-L)

| Property | Value |
|----------|-------|
| Layers | 24 |
| Hidden dim | 1024 |
| Attention heads | 16 |
| MLP dim | 4096 |
| Patch size | 16x16 spatial, 2 temporal |
| Parameters | 304M |

### 8.2 Predictor (Llama-3.2-1B Last 8 Layers)

| Property | Value |
|----------|-------|
| Layers | 8 (last 8 of 16 total in Llama-3.2-1B) |
| Hidden dim | 2048 |
| Attention heads | 32 |
| Key/value heads | 8 (GQA) |
| MLP dim | 8192 |
| Vocabulary | Not used (no text generation) |
| Parameters | ~490M |

Note: The predictor uses Llama's architecture but does NOT use its vocabulary or token embeddings. The input is projected visual features, not text tokens.

### 8.3 Y-Encoder (EmbeddingGemma-300M)

| Property | Value |
|----------|-------|
| Architecture | Gemma-based text embedding model |
| Parameters | 300M |
| Output dim | Embedding dimension matching predictor output |
| Input | Tokenized text |
| Max sequence length | Model-dependent |

---

## 9. Training Infrastructure

### 9.1 Pretraining

- Large-scale video-text pair datasets.
- Distributed training across multiple GPUs.
- Mixed precision (bfloat16 / float16) for efficiency.
- Gradient accumulation for effective large batch sizes (important for InfoNCE).

### 9.2 Key Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Predictor learning rate | Full LR |
| Y-Encoder learning rate | 0.05x predictor LR |
| X-Encoder learning rate | 0 (frozen) |
| InfoNCE temperature | Learnable |
| Batch size | Large (InfoNCE benefits from more negatives) |
| Optimizer | AdamW (likely) |

---

## 10. Significance and Novelty

### 10.1 What VL-JEPA Proves

1. **JEPA extends to cross-modal learning**: The predict-in-latent-space paradigm works not just within a modality (image patches predicting image patches) but across modalities (video features predicting text embeddings).

2. **Frozen visual encoders work**: You do not need to jointly train the visual encoder with the language objective. V-JEPA 2's features are good enough to use frozen.

3. **Predictive > purely contrastive for cross-modal alignment**: The addition of a predictor (vs. CLIP's direct contrastive approach) enables richer cross-modal mappings.

4. **Selective decoding is practical**: Ward clustering enables significant inference speedups for classification and retrieval tasks.

### 10.2 Limitations

1. **Not generative**: VL-JEPA cannot generate free-form text the way InstructBLIP or Qwen-VL can. It predicts embeddings, not tokens.

2. **Dependent on Y-Encoder quality**: The quality of text embeddings from EmbeddingGemma-300M upper-bounds what the predictor can learn.

3. **Selective decoding only works for closed-set tasks**: Open-ended generation requires a different approach.

4. **Benchmark averages can mask variance**: 46.4% average across 8 benchmarks -- some individual benchmarks may be significantly higher or lower.

### 10.3 Open Research Directions

- Scaling the predictor (more layers, larger Llama variant).
- Using the full V-JEPA 2 1.2B ViT-g instead of ViT-L.
- Combining JEPA-style prediction with generative decoding for open-ended tasks.
- Extending to audio-visual-language (three modalities).
- Improving selective decoding with better clustering algorithms.
- Training with higher-resolution video.

---

## 11. How to Think About VL-JEPA

### Mental Model

Think of VL-JEPA as answering the question: "Given a video, can I predict what text describes it, without ever generating text token-by-token?"

The answer is yes: by predicting a text embedding (a compressed semantic summary of the text) from visual features, and then comparing that predicted embedding against candidate text embeddings.

### Position in the JEPA Lineage

```
LeCun 2022 (theory)
    |
    v
I-JEPA (images, 2023)
    |
    v
V-JEPA (video, 2024)
    |
    v
V-JEPA 2 (scaled video + robotics, 2025)
    |
    v
VL-JEPA (video + language, 2025)  <-- we are here
```

Each step adds a dimension:
- I-JEPA: spatial prediction in latent space.
- V-JEPA: spatiotemporal prediction in latent space.
- V-JEPA 2: autoregressive prediction + action conditioning.
- VL-JEPA: cross-modal prediction (visual to language embedding space).

---

*Last updated: 2026-03-07*
