# VL-JEPA Research Lab

A comprehensive research lab for studying, implementing, and commercializing **VL-JEPA** (Vision-Language Joint Embedding Predictive Architecture) and its family of models — including the first public implementation of **Apple's SALT** (Static-teacher Asymmetric Latent Training, ICLR 2025).

## Highlights

- **112 unit tests** passing in Docker
- **6 Docker containers** for tests, training, evaluation, demos, notebooks, and SALT
- **SALT implementation** — first public implementation of Apple's ICLR 2025 paper
- **I-JEPA → V-JEPA → VL-JEPA** full progression implemented
- **End-to-end verified**: CIFAR-10 train → checkpoint → eval pipelines for both I-JEPA and SALT

## What is VL-JEPA?

VL-JEPA is a non-generative vision-language model from Meta AI/FAIR that predicts semantic embeddings in abstract representation space rather than generating tokens word-by-word. Key advantages:

- **1.6B parameters** — 50% fewer than comparable VLMs
- **2.85x faster inference** via selective decoding
- **Surpasses CLIP, SigLIP2** on video understanding tasks
- **Matches 7B-13B models** on VQA with a fraction of the compute

### VL-JEPA Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       VL-JEPA                            │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  X-Encoder   │  │  Predictor   │  │   Y-Encoder   │  │
│  │  V-JEPA 2    │──│  Llama-3.2   │──│  Gemma-300M   │  │
│  │  ViT-L       │  │  Last 8 Lyrs │  │  Text Embed   │  │
│  │  304M params  │  │  490M params │  │  300M params  │  │
│  │  (FROZEN)    │  │  (TRAINABLE) │  │  (0.05x LR)   │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  Loss: Bi-directional InfoNCE in 1536-d shared space     │
└─────────────────────────────────────────────────────────┘
```

## SALT: Apple's Next-Gen Visual Encoder Training (ICLR 2025)

**SALT** (Static-teacher Asymmetric Latent Training) is Apple's method that **eliminates EMA** from JEPA training while **outperforming V-JEPA 2** with ~30% fewer FLOPs. This repo contains the first public implementation.

### How SALT Works

```
Stage 1 — V-Pixel (MAE Teacher Training):
  ViT Encoder + Decoder → pixel reconstruction (MSE on masked patches)
  No EMA, no target encoder. Decoder is discarded after training.

Stage 2 — Frozen-Teacher JEPA (Student Training):
  Freeze Stage 1 teacher completely
  Train Student + Predictor → L1 loss on teacher's latent representations
  No EMA, no stop-gradient. "Weak teacher, strong student."
```

### SALT + VL-JEPA Integration

SALT and VL-JEPA are **complementary** — SALT replaces how the visual encoder is pretrained:

```
Before:  V-JEPA 2 (Meta, EMA) → frozen X-Encoder → VL-JEPA
After:   SALT (Apple, no EMA) → frozen X-Encoder → VL-JEPA
```

This eliminates dependency on Meta's non-commercial V-JEPA 2 weights and enables training custom domain-specific encoders (e.g., for robotics).

### Verified SALT Results (CIFAR-10 Smoke Test)

| Metric | SALT | I-JEPA | Winner |
|--------|------|--------|--------|
| k-NN accuracy | **29.2%** | 26.2% | SALT |
| Linear probe accuracy | **37.8%** | 35.2% | SALT |
| Has EMA? | No | Yes | SALT |
| Stage 1 loss | 0.8406 → 0.7609 | — | Decreasing |
| Stage 2 loss | 0.5093 → 0.4259 | — | Decreasing |

Reference: [SALT paper (arXiv:2509.24317)](https://arxiv.org/abs/2509.24317) — Apple Research, ICLR 2025

## JEPA Family Evolution

| Model | Year | Domain | Params | Repository |
|-------|------|--------|--------|------------|
| I-JEPA | 2023 | Images | ~632M | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| V-JEPA | 2024 | Video | ~300M+ | [facebookresearch/jepa](https://github.com/facebookresearch/jepa) |
| V-JEPA 2 | Jun 2025 | Video + Robotics | 1.2B | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| VL-JEPA | Dec 2025 | Vision + Language | 1.6B | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) |
| **SALT** | **2025** | **Video (no EMA)** | **ViT-L/G** | **[arXiv:2509.24317](https://arxiv.org/abs/2509.24317)** |

## Quickstart (Everything Runs in Docker)

### Run All Tests

```bash
docker build -t vljepa-tests -f Dockerfile.test .
docker run --rm vljepa-tests
# → 112 passed in ~15s
```

### Run SALT End-to-End (Stage 1 → Stage 2 → Eval → I-JEPA Comparison)

```bash
docker build -t vljepa-salt -f Dockerfile.salt .
docker run --rm vljepa-salt
# → SALT vs I-JEPA side-by-side comparison on CIFAR-10
```

### Run I-JEPA End-to-End Training

```bash
docker build -t vljepa-e2e -f Dockerfile.e2e .
docker run --rm vljepa-e2e
# → Train → save → reload → k-NN + linear probe evaluation
```

### Run Inference Demo

```bash
docker build -t vljepa-demo -f Dockerfile.demo .
docker run --rm vljepa-demo
# → Feature similarity + k-NN + VL-JEPA pipeline demo
```

### Docker Containers

| Container | Purpose | Time |
|-----------|---------|------|
| `vljepa-tests` | 112 unit tests | ~15s |
| `vljepa-salt` | Full SALT pipeline + I-JEPA comparison | ~40min |
| `vljepa-e2e` | I-JEPA train → eval on CIFAR-10 | ~17min |
| `vljepa-demo` | Inference demos | ~30s |
| `vljepa-notebooks` | Jupyter notebook execution | ~2min |
| `vljepa-train` | GPU training (A100/H100) | varies |

### Project Structure

```
src/
├── common/        # Shared utilities (config, logging, distributed, checkpointing)
├── ijepa/         # Image JEPA (encoder, predictor, masking, evaluation)
├── vjepa/         # Video JEPA (3D RoPE, tubelet embedding, spatiotemporal)
├── vljepa/        # Vision-Language JEPA (X-Encoder, Predictor, Y-Encoder, InfoNCE)
├── salt/          # SALT (Apple ICLR 2025) — MAE decoder, pixel loss, Stage 1+2 trainers
└── robotics/      # Embodied AI / action-conditioned extensions

configs/
├── ijepa/         # I-JEPA training configs
├── vjepa/         # V-JEPA training configs
├── vljepa/        # VL-JEPA training configs
├── salt/          # SALT Stage 1 + Stage 2 configs
└── robotics/      # Robotics configs

tests/             # 112 tests across 11 test modules
scripts/           # E2E smoke tests, demos, evaluation scripts
```

### File Counts

| Category | Count |
|----------|-------|
| Python source files | 65 |
| YAML configs | 11 |
| Dockerfiles | 8 |
| Test modules | 11 |
| Unit tests | 112 |

## Research Roadmap

1. **Weeks 1-8**: Theoretical foundations + code study + local experiments
2. **Weeks 9-14**: I-JEPA reproduction (foundational patterns)
3. **Weeks 15-22**: V-JEPA video extension
4. **Weeks 23-34**: VL-JEPA full implementation (primary target)
5. **Weeks 35-46**: Robotics & embodied AI + **SALT integration**
6. **Weeks 47+**: Production deployment & commercialization

## Key Papers

- LeCun (2022) — [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- I-JEPA (CVPR 2023) — [Self-Supervised Learning from Images with a JEPA](https://arxiv.org/abs/2301.08243)
- V-JEPA 2 (2025) — [Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- VL-JEPA (2025) — [Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)
- **SALT (ICLR 2025) — [Rethinking JEPA: Static-teacher Asymmetric Latent Training](https://arxiv.org/abs/2509.24317)** — Apple Research

## Technical Notes

- **SALT eliminates EMA** — the most complex component of JEPA training loops
- **"Weak teacher, strong student"** — a smaller teacher (ViT-L) can train a larger student (ViT-G)
- **Student loss predicts downstream accuracy** (R² = 0.951) — enables cheap model selection
- **Multi-block masking** outperforms random masking for pixel reconstruction (validates our I-JEPA implementation)
- **V-JEPA 2 uses L1 loss** (not smooth_l1), same as SALT Stage 2
- **HuggingFace V-JEPA 2**: `AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`

## License

This research lab code is proprietary. See `docs/licensing-analysis.md` for upstream dependency licenses.
