# GPU Cost Analysis for VL-JEPA Research Lab

## Local Development (M4 Mac — Available Now)

| Capability | Feasible? | Notes |
|-----------|-----------|-------|
| Code development | Yes | Full IDE, debugging, git |
| Unit tests | Yes | All tests run on CPU/MPS |
| Tiny model training | Yes | ViT-Tiny on CIFAR-10, ~30 min |
| Pretrained inference | Yes | V-JEPA 2 ViT-L via MPS/MLX |
| Full-scale training | No | Requires dedicated GPU cluster |

## Cloud GPU Pricing (As of March 2026)

### A100 80GB (Primary Training GPU)

| Provider | On-Demand | Spot/Interruptible |
|----------|-----------|-------------------|
| Vast.ai | $0.78-1.20/hr | $0.50-0.80/hr |
| RunPod | $1.09/hr | $0.79/hr |
| Lambda Cloud | $1.29/hr | N/A |
| AWS p4d.24xlarge (8x) | $32.77/hr ($4.10/GPU) | $13-20/hr |
| GCP a2-highgpu-1g | $3.67/hr | $1.10/hr |

### H100 80GB (Premium, 2-3x faster than A100)

| Provider | On-Demand | Spot/Interruptible |
|----------|-----------|-------------------|
| RunPod | $2.49/hr | $1.99/hr |
| Lambda Cloud | $2.49/hr | N/A |
| CoreWeave | $4.76/hr | N/A |

## Phase-by-Phase Cost Estimates

### Phase 1: I-JEPA (Weeks 9-14)

| Experiment | GPUs | Hours | Cost Range |
|-----------|------|-------|------------|
| ViT-Tiny debugging | 1x A100 | 4 | $3-5 |
| ViT-B/16 on ImageNet-100 | 1x A100 | 48-72 | $40-90 |
| ViT-L/16 on ImageNet-1K | 4x A100 | 50-100 | $160-480 |
| **Phase 1 Total** | | | **$200-$575** |

### Phase 2: V-JEPA (Weeks 15-22)

| Experiment | GPUs | Hours | Cost Range |
|-----------|------|-------|------------|
| Video ViT-B debugging | 1x A100 | 8 | $6-10 |
| ViT-L on Kinetics-400 subset | 4x A100 | 100-200 | $320-960 |
| Full V-JEPA reproduction | 4x A100 | 200-400 | $640-1920 |
| **Phase 2 Total** | | | **$960-$2,890** |

### Phase 3: VL-JEPA (Weeks 23-34)

| Experiment | GPUs | Hours | Cost Range |
|-----------|------|-------|------------|
| Smoke test (tiny models) | 1x A100 | 4 | $3-5 |
| Pretraining on CC3M | 4x A100 | 200-400 | $640-1920 |
| Pretraining on CC3M+WebVid | 8x A100 | 300-600 | $1920-5760 |
| SFT on VQA datasets | 4x A100 | 100-200 | $320-960 |
| Ablation experiments | 2x A100 | 100-200 | $160-480 |
| **Phase 3 Total** | | | **$3,040-$9,125** |

### Phase 4: Robotics (Weeks 35-46)

| Experiment | GPUs | Hours | Cost Range |
|-----------|------|-------|------------|
| eb_jepa Two Rooms | 1x A100 | 4-8 | $3-10 |
| AC predictor training | 2x A100 | 100-200 | $160-480 |
| MPC planning experiments | 1x A100 | 50-100 | $40-120 |
| Simulation runs | 1x A100 | 100-200 | $80-240 |
| **Phase 4 Total** | | | **$280-$850** |

### Total Year 1

| Scenario | Estimate |
|----------|----------|
| Minimum (spot, aggressive) | ~$4,500 |
| Expected (mixed spot/demand) | ~$10,000 |
| Maximum (on-demand, thorough) | ~$13,500 |

## Cost Optimization Strategies

1. **Spot instances**: Use interruptible/spot instances with checkpointing every 30 min
2. **Gradient checkpointing**: Trade compute for memory — train larger batches on fewer GPUs
3. **Mixed precision (BF16)**: 2x memory savings, ~1.5x speedup on A100/H100
4. **FSDP**: Shard model across GPUs for efficient multi-GPU training
5. **Start small**: Train tiny models locally, medium on 1 GPU, scale only when validated
6. **Pre-trained models**: Use Meta's V-JEPA 2 checkpoints instead of training from scratch
7. **Off-peak hours**: Cloud GPU prices often lower during off-peak times

## Recommendations

- **Start with Vast.ai or RunPod** for best price-performance on individual experiments
- **Consider Lambda Cloud** for extended multi-week training runs (reserved pricing)
- **Use spot instances** for all exploratory work; on-demand only for final training runs
- **Checkpoint aggressively** — save every 30 minutes, resume from latest on preemption
