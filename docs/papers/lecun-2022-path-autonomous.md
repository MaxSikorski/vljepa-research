# Annotated Notes: "A Path Towards Autonomous Machine Intelligence"

**Author**: Yann LeCun
**Date**: June 27, 2022 (v0.9.2)
**Venue**: Position paper (published on OpenReview)
**Link**: [https://openreview.net/pdf?id=BZ5a1r-kVsf](https://openreview.net/pdf?id=BZ5a1r-kVsf)

---

## 1. Core Thesis

LeCun argues that current AI systems -- including large language models and generative image models -- are fundamentally limited because they operate in high-dimensional input space (pixels, tokens) rather than in compact abstract representations. The paper lays out a blueprint for how machines could learn world models that support reasoning, planning, and acting at multiple levels of abstraction.

### Why Generative Models Are Wasteful

The central argument against generative approaches (autoregressive LLMs, diffusion models, GANs for pixel prediction):

- **Prediction in pixel/token space is intractable**: The world is high-dimensional and stochastic. Predicting the exact RGB values of the next video frame requires modeling every irrelevant detail (exact leaf positions, cloud shapes, lighting variations). This is a waste of capacity.
- **Generative models must allocate capacity to irrelevant detail**: A model that predicts pixels must represent the full distribution of possible outputs, including all the variation that is irrelevant to the task at hand.
- **The curse of dimensionality**: As output dimensionality grows, the volume of possible outputs explodes. Modeling this full distribution accurately becomes exponentially harder.
- **Contrast with biological systems**: Humans do not predict the exact pixel content of their visual field. We predict abstract outcomes -- "the ball will land over there" -- without specifying every photon.

> Key insight: The amount of information humans extract from visual input in a single glance vastly exceeds what language conveys. Video understanding cannot be solved by converting everything to tokens and predicting the next one.

### Why Prediction in Latent Space Is Better

Instead of predicting y from x in input space, predict s_y from s_x in a learned abstract representation space:

```
Traditional:  x -> Encoder -> Decoder -> y_hat  (predict in pixel space)
JEPA:         x -> Encoder_x -> s_x -> Predictor -> s_y_hat
              y -> Encoder_y -> s_y
              Loss: distance(s_y_hat, s_y)  (predict in latent space)
```

Benefits:
- The encoder can discard irrelevant information, making prediction tractable.
- Multiple valid futures (ball goes left OR right) can be represented as a set of compatible latent states rather than requiring a distribution over all possible pixel configurations.
- The representation is learned end-to-end to retain only what matters for prediction.

---

## 2. The Proposed Architecture

LeCun proposes a modular cognitive architecture with six components. This is the conceptual blueprint that eventually leads to JEPA-family models.

### 2.1 World Model

The world model is the centerpiece. It predicts the consequences of actions (or the natural evolution of the world) in abstract representation space.

Two sub-components:
- **Encoder**: Maps observations (percepts) to latent state representations.
- **Predictor**: Given a latent state and an action, predicts the next latent state.

The world model does NOT predict pixels. It predicts in the space of abstract representations.

### 2.2 Cost Module (Critic)

Computes a scalar cost (or energy) that measures how "good" or "bad" a predicted state is relative to objectives. This is learned or specified:
- **Intrinsic cost**: Hardwired objectives (analogous to pain, hunger, curiosity).
- **Critic**: A learned module that estimates future cumulative cost from the current state. Analogous to the value function in RL, but operating on latent states.

### 2.3 Actor

The actor computes action sequences that minimize the expected cost as estimated by the world model + critic. This is planning -- the actor uses the world model to simulate future trajectories and picks the action sequence with the lowest cost.

### 2.4 Short-Term Memory

Stores recent percepts and predicted states. Enables the system to reason about temporal sequences without re-encoding everything from scratch.

### 2.5 Perception Module

Estimates the current state of the world from sensory input. This is essentially the encoder portion of the world model, but LeCun separates it conceptually because it includes preprocessing and attention mechanisms.

### 2.6 Configurator

This is the most speculative component. The configurator modulates the behavior of all other modules depending on the current task:
- Adjusts the cost module to reflect current objectives.
- Primes the perception module to attend to relevant features.
- Sets the time horizon and precision for the world model's predictions.
- Configures the actor's planning depth.

The configurator is analogous to executive control / prefrontal cortex function. It is what makes the system "autonomous" -- able to set its own goals and sub-goals.

### Architecture Diagram (Simplified)

```
Observation --> [Perception] --> state_t
                                    |
                                    v
                    [World Model Predictor] <-- action_t (from Actor)
                                    |
                                    v
                              state_{t+1} (predicted)
                                    |
                                    v
                            [Cost Module / Critic]
                                    |
                                    v
                                  cost
                                    |
                                    v
                                [Actor] --> action_t+1
                                    ^
                                    |
                            [Configurator] (modulates everything)
```

---

## 3. JEPA: Joint-Embedding Predictive Architecture

This is the concrete learning framework proposed for building the world model.

### What "Joint-Embedding" Means

Both the input x and the target y are mapped to the same abstract embedding space by their respective encoders. The system never reconstructs the input -- it only operates in embedding space.

Contrast with:
- **Autoencoders / MAE**: Reconstruct the input from a latent code. Operate in pixel space.
- **Contrastive learning (SimCLR, CLIP)**: Map both inputs to embeddings, but use contrastive loss to pull positives together and push negatives apart. Requires explicit negatives.
- **JEPA**: Map both inputs to embeddings, but use a predictor to map from one embedding to another. The loss is purely on the quality of the prediction in embedding space.

### What "Predictive" Means

There is an explicit predictor module that takes the embedding of x (and possibly additional context like a mask or action) and predicts the embedding of y. This is NOT just a similarity metric -- it involves a learned transformation.

```
JEPA Architecture:
  x --> Encoder_x --> s_x --> Predictor(s_x, z) --> s_y_hat
  y --> Encoder_y --> s_y
  Loss: D(s_y, s_y_hat)   where D is a distance in latent space
```

Here z is an optional latent variable that can capture the inherent uncertainty (multiple valid predictions). This is crucial for handling multimodal futures.

### Why JEPA Is Not Contrastive

Contrastive methods require negative samples and can be brittle:
- Hard to mine good negatives.
- Computational cost scales with batch size.
- Can lead to dimensional collapse if negatives are not diverse enough.

JEPA avoids the need for negatives entirely. The challenge shifts to avoiding a different failure mode: representation collapse.

---

## 4. Energy-Based Models and Avoiding Collapse

### The Collapse Problem

If both encoders and the predictor are trained jointly to minimize prediction error in latent space, there is a trivial solution: map everything to the same constant vector. Then the prediction error is always zero, but the representation is useless.

This is called **representation collapse** or **dimensional collapse**.

### Energy-Based Model (EBM) Perspective

LeCun frames JEPA as an energy-based model:
- **Energy function** E(x, y) = D(Enc_y(y), Predictor(Enc_x(x), z))
- Compatible (x, y) pairs should have low energy.
- Incompatible pairs should have high energy.
- The challenge: how to ensure high energy for incompatible pairs without explicitly pushing them up (without contrastive negatives)?

### Four Strategies to Avoid Collapse

1. **Contrastive methods**: Explicitly push up energy for negative pairs. Works but requires negatives. (SimCLR, MoCo, CLIP)

2. **Regularization methods**: Add terms to the loss that prevent the representation from collapsing. This is the preferred approach. (VICReg, Barlow Twins)

3. **Architectural constraints**: Design the architecture so that collapse is impossible (e.g., use a very high-dimensional representation with constraints on its structure).

4. **Latent variable with informative prior**: Use the latent variable z to capture variation, with a prior that prevents it from being ignored.

LeCun advocates strongly for approach (2) -- regularization methods -- combined with (4) -- latent variables for multi-modal predictions.

---

## 5. VICReg: Variance-Invariance-Covariance Regularization

VICReg (Bardes, Ponce, LeCun, 2022) is the specific regularization method that prevents collapse in JEPA-family models.

### Three Components

1. **Variance**: Each dimension of the embedding must maintain a minimum standard deviation across the batch. This prevents all samples from mapping to the same point.

   ```
   L_var = (1/d) * sum_j max(0, gamma - sqrt(Var(s_j) + epsilon))
   ```
   where s_j is the j-th dimension across the batch, and gamma is a threshold (typically 1).

2. **Invariance**: The embeddings of compatible pairs (x, y) should be similar. This is the standard prediction/matching objective.

   ```
   L_inv = (1/n) * sum_i ||s_x_i - s_y_i||^2
   ```

3. **Covariance**: Off-diagonal elements of the embedding covariance matrix should be zero. This decorrelates the embedding dimensions, preventing them from encoding redundant information.

   ```
   L_cov = (1/d) * sum_{i != j} Cov(s_i, s_j)^2
   ```

### Why VICReg Matters for the JEPA Lineage

- VICReg provides the mathematical machinery to train JEPA-style models without contrastive negatives.
- It directly influences how I-JEPA, V-JEPA, and V-JEPA 2 maintain representation quality.
- In later models (I-JEPA onward), the target encoder uses EMA rather than VICReg directly, but the design philosophy of "regularize to prevent collapse, don't use negatives" persists.
- V-JEPA 2 and VL-JEPA use InfoNCE loss (which is technically contrastive), showing that the field has evolved past strict non-contrastive approaches where practical gains warrant it.

---

## 6. Hierarchical Planning Across Time Scales

### The Multi-Scale Problem

Real-world planning operates at multiple time scales:
- **Low-level** (milliseconds): Motor control, muscle activations.
- **Mid-level** (seconds): Reaching for an object, taking a step.
- **High-level** (minutes to hours): Navigating to a destination, cooking a meal.

### Hierarchical JEPA

LeCun proposes stacking JEPA modules hierarchically:

```
Level 3 (abstract):  s3_t --> Predictor_3 --> s3_{t+T3}    [plan: "go to kitchen"]
                        |                         |
Level 2 (mid):        s2_t --> Predictor_2 --> s2_{t+T2}    [plan: "walk through door"]
                        |                         |
Level 1 (concrete):   s1_t --> Predictor_1 --> s1_{t+T1}    [plan: "move left foot"]
```

Key properties:
- Higher levels predict further into the future.
- Higher levels operate on more abstract representations.
- Higher levels have lower temporal resolution.
- Each level can plan at its own time scale.
- Lower levels fill in the details left unspecified by higher levels.

### Connection to V-JEPA 2 and Robotics

This hierarchical planning vision is directly realized in V-JEPA 2-AC (action-conditioned):
- The world model predicts future latent states given actions.
- Model Predictive Control (MPC) plans action sequences by simulating trajectories through the world model.
- The robot (Franka arm) executes plans in a zero-shot manner -- the world model learned from passive video enables physical control without task-specific training.

---

## 7. Key Quotes and Page References

| Page | Quote / Concept | Significance |
|------|----------------|-------------|
| p.3 | The "one learning algorithm" hypothesis | Motivation for a unified architecture |
| p.4 | Animals learn vast amounts of background knowledge about the world through observation | Justification for self-supervised learning from video |
| p.7 | Generative models are doomed to waste resources on irrelevant details | Core argument for latent prediction |
| p.8 | JEPA architecture diagram | The foundational diagram for the entire model family |
| p.12-14 | Energy-based model formulation | Mathematical framework for understanding JEPA training |
| p.16-18 | VICReg and regularization strategies | How to avoid collapse without contrastive learning |
| p.20-24 | Hierarchical planning with world models | Blueprint for V-JEPA 2-AC and robotics applications |
| p.25 | Configurator module | Most speculative component; not yet realized in published work |

---

## 8. Connection to VL-JEPA's Actual Implementation

This position paper is the philosophical ancestor of the entire JEPA model family. Here is how each idea maps to implementations:

| LeCun 2022 Concept | Implementation in Practice |
|---|---|
| Prediction in latent space (not pixel space) | I-JEPA, V-JEPA, V-JEPA 2 all predict in embedding space, never reconstruct pixels |
| Joint-embedding architecture | All JEPA models use separate encoders for context and target, operating in shared embedding space |
| Avoid collapse via regularization | I-JEPA/V-JEPA use EMA target encoder; VL-JEPA uses InfoNCE |
| World model predicts consequences of actions | V-JEPA 2-AC conditions predictions on robot actions |
| Hierarchical planning | V-JEPA 2-AC uses MPC with the world model for robot control |
| Configurator modulates system behavior | Not directly implemented; closest analogue is the task-specific fine-tuning or prompting in VL-JEPA |
| Cost module evaluates states | In V-JEPA 2-AC, the cost/reward is task-specific (e.g., distance to goal state) |
| Actor plans via world model | MPC in V-JEPA 2-AC generates action sequences by optimizing through the world model |

### What the Paper Got Right

- Prediction in latent space is now the dominant paradigm for visual self-supervised learning.
- JEPA-style architectures outperform MAE and contrastive methods on many benchmarks.
- Video understanding benefits enormously from not having to predict pixels.

### What Remains Unresolved

- The configurator module has no concrete implementation.
- Hierarchical JEPA at multiple abstraction levels remains mostly theoretical.
- The paper does not address language integration -- VL-JEPA had to solve this separately.
- Energy-based models with latent variables for multi-modal prediction are still an active research area.

---

## 9. Reading Recommendations

If you are new to this area, read in this order:
1. Sections 1-3 of this paper (motivation and architecture overview)
2. The I-JEPA paper (first concrete implementation)
3. Section 4 of this paper (JEPA formalization)
4. The V-JEPA / V-JEPA 2 papers (video extension)
5. Sections 5-6 of this paper (hierarchical planning)
6. The VL-JEPA paper (language integration)

---

*Last updated: 2026-03-07*
