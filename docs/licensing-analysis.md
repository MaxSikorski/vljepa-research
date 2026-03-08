# Licensing Analysis: VL-JEPA Stack Components

**Prepared for**: VL-JEPA Research Lab
**Last updated**: 2026-03-07

---

## 1. Component License Summary Table

| Component | Version / Variant | License | Commercial Use? | Key Restriction | Repository |
|-----------|-------------------|---------|----------------|-----------------|------------|
| I-JEPA | ViT-H/14 | CC BY-NC 4.0 | **NO** | Non-commercial only | github.com/facebookresearch/ijepa |
| V-JEPA | ViT-L | CC BY-NC 4.0 | **NO** | Non-commercial only | github.com/facebookresearch/jepa |
| V-JEPA 2 | ViT-L / ViT-g | MIT License (primary) + some Apache 2.0 files | **YES** | Minimal restrictions | github.com/facebookresearch/jepa |
| Llama-3.2-1B | 1B params | Llama 3.2 Community License | **YES, with conditions** | 700M MAU threshold; no Llama-to-train-other-LLM clause | llama.meta.com |
| EmbeddingGemma-300M | 300M params | Gemma Terms of Use | **YES, with conditions** | Acceptable use policy; redistribution rules | ai.google.dev/gemma |

---

## 2. Detailed License Analysis

### 2.1 I-JEPA: CC BY-NC 4.0 (Non-Commercial)

**License**: Creative Commons Attribution-NonCommercial 4.0 International
**Full text**: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

**What this means:**
- You MAY use, share, and adapt the model weights and code for **non-commercial purposes only**.
- You MUST give appropriate credit (attribution).
- You MUST indicate if changes were made.
- You MAY NOT use the material for commercial purposes.
- "Commercial purposes" means primarily intended for or directed toward commercial advantage or monetary compensation.

**Practical impact:**
- Research and academic use: ALLOWED.
- Internal prototyping: ALLOWED (as long as not deployed commercially).
- Commercial products: NOT ALLOWED.
- Consulting work for paying clients: GRAY AREA -- likely not allowed if the work product is commercially deployed.

**Note for our lab**: I-JEPA is a predecessor to V-JEPA 2. For any work that may become commercial, use V-JEPA 2 (MIT licensed) instead. The I-JEPA weights are primarily useful for reproducing the original paper's results and for educational purposes.

---

### 2.2 V-JEPA (v1): CC BY-NC 4.0 (Non-Commercial)

**License**: Creative Commons Attribution-NonCommercial 4.0 International
**Full text**: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

**Identical restrictions to I-JEPA above.**

**Practical impact:**
- Same as I-JEPA: research and academic use only.
- V-JEPA (v1) is superseded by V-JEPA 2 in capability.
- No reason to use V-JEPA v1 in any commercial context.

---

### 2.3 V-JEPA 2: MIT License (Primary) + Apache 2.0 (Some Files)

**Primary license**: MIT License
**Full text**: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

**Some files**: Apache License 2.0
**Full text**: [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

**What the MIT License allows:**
- Commercial use: YES.
- Modification: YES.
- Distribution: YES.
- Private use: YES.
- Only requirement: Include the original copyright notice and license text in copies/distributions.

**What Apache 2.0 allows (for applicable files):**
- Same freedoms as MIT, plus:
- Explicit patent grant (protects users from patent claims by contributors).
- Requirement to state changes if you modify the code.
- Must include the NOTICE file if one exists.

**Practical impact:**
- V-JEPA 2 is the most permissively licensed model in the JEPA family.
- You CAN use it in commercial products.
- You CAN modify and redistribute it.
- You CAN build proprietary systems on top of it.
- You MUST include the license text when distributing.
- Check individual file headers for whether MIT or Apache 2.0 applies.

**This is the visual backbone you should use for commercial work.**

---

### 2.4 Llama-3.2-1B: Llama 3.2 Community License

**License**: Llama 3.2 Community License Agreement
**Full text**: [https://www.llama.com/llama3_2/license/](https://www.llama.com/llama3_2/license/)

**Key terms:**

1. **Commercial use**: ALLOWED, with conditions.

2. **Monthly Active User (MAU) threshold**: If your product or service has more than **700 million monthly active users** in the preceding calendar month, you must request a license from Meta. Below 700M MAU, the license is effectively permissive for commercial use.

3. **Acceptable Use Policy**: You must comply with Meta's Acceptable Use Policy, which prohibits uses such as:
   - Weapons development.
   - Surveillance and privacy violations.
   - Generating disinformation.
   - Other harmful applications (detailed in the AUP).

4. **Attribution**: You must include "Built with Llama" in any product or service that uses Llama models, and in any related documentation.

5. **No using Llama to train competing LLMs**: You may NOT use Llama outputs to improve or train any other large language model (other than Llama derivatives). This is the "no model distillation to competitors" clause.

6. **Redistribution**: If you redistribute Llama or derivatives:
   - Must include the license agreement.
   - Must provide a copy of Meta's Acceptable Use Policy.
   - Must include "Built with Llama" attribution.

**Practical impact for VL-JEPA:**
- The Llama-3.2-1B serves as the predictor (last 8 layers). This is a **modification** of the Llama model (using only a subset of layers for a non-text-generation purpose).
- Commercial deployment: ALLOWED (assuming < 700M MAU, which is virtually certain for a research lab).
- The "no training competing LLMs" clause does NOT apply to VL-JEPA because VL-JEPA is not a language model -- it is a vision-language embedding model.
- You MUST include "Built with Llama" attribution.
- You MUST comply with the Acceptable Use Policy.

**Risk assessment**: LOW. The 700M MAU threshold is unreachable for most organizations. The AUP restrictions are standard ethical guardrails. The anti-distillation clause does not apply to VL-JEPA's use case.

---

### 2.5 EmbeddingGemma-300M: Gemma Terms of Use

**License**: Gemma Terms of Use
**Full text**: [https://ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)

**Key terms:**

1. **Commercial use**: ALLOWED, with conditions.

2. **Acceptable Use Policy**: You must comply with Google's Gemma Prohibited Use Policy, which includes restrictions on:
   - Generating harmful or illegal content.
   - Harassment, surveillance, disinformation.
   - Military and weapons applications (with exceptions for legitimate defense/security).

3. **Redistribution**: If you redistribute Gemma or derivatives:
   - Must include the Gemma Terms of Use.
   - Must include the Gemma Prohibited Use Policy.
   - Must include a notice about model modifications.

4. **No trademark use**: You may not use Google's trademarks (Gemma, Google, etc.) to endorse or promote your products without permission.

5. **Generated output ownership**: Google does not claim ownership of outputs generated by Gemma models.

6. **No additional restrictions**: You may not impose legal terms or technological measures on others that restrict them beyond what the Gemma Terms allow.

**Practical impact for VL-JEPA:**
- EmbeddingGemma-300M serves as the Y-Encoder (text embedding target). It is a component of the training pipeline.
- Commercial deployment: ALLOWED.
- You MUST comply with the Prohibited Use Policy (standard ethical restrictions).
- You MUST include Gemma Terms of Use when redistributing.
- You should NOT claim Google endorsement.

**Risk assessment**: LOW. The terms are relatively permissive. The Prohibited Use Policy is similar to Llama's AUP -- standard ethical guardrails.

---

## 3. What You CAN Do Commercially

Given the license landscape, here is what is commercially viable:

### 3.1 Fully Commercial (No Restrictions Beyond Attribution)

- **Use V-JEPA 2 as a visual encoder** in any commercial product.
  - MIT-licensed. Include the license text. Done.

### 3.2 Commercial with Conditions

- **Use the full VL-JEPA stack** (V-JEPA 2 + Llama predictor + EmbeddingGemma) in commercial products, provided:
  - Your product has fewer than 700M MAU (Llama condition).
  - You include "Built with Llama" attribution.
  - You comply with Meta's Acceptable Use Policy.
  - You comply with Google's Gemma Prohibited Use Policy.
  - You include relevant license texts when distributing.

- **Fine-tune and deploy** VL-JEPA for commercial video understanding tasks.

- **Build proprietary extensions** on top of the VL-JEPA architecture.

- **Offer VL-JEPA-based services** (API, SaaS, consulting).

### 3.3 Summary of Commercial Obligations

| Obligation | Source | Action Required |
|-----------|--------|----------------|
| Include MIT license text | V-JEPA 2 | Copy license file into distribution |
| Include "Built with Llama" | Llama 3.2 | Add attribution to product/docs |
| Include Llama license | Llama 3.2 | Distribute license text |
| Include Llama AUP | Llama 3.2 | Distribute AUP text |
| Include Gemma Terms of Use | EmbeddingGemma | Distribute terms text |
| Include Gemma Prohibited Use Policy | EmbeddingGemma | Distribute policy text |
| Comply with Meta AUP | Llama 3.2 | Do not build prohibited applications |
| Comply with Gemma Prohibited Use Policy | EmbeddingGemma | Do not build prohibited applications |

---

## 4. What You CANNOT Do Commercially

### 4.1 Hard Restrictions

- **Use I-JEPA or V-JEPA v1 weights in commercial products**: CC BY-NC 4.0 prohibits this entirely.

- **Exceed 700M MAU without Meta's permission**: If your product reaches this scale, you need a separate license from Meta for the Llama component.

- **Use Llama outputs to train competing LLMs**: The Llama license prohibits using Llama outputs to improve other large language models. (This does not apply to VL-JEPA's use case, but be aware if you use Llama for other purposes in the lab.)

- **Claim Google/Meta endorsement**: You cannot use their trademarks to suggest endorsement.

- **Build prohibited applications**: Weapons, surveillance, disinformation, etc. as defined in both Meta's AUP and Google's Prohibited Use Policy.

### 4.2 Gray Areas

- **Derivative model weights**: If you train a new model using VL-JEPA's pipeline and weights, the resulting model inherits license obligations from all components used in training. If you used Llama weights, the Llama license applies to derivatives. If you used EmbeddingGemma, the Gemma Terms apply.

- **Distillation**: If you distill VL-JEPA into a smaller model, the resulting model likely inherits license obligations. Consult legal counsel for specifics.

- **Research to commercial pipeline**: If you develop techniques using I-JEPA / V-JEPA v1 (non-commercial) and then re-implement them using V-JEPA 2 (MIT), the re-implementation should be clean. Document the separation.

---

## 5. How to Build a Commercially Viable Stack

### 5.1 Recommended Architecture for Commercial Use

```
Visual Encoder:   V-JEPA 2 ViT-L or ViT-g  (MIT License -- fully permissive)
Predictor:        Llama-3.2-1B (last 8 layers)  (Llama 3.2 License -- commercial OK)
Text Encoder:     EmbeddingGemma-300M  (Gemma Terms -- commercial OK)
```

This is the VL-JEPA stack, and all components are commercially usable.

### 5.2 If You Want to Minimize License Obligations

Replace Llama and/or EmbeddingGemma with MIT/Apache-licensed alternatives:

| Component | Commercial Alternative | License | Trade-off |
|-----------|----------------------|---------|-----------|
| Predictor (Llama-3.2-1B) | Any MIT/Apache transformer | MIT/Apache | Must retrain; may lose quality |
| Y-Encoder (EmbeddingGemma) | Sentence-BERT, E5, GTE | Apache 2.0 | Must retrain; different embedding space |
| Visual Encoder (V-JEPA 2) | Keep as-is | MIT | No change needed |

**Minimal-obligation stack:**
```
Visual Encoder:   V-JEPA 2 (MIT)
Predictor:        Custom transformer trained from scratch (your own license)
Text Encoder:     E5-base or GTE-base (Apache 2.0)
```

This eliminates the Llama and Gemma license obligations entirely, but requires retraining the predictor and Y-Encoder from scratch, with potentially different quality.

### 5.3 Practical Recommendation

For most labs and startups, the **standard VL-JEPA stack is fine for commercial use**:
- The Llama 700M MAU threshold is not a concern for 99.99% of organizations.
- The attribution requirements ("Built with Llama") are trivial.
- The AUPs from both Meta and Google are standard ethical restrictions you should follow anyway.
- Only consider the minimal-obligation stack if you have specific legal requirements (e.g., government contracts that restrict third-party licenses).

---

## 6. Recommendations for the Lab's Own Licensing Strategy

### 6.1 For Research Outputs (Papers, Experiments)

- No licensing concerns. All components (including CC BY-NC 4.0 ones) are usable for research.
- Publish freely. Cite the original papers.

### 6.2 For Code You Write

Recommendation: **Apache 2.0** for code that does not contain or derive from model weights.

Why Apache 2.0:
- Permissive: allows commercial use by others.
- Patent grant: protects you and your users.
- Compatible with MIT, CC BY-NC 4.0, Llama, and Gemma licenses.
- Industry standard for ML research code (used by Hugging Face, TensorFlow, etc.).

### 6.3 For Model Weights You Train

This is where it gets nuanced. Your trained weights inherit obligations from the training components:

**If trained with V-JEPA 2 + Llama + EmbeddingGemma (the full VL-JEPA stack):**
- Your weights inherit Llama 3.2 Community License obligations.
- Your weights inherit Gemma Terms of Use obligations.
- You MUST distribute both license texts with your weights.
- You MUST include "Built with Llama" attribution.
- Recommendation: Release under a custom license that includes the downstream obligations from Llama and Gemma, plus your own terms.

**If trained with V-JEPA 2 only (no Llama, no Gemma):**
- Your weights inherit only MIT obligations (minimal).
- You can release under any license you choose.
- Recommendation: Apache 2.0 for maximum adoption, or a proprietary license if you want to commercialize.

### 6.4 For Demos and Prototypes

- Use whatever components are convenient (including CC BY-NC 4.0 ones).
- Just do not deploy these commercially.
- Clearly mark demos that use non-commercial components.

### 6.5 License Compatibility Matrix

| Your Use | V-JEPA 2 (MIT) | Llama 3.2 | EmbeddingGemma | I-JEPA / V-JEPA v1 (CC BY-NC) |
|----------|----------------|-----------|----------------|-------------------------------|
| Academic research | OK | OK | OK | OK |
| Internal prototyping | OK | OK | OK | OK |
| Commercial product | OK | OK (< 700M MAU) | OK | **NOT OK** |
| Open-source release | OK (include MIT) | OK (include Llama license + AUP) | OK (include Gemma terms) | OK (include CC BY-NC text) |
| SaaS deployment | OK | OK (< 700M MAU) | OK | **NOT OK** |
| Government contract | OK | Check with counsel | Check with counsel | **NOT OK** |

---

## 7. Links to License Texts

| Component | License | Full Text URL |
|-----------|---------|---------------|
| I-JEPA | CC BY-NC 4.0 | https://creativecommons.org/licenses/by-nc/4.0/legalcode |
| V-JEPA (v1) | CC BY-NC 4.0 | https://creativecommons.org/licenses/by-nc/4.0/legalcode |
| V-JEPA 2 | MIT License | https://opensource.org/licenses/MIT |
| V-JEPA 2 (some files) | Apache 2.0 | https://www.apache.org/licenses/LICENSE-2.0 |
| Llama-3.2-1B | Llama 3.2 Community License | https://www.llama.com/llama3_2/license/ |
| Llama AUP | Acceptable Use Policy | https://www.llama.com/llama3_2/use-policy/ |
| EmbeddingGemma-300M | Gemma Terms of Use | https://ai.google.dev/gemma/terms |
| EmbeddingGemma AUP | Gemma Prohibited Use Policy | https://ai.google.dev/gemma/prohibited_use_policy |

---

## 8. Disclaimer

This analysis is provided for informational purposes and represents a good-faith interpretation of publicly available license texts as of the date above. It is NOT legal advice. For binding legal decisions -- especially regarding commercial deployment, government contracts, or redistribution of model weights -- consult a qualified intellectual property attorney.

License terms may change. Always verify against the current version of each license before making deployment decisions.

---

*Last updated: 2026-03-07*
