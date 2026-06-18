# Fine-Tuning Techniques Research

> **Note:** This document captures the fine-tuning landscape research. Our final pipeline used **bf16 LoRA SFT** (r=64, α=128) + **WiSE-FT** reasoning recovery. GRPO/DAPO were implemented and validated the SFT ceiling. Final composite: 0.862.


---

## Techniques Landscape (2026)

### Overview

```
                    ┌─────────────────────────────┐
                    │     Post-Training Pipeline   │
                    │                               │
                    │  ┌─────────┐   ┌──────────┐  │
                    │  │  SFT    │   │  DAPO     │  │  ← DAPO explored, FAILED
                    │  │(bf16)   │   │  (RL)     │  │    SFT shipped alone
                    │  └─────────┘   └──────────┘  │
                    │       │              │        │
                    │  Format &      Reasoning &   │
                    │  Knowledge     Optimization   │
                    └─────────────────────────────┘
```

### Technique Deep Dive

#### 1. SFT (Supervised Fine-Tuning) ← WE USE THIS
**What:** Show the model thousands of correct input→output pairs  
**Analogy:** Teaching by example (textbook learning)  
**Best for:** Teaching format, domain vocabulary, task structure  
**Limitation:** Model may memorize patterns without understanding  

#### 2. LoRA (Low-Rank Adaptation) ← WE USE THIS (bf16, not quantized)
**What:** Train small adapter layers while keeping base model frozen
**Analogy:** Teaching a specialist through focused study sessions instead of re-reading every book
**Best for:** Parameter-efficient training (99% fewer trained params)
**Our config:** rank r=64, alpha α=128, bf16 precision (no quantization — 192 GB VRAM allows full precision)  

#### 3. GRPO (Group Relative Policy Optimization) ← SUPERSEDED BY DAPO
**What:** Generate multiple outputs per prompt, reward the best ones  
**Analogy:** A medical student reviews multiple draft diagnoses and learns which reasoning paths lead to correct conclusions  
**Status:** We use DAPO (see below), which is GRPO with critical fixes.

#### 3b. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) ← WE USE THIS
**What:** GRPO successor (TRL v1.0, June 2026) with 3 critical fixes:  
**Fixes:**
1. **Token-level loss normalization** — prevents bias against long thinking outputs
2. **Clip-higher** — separate upper/lower clipping prevents entropy collapse
3. **Dynamic sampling** — skips batches where all outputs are identical (saves ~30% compute)

**How it works:**
1. Give model a prompt
2. Generate N outputs (N=4)
3. Score each output with 4 weighted reward functions
4. Normalize rewards per-signal before summing (normalize_then_sum)
5. Compute DAPO advantages with clip-higher
6. Reinforce outputs with higher-than-average rewards

**Configuration:** `loss_type="dapo"` in GRPOConfig (TRL v1.0)

#### 4. DPO (Direct Preference Optimization) ❌ NOT USED
**What:** Learn from pairs of "good" vs "bad" outputs  
**Why we skip it:** Creating preference pairs requires either human annotators (no time) or a stronger model as judge (adds complexity)  

#### 5. ORPO (Odds Ratio Preference Optimization) ❌ NOT USED
**What:** Combined SFT + alignment in one step  
**Why we skip it:** GRPO is better for reasoning-heavy tasks; ORPO is better for general alignment  

#### 6. SimPO (Simple Preference Optimization) ❌ NOT USED
**What:** Length-normalized preference optimization  
**Why we skip it:** Addresses length bias in DPO, but we don't have preference pairs  
**Note:** Good alternative if we had preference data  

#### 7. Evolution Strategies (ES) 🆕 EMERGING (2026)
**What:** Optimize directly in parameter space using evolutionary algorithms  
**Why we skip it:** Promising for avoiding reward hacking, but not production-ready  
**Note:** Worth watching for future iterations  

#### 8. Full Fine-Tuning ❌ NOT USED
**What:** Update all model parameters  
**Why we skip it:** 31B needs ~120GB+ VRAM for full FT. bf16 LoRA achieves 95%+ quality with <10% of trainable params.

### Accuracy Comparison (Medical Domain Research)

| Approach | Typical Improvement Over Base | Training Time | Memory |
|----------|------------------------------|---------------|--------|
| SFT (bf16 LoRA) | +15-25% F1 | 2-10 hours | 9-15 GB |
| SFT + DPO | +20-30% F1 | 4-6 hours | 15-20 GB |
| SFT + GRPO | +25-40% F1 | 4-6 hours | 12-18 GB |
| Full FT | +30-45% F1 | 8-16 hours | 56+ GB |

*Source: Multiple 2025-2026 research papers on medical LLM fine-tuning*

### Our Pipeline Justification

```
SFT (bf16 LoRA) → GRPO/DAPO (planned)
     ↓                 ↓
"Learn WHAT        "Learn HOW
 to say"            to reason"
     ↓                 ↓
 Format +         FAILED: reward
 Domain vocab     variance collapse
```

**Actual outcome:** SFT alone achieved 0.862 composite. GRPO/DAPO failed because the SFT model was already too consistent (all 8 generations produced identical answers → zero reward variance → no gradient).

### Hyperparameter Research (2025-2026 Best Practices)

| Parameter | Recommended | Source |
|-----------|-------------|--------|
| LoRA rank (r) | 64 (our choice), 16-32 (start) | Industry consensus; we went higher for deep domain |
| LoRA alpha | 2× rank (128 for r=64) | Sebastian Raschka's research |
| Learning rate (SFT) | 2e-4 | TowardsAI, multiple papers |
| Learning rate (GRPO) | 5e-6 to 1e-5 | GRPO paper, AMD guide |
| Target modules | ALL linear layers | 2026 best practice (not just attention) |
| Batch size (effective) | 16-32 | With gradient accumulation |
| Epochs (SFT) | 2-3 | Beyond 3 = overfitting risk |
| Epochs (GRPO) | 1 | Standard for RL fine-tuning |
| GRPO group size | 4-8 | Unsloth recommendation |
| Warmup ratio | 0.05 | Standard |
| Scheduler | Cosine | Standard |
| Data quality | 500-2K excellent > 10K mediocre | 2026 consensus |

### Iteration Plan

| Experiment | When | What to Measure |
|------------|------|-----------------|
| Baseline (zero-shot) | Day 2 | Raw Gemma 4 12B performance on all tasks |
| SFT only | Day 3 | Training loss, format compliance |
| SFT + GRPO | Day 3-4 | Reward improvement, reasoning quality |
| Compare models (if time) | Day 4 | Gemma 4 12B vs MedGemma 27B on same data |
