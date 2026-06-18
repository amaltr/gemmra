# ADR-003: Training Strategy

**Status:** ✅ FINAL — SFT shipped (0.862 composite). DAPO was implemented but failed due to reward variance collapse. See `docs/evaluation/analysis/grpo_first_principles_analysis.md`. SFT hyperparameters (v4: bf16, r=64, 8192 seq) remain accurate.
**Date:** June 10, 2026 (Stage 2 failure confirmed June 14)
**Decision Makers:** Team
**Supersedes:** v3 (4-bit QLoRA — based on false 28GB total storage assumption)

---

## Context

We need to fine-tune a base model for 4 pharmacovigilance tasks within 20 total GPU hours (4 hrs/day × 5 days on AMD MI300X, 192 GB VRAM, 3.1 TB ephemeral storage).

## Decision: SFT → DAPO (Planned) → SFT-Only (Shipped)

> **Outcome:** DAPO (Stage 2) was implemented and tested but failed due to reward variance collapse. SFT alone shipped with 0.862 composite. The DAPO config and code remain for reference.

### v4 Key Changes (from v3)
| Change | v3 | v4 (current) | Reason |
|--------|-----|-------------|--------|
| Precision | 4-bit QLoRA | **bf16 LoRA (no quantization)** | 192 GB VRAM; no need to quantize; better gradients |
| LoRA rank | r=32 | **r=64** | Deeper domain adaptation; trivial VRAM cost |
| Sequence length | 4096 | **8192** | Longer case narratives; VRAM headroom allows it |
| Batch size | 4 (eff. 16) | **8 (eff. 32)** | Better gradients; faster training |
| Training data | 21K pairs | **~31K pairs (from 29 quarters)** | 7 years of FAERS data (2019-2026); sufficient for all 4 tasks |
| bitsandbytes | Required for 4-bit | **Not needed for model loading** | Removes failure point #1 |

### v3 Key Changes (from v1/v2)
| Change | v1/v2 | v3 (current) | Source |
|--------|-------|-------------|--------|
| RL algorithm | Vanilla GRPO | **DAPO** (loss_type="dapo") | TRL v1.0, June 2026 |
| Token format | `<think>/<answer>` | **Gemma 4 native** (`<\|channel>thought`) | Gemma 4 docs |
| LoRA rank | r=16 | **r=32** | 2026 research consensus |
| Rewards | 4 signals, equal weight | **4 signals, weighted + normalize_then_sum** | TRL v1.0, MediX-R1 |
| Data quality | Basic filtering | **Decontamination + gold-label resampling** | MeditronFO (EPFL) |
| Adversarial data | None | **12% adversarial negatives** | 2025-2026 consensus |
| Curriculum | Random order | **Easy→hard ordering** | 2026 best practice |

```
Stage 1: SFT (Supervised Fine-Tuning) via bf16 LoRA
├── Purpose: Teach the model FORMAT and DOMAIN KNOWLEDGE
├── Data: ~31K instruction pairs across 4 tasks (Gemma 4 native chat format)
├── LoRA: r=64, α=128, all linear layers, Unsloth gradient checkpointing
├── Precision: bf16 (NO quantization — 192 GB VRAM)
├── Duration: ~3.5 GPU hours
├── Quality: MeditronFO-adopted decontamination + gold-label resampling
├── Curriculum: Easy→hard ordering by task difficulty
├── Adversarial: 12% negative examples to prevent fragile reasoning
├── Output: Model that uses Gemma 4 thinking mode with structured PV output
└── Metric: Training loss < 0.5

    ↓ (checkpoint)

Stage 2: DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
├── Purpose: Teach the model REASONING QUALITY + FAITHFULNESS
├── Data: Subset of 2K best prompts
├── Mode: DAPO (clip-higher=0.28 + token-level loss normalization)
├── Rewards: 4 weighted signals (format, structure, reasoning, faithfulness)
├── Aggregation: normalize_then_sum (prevents reward domination)
├── Duration: ~2 GPU hours
├── Output: Model with clinically faithful, evidence-grounded reasoning
└── Metric: Reward score improving; format compliance >95%
```

### Why DAPO Instead of Vanilla GRPO?

| Problem in Vanilla GRPO | How DAPO Fixes It |
|--------------------------|-------------------|
| Bias against long reasoning outputs | Token-level loss normalization |
| Entropy collapse (converge to single pattern) | Clip-higher (separate upper/lower clipping) |
| Wasted compute on trivial batches | Dynamic sampling (skip all-correct/all-wrong groups) |

DAPO is available via `loss_type="dapo"` in TRL v1.0 GRPOConfig — zero extra code.

### Why Two-Stage?

| Aspect | SFT Only (SHIPPED) | SFT + DAPO (PLANNED, FAILED) |
|--------|----------|-----------|
| Format compliance | ~85% | ~99% |
| Reasoning quality | Memorized patterns | Clinically faithful reasoning |
| Evidence grounding | Low | High (faithfulness reward) |
| Metric improvement | 40-60% over baseline | 60-80% over baseline |
| "Wow" factor | Standard | State-of-the-art |
| Hackathon differentiation | Low (many teams do SFT) | High (DAPO + composite rewards) |

> **Actual result:** SFT achieved **0.862 composite** with **100% format compliance**, exceeding the "SFT Only" estimates above. DAPO failed due to reward variance collapse — the SFT model was already too consistent.

## Hyperparameters

### SFT Configuration (v4)
```yaml
# See configs/sft_config.yaml
load_in_4bit: false            # bf16 LoRA — no quantization on 192 GB VRAM
lora_r: 64                     # Increased from 32 — deeper domain adaptation
lora_alpha: 128                # 2× rank
learning_rate: 2e-4
epochs: 2
batch_size: 8                  # Increased from 4 — better gradients
gradient_accumulation: 4       # Effective batch: 32
max_seq_length: 8192           # Increased from 4096 — longer case narratives
target_modules: all_linear     # q, k, v, o, gate, up, down
curriculum_learning: true      # Easy→hard ordering
adversarial_ratio: 0.12        # 12% negative examples
```

### DAPO Configuration (v4)
```yaml
# See configs/grpo_config.yaml
loss_type: "dapo"              # DAPO mode
epsilon_high: 0.28             # Clip-higher
learning_rate: 5e-6
num_generations: 4
max_new_tokens: 512
epochs: 1
batch_size: 2
multi_objective_aggregation: "normalize_then_sum"
mask_truncated_completions: true  # DAPO best practice: don't penalize incomplete sequences
```

## Reward Function Design (DAPO — 4-Signal Composite)

Research-informed composite reward (Clinical-R1/CRPO, MediX-R1):

| Component | Weight | Check | Source |
|-----------|--------|-------|--------|
| **Format compliance** | 1.0 | Gemma 4 thinking tokens OR structured output | Gemma 4 docs |
| **Task structure** | 1.0 | Task-specific fields present? (SERIOUS:, MedDRA PT:, etc.) | Domain expertise |
| **Reasoning quality** | 0.8 | Domain terminology + appropriate length | MediX-R1 (MBZUAI 2026) |
| **Faithfulness** | 1.2 | References specific case data, temporal info, criteria | Clinical-R1/CRPO (AAAI 2026) |

**Aggregation:** `normalize_then_sum` — each reward is normalized within its
group before summing, preventing any single signal from dominating training.

## MeditronFO-Adopted Techniques

| Technique | What | Where |
|-----------|------|-------|
| Gold-label resampling | Generate 4-8 SFT outputs, keep best match to ground truth | `03_build_training_data.py` |
| Decontamination | 10% holdout, hash-verified, logged | `03_build_training_data.py` |
| Guideline grounding | ICH E2A/E2B/WHO-UMC criteria in system prompts | System prompts |
| PSEBench clause-card reasoning | Decompose regulatory criteria into atomic decisions | T1 system prompt |

## Consequences

- DAPO (Stage 2) was fully implemented but failed due to reward variance collapse
- SFT alone achieved 0.862 composite — exceeding original SFT-only estimates
- SFT checkpoint is the final shipped model
- GRPO/DAPO code and config remain as reference for future work

## Risk Mitigation

- SFT checkpoint saved before DAPO → SFT-only was the valid fallback (and became the final model)
- DAPO was monitored and aborted when reward variance collapsed
- Low DAPO learning rate (5e-6) prevented catastrophic forgetting during testing
