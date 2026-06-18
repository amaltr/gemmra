# Research Pipeline — Decision Points for Solution Development

> **Purpose:** This document identifies every decision point in the solution
> development pipeline. For each decision, it defines what was researched,
> what was decided, and why. Includes MeditronFO-adopted techniques.
>
> **Final State:** All decision points resolved. SFT+WiSE-FT shipped as final model (0.862 composite).

---

## Pipeline Overview

The complete solution has **10 decision points** (expanded from 9 after
MeditronFO analysis added DP-10: Data Quality Assurance):

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION PIPELINE                             │
│                                                                 │
│  DP-1: Base Model Selection          ✅ Gemma 4 31B             │
│    ↓                                                            │
│  DP-2: Fine-Tuning Technique         ✅ SFT + WiSE-FT shipped   │
│    ↓                                                            │
│  DP-3: Training Data — Sources       ✅ FAERS+OnSIDES+BioDEX    │
│    ↓                                                            │
│  DP-4: Training Data — Format        ✅ Gemma 4 thinking mode   │
│    ↓                                                            │
│  DP-5: Hyperparameter Configuration  ✅ r=64, α=128, bf16       │
│    ↓                                                            │
│  DP-6: Reward Function Design        ✅ Designed (GRPO failed)  │
│    ↓                                                            │
│  DP-10: Data Quality Assurance       ✅ MeditronFO-adopted      │
│    ↓                                                            │
│  DP-7: Inference & Serving           ✅ Unsloth + Ollama/GGUF   │
│    ↓                                                            │
│  DP-8: Evaluation Methodology        ✅ Gemmra-Bench v1.0      │
│    ↓                                                            │
│  DP-9: Demo & Deployment             ✅ Website + HuggingFace   │
└─────────────────────────────────────────────────────────────────┘
```

---

## DP-1: Base Model Selection

**ADR:** `docs/architecture/decisions/ADR-002-base-model.md`  
**Status:** ✅ DECIDED v3 — Gemma 4 31B (June 9, devil's advocate analysis)

### Decision History

| Version | Model | Error in Decision |
|---------|-------|-------------------|
| v1 (June 9 AM) | Gemma 4 12B | Underweighted thinking mode |
| v2 (June 9 mid) | Qwen3-32B | FALSE claim Gemma 4 lacks thinking mode |
| **v3 (June 9 PM)** | **Gemma 4 31B** | **Corrected — Gemma 4 HAS thinking mode** |

### Final Decision

| Role | Model | Rationale |
|------|-------|-----------|
| **PRIMARY** | `google/gemma-4-31b-it` | MMLU-Pro 85.2%, thinking mode, April 2026, MeditronFO-validated architecture |
| Fallback | `google/gemma-4-12b-it` | Same family, same tokens, ~7GB, 2x faster |
| Showcase | `meta-llama/Llama-3.3-70B-Instruct` | AMD MI300X exclusive (140GB VRAM) |

### Why Not Apertus-70B-MeditronFO?
Paper's best model has 2 hard blockers: no thinking mode (Sep 2025),
and weaker benchmarks than Gemma 4 31B. Storage is NOT a blocker (3.1 TB working storage).
See `docs/research/meditron_fo_reference.md` for full analysis.

---

## DP-2: Fine-Tuning Technique

**ADR:** `docs/architecture/decisions/ADR-003-training-strategy.md`  
**Status:** ✅ FINAL — SFT + WiSE-FT shipped. GRPO/DAPO/RAFT all validated SFT ceiling.

### Decision

**SFT + WiSE-FT** (final shipped pipeline):
1. Stage 1: Supervised Fine-Tuning on PV-specific data (r=64, α=128, bf16 LoRA) ✅ **SHIPPED**
2. Stage 2: WiSE-FT reasoning recovery (α=0.9: 90% SFT + 10% base weights) ✅ **SHIPPED**
3. Stage 3: GRPO reinforcement learning ❌ **FAILED** (+0.003 then reward collapse)
4. Stage 4: RAFT rejection sampling ❌ **VALIDATED CEILING** (no improvement)

**Why DAPO failed (discovered June 14):**
- SFT model was already too consistent (0.86+ composite)
- All 8 GRPO generations produced identical answers → zero reward variance → no gradient
- See `docs/evaluation/analysis/grpo_first_principles_analysis.md` for full post-mortem

MeditronFO paper used SFT-only (no RL). Their approach was ultimately correct for this task type.
SFT checkpoint shipped as final model with **0.862 composite** score.

---

## DP-3: Training Data — Sources & Volume

**ADR:** `docs/architecture/decisions/ADR-004-data-pipeline.md`  
**Status:** ✅ EXPANDED — PV-specific data (not general medical QA)

### Decision

| Dataset | Type | Size | Purpose |
|---------|------|------|---------|
| FAERS 2019Q1–2026Q1 | Real PV cases (29 quarters) | ~32K pairs | Primary training data (all 4 tasks) |
| OnSIDES v3.1.1 | Drug label ADE extraction | 831K drug-AE pairs | T3 ground truth (labelling status) |

### Why NOT Add General Medical QA (MedQA, MedMCQA)?

MeditronFO's pipeline includes 8 general medical QA datasets. We skip these:
- **Dilution risk:** General medical QA would dilute our PV-specific signal
- **Gemma 4 31B already knows medicine:** MMLU-Pro 85.2% shows strong base
- **Our FAERS data is more targeted:** Directly maps to our 4 tasks
- **Time constraint:** Adding 400K+ general QA examples doubles training time

---

## DP-4: Training Data — Format & Template

**Status:** ✅ DECIDED — Gemma 4 native thinking mode format

### Decision

Use Gemma 4's native chat template with thinking mode ON for ALL tasks:

**All 4 tasks (thinking=ON — ADR-006):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a pharmacovigilance expert. Think step by step."},
    {"role": "user", "content": "[case data]"},
    {"role": "assistant", "content": "<|channel>thought\n[step-by-step reasoning]\n<channel|>\nSERIOUS: YES\nCriteria: HO\nRationale: ..."}
  ]
}
```

### Why This Format?
- Native to Gemma 4 → no custom token injection needed
- Thinking tokens preserved during Unsloth QLoRA training
- Same format works for both 31B (primary) and 12B (fallback)

---

## DP-5: Hyperparameter Configuration

**Status:** ✅ DECIDED — validated for Gemma 4 31B on MI300X

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | **64** | Deep domain adaptation for PV reasoning (192 GB VRAM makes this trivial) |
| LoRA alpha | **128** (2×r) | Standard ratio |
| Learning rate (SFT) | 2e-4 | Proven for LoRA fine-tuning |
| Learning rate (DAPO) | 5e-6 | Conservative for RL stability |
| Batch size (effective) | 32 | MI300X with 192 GB VRAM supports BS=8 × GA=4 |
| Epochs (SFT) | 2 | Sufficient for ~32K examples |
| Target modules | All linear | Dense model → all layers contribute |
| Max seq length | 8192 | Increased from 4096 — thinking traces need headroom |
| Load in 4-bit | false | bf16 LoRA — no quantization needed on 192 GB VRAM |
| DAPO loss_type | "dapo" | Clip-higher + token-level loss (Discovery 27) |
| DAPO epsilon_high | 0.28 | Prevents entropy collapse |
| Adversarial ratio | 12% | Prevents fragile reasoning (Discovery 23) |
| Curriculum learning | true | Easy→hard ordering (Discovery 24) |

---

## DP-6: Reward Function Design (DAPO)

**Status:** ✅ DECIDED v2 — 4-signal composite design (Clinical-R1/CRPO + MediX-R1 inspired)

| Component | What It Checks | Weight | Source |
|-----------|---------------|--------|--------|
| **Format reward** | Gemma 4 thinking tokens + structured output | 1.0 | Gemma 4 docs |
| **Structure reward** | Task-specific fields (SERIOUS:, MedDRA PT:, etc.) | 1.0 | Domain expertise |
| **Reasoning quality** | Domain terminology + appropriate length | 0.8 | MediX-R1 (MBZUAI 2026) |
| **Faithfulness** | References specific case data, temporal info, criteria | 1.2 | Clinical-R1/CRPO (AAAI 2026) |

**Aggregation:** `normalize_then_sum` (TRL v1.0) — each reward is normalized
within its group before summing, preventing any single signal from dominating.

**Key insight:** The faithfulness reward (weight 1.2, highest) explicitly trains
the model to ground reasoning in SPECIFIC case data rather than generic medical
knowledge. This is the key differentiator from standard SFT.

---

## DP-10: Data Quality Assurance (NEW — MeditronFO-Adopted)

**Status:** ✅ DECIDED — 3 techniques adopted from MeditronFO paper

### 10a. Gold-Label Resampling
**Source:** MeditronFO paper (arXiv:2605.16215)

**What:** Generate N candidate responses per training example, keep only
the one that best matches the ground truth label.

**Our implementation:**
1. For each FAERS case, use base Gemma 4 31B to generate 4 candidate responses
2. Compare each response against FAERS ground truth labels
3. Select the response with highest match score
4. Use selected response as the SFT training target

**Why:** Eliminates low-quality training examples. The model learns from
responses that are both correct AND naturally formatted.

### 10b. System-Wide Decontamination
**Source:** MeditronFO paper

**What:** Ensure zero overlap between training data and evaluation data.

**Our implementation:**
1. Hold out 10% of FAERS cases as evaluation set (stratified by seriousness)
2. Hash all evaluation case IDs
3. Before training, verify no evaluation case appears in training JSONL
4. Log decontamination results for auditability

### 10c. Guideline-Grounding
**Source:** MeditronFO paper (adapted for PV)

**What:** Include authoritative clinical guidelines in training context.

**Our implementation:**
- ICH E2A criteria in system prompt for Task 1 (seriousness)
- ICH E2B R3 fields in system prompt for Task 3 (labelling)
- WHO-UMC causality scale in system prompt for Task 4 (causality)
- MedDRA hierarchy description in system prompt for Task 2 (coding)

---

## DP-7: Inference & Serving Strategy

**ADR:** `docs/architecture/decisions/ADR-005-inference-demo.md`  
**Status:** ✅ DECIDED — vLLM on ROCm (SGLang alternative)

---

## DP-8: Evaluation Methodology

**Status:** ✅ DECIDED (enhanced with LLM-as-judge from MeditronFO)

### Multi-Layer Evaluation

| Layer | What | Metrics |
|-------|------|---------|
| **Quantitative** | Holdout FAERS test set | F1, precision, recall, accuracy per task |
| **Format** | Output structure validation | JSON validity, required fields present |
| **LLM-as-judge** 🆕 | Multi-dimensional quality scoring | Clinical accuracy, reasoning quality, evidence |
| **Before/after** | Base vs SFT vs GRPO | Show progressive improvement |
| **Thinking trace** | Visible reasoning analysis | Demo-ready thinking chain |

### LLM-as-Judge Protocol (from MeditronFO)
Use Gemini or GPT-4o to score model outputs on 5 dimensions:
1. Clinical accuracy (does the answer match expert consensus?)
2. Reasoning quality (is the thinking trace logical?)
3. Format compliance (does it follow the required structure?)
4. Evidence citation (does it reference specific case data?)
5. Thinking trace coherence (does the reasoning lead to the conclusion?)

---

## DP-9: Demo & Deployment

**Status:** ✅ COMPLETE

### Final Deliverables

- **Website:** [gemmra.bhaskarjha.dev](https://gemmra.bhaskarjha.dev) — Astro-based project showcase
- **HuggingFace:** LoRA adapters, merged bf16, GGUF (Q4_K_M, Q8_0, Q5_K_M)
- **Local inference:** Ollama-compatible GGUF model
- **Demo:** Interactive terminal console + Gradio web app

---

## Research Execution Summary

| # | Decision Point | Status | Final Choice |
|---|---------------|--------|-------------|
| ✅ | DP-1: Base Model | FINAL | Gemma 4 31B (thinking mode + 85.2% MMLU-Pro) |
| ✅ | DP-2: Fine-Tuning | FINAL | SFT + WiSE-FT shipped. GRPO/RAFT validated ceiling. |
| ✅ | DP-3: Data Sources | FINAL | FAERS 2019Q1–2026Q1 + OnSIDES + BioDEX (~32K pairs) |
| ✅ | DP-4: Data Format | FINAL | Gemma 4 native thinking tokens (ALL tasks) |
| ✅ | DP-5: Hyperparameters | FINAL | r=64, α=128, bf16 LoRA, completion-only loss |
| ✅ | DP-6: Reward Functions | HISTORICAL | 4-signal composite designed. GRPO failed → proves SFT ceiling. |
| ✅ | DP-7: Inference | FINAL | Unsloth native + Ollama GGUF + vLLM |
| ✅ | DP-8: Evaluation | FINAL | Gemmra-Bench v1.0 (3,645 decontaminated samples) |
| ✅ | DP-9: Demo | FINAL | Website + terminal demo + Gradio |
| ✅ | DP-10: Data Quality | FINAL | Gold-label resample + decontamination + guideline-grounding |

---

*All decision points resolved. Final model: SFT + WiSE-FT, 0.862 composite on Gemmra-Bench v1.0.*
