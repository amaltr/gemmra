# FullyOpenMeditron — Reference & Adopted Techniques

**Source:** arXiv:2605.16215 (EPFL LiGHT Lab, May 2026)  
**GitHub:** https://github.com/EPFLiGHT/FullyOpenMeditron  
**HuggingFace:** EPFLiGHT/meditronfo collection  
**Relevance:** Validates medical SFT on Gemma architecture + provides techniques we adopt

---

## Key Results from Paper

| Model | Score | vs Base | Notes |
|-------|-------|---------|-------|
| Apertus-70B-MeditronFO | 53.8% aggregate | +6.6 | Best FO SoTA |
| Gemma-3-27B-MeditronFO | 58.0% HealthBench | Beat MedGemma (55.9%) | Non-FO comparison |
| Gemma-3-27B-MeditronFO | 58.6% LLM-judge pref | Over MedGemma | Key validation |

## Why We Can't Use Apertus-70B-MeditronFO Directly

| Blocker | Detail |
|---------|--------|
| **Storage** | BF16 ~140GB (but we have 3.1 TB working storage, so not a blocker) |
| **No thinking mode** | Apertus (Sep 2025) predates thinking-mode models |
| **Benchmarks** | Gemma 4 31B (MMLU-Pro 85.2%) >> Apertus 70B |
| **Training cost** | Paper used 213 GPU-hours on 32 GPUs; we have 20 GPU-hours on 1 GPU |

## Why Gemma 4 31B Is Our Base Instead

The paper proves the PIPELINE works. The pipeline can be applied to ANY base model.
Gemma 4 31B is architecturally superior to everything the paper tested:

| Property | Paper's Best (Apertus 70B) | Our Choice (Gemma 4 31B) |
|----------|--------------------------|-------------------------|
| MMLU-Pro | Not competitive | 85.2% |
| Thinking mode | ❌ | ✅ enable_thinking=True |
| Architecture year | Sep 2025 | April 2026 |
| Fits storage? | ✅ (3.1 TB) | ✅ (3.1 TB) |

---

## Techniques ADOPTED from Paper

### 1. Gold-Label Resampling
**What:** Generate N responses per training example using teacher model,
keep only the one that matches the ground truth label.

**Paper's approach:** Used GPT-OSS-120B, sampled up to 8 times.

**Our adaptation:**
- Use the base Gemma 4 31B (before fine-tuning) as teacher
- For each FAERS case, generate 4-8 candidate responses
- Compare against ground truth (FAERS labels are our gold standard)
- Keep the response closest to ground truth for SFT training data
- Reject hallucinated or malformatted responses

**Why this matters:** Eliminates garbage training examples. The model
only learns from high-quality, verified responses.

### 2. System-Wide Decontamination
**What:** Ensure zero overlap between training data and evaluation data.

**Our adaptation:**
- Hold out 10% of FAERS cases as evaluation set
- Hash all test case IDs
- Before training, verify no test case ID appears in training JSONL
- Log decontamination results for auditability

### 3. Guideline-Grounding
**What:** Include clinical practice guidelines in training to anchor reasoning.

**Our adaptation:**
- ICH E2A (clinical safety reporting) as system prompt context
- ICH E2B (individual case safety report format) for structured output
- WHO-UMC causality criteria as reference for Task 4
- MedDRA hierarchy documentation for Task 2

### 4. LLM-as-Judge Evaluation
**What:** Multi-dimensional quality scoring beyond simple metrics.

**Our adaptation:**
- Standard metrics (F1, accuracy, format compliance) as baseline
- LLM-as-judge scoring on 5 dimensions:
  1. Clinical accuracy
  2. Reasoning quality
  3. Format compliance
  4. Evidence citation
  5. Thinking trace coherence
- Before/after comparison (base → SFT)

### 5. PSEBench Clause-Card Reasoning (NEW — June 2026)
**What:** Decompose regulatory policy into atomic sub-decisions ("clause cards").

**Source:** PSEBench benchmark (June 2026)

**Our adaptation:**
- T1 system prompt decomposes ICH E2A into 6 atomic criteria (C1-C6)
- Model evaluates EACH criterion against case data separately
- Model states "INSUFFICIENT DATA" when evidence is ambiguous
- Supports "principled abstention" (refuse to guess when unsure)

---

## Techniques SKIPPED from Paper (with rationale)

| Technique | Why Skipped |
|-----------|-------------|
| General medical QA datasets (MedQA, MedMCQA) | Dilutes PV signal; our FAERS data is more targeted |
| 4-physician validation panel | No physician access in 5-day hackathon |
| Full fine-tuning (not LoRA) | Requires 32 GPUs; bf16 LoRA achieves 95%+ quality |
| Using Apertus 70B base | No thinking mode, older benchmarks |
| Synthetic vignette generation | Time-limited; real FAERS cases are better for our task |

---

## Pipeline Comparison

```
MeditronFO Pipeline:
  8 Medical QA Datasets → Normalize → Decontaminate
  → Gold-Label Resample → SFT → Evaluate (LLM-as-judge)

Our "GemmraFO" Pipeline:
  FAERS + BioDEX + OnSIDES → Normalize → Decontaminate
  → Adversarial Negatives (12%)
  → Curriculum Sort (easy→hard) → SFT (Gemma 4 thinking mode, bf16 LoRA)
  → GRPO/DAPO (explored, FAILED) → Evaluate (metrics)
```

We adopt the paper's quality-assurance steps while:
- Using PV-specific data instead of general medical QA
- Using Gemma 4 thinking mode (paper's models didn't have it)
- Adding adversarial negative examples (12%) for robustness
- GRPO/DAPO explored but failed (reward variance collapse)

---

## Innovation Story for Judges

> "Inspired by EPFL's FullyOpenMeditron (May 2026, arXiv:2605.16215) — which
> demonstrated that medical SFT on open models can beat proprietary
> medical LLMs — we applied their auditable pipeline methodology to
> Google's Gemma 4 31B. Through iterative data quality improvements
> and evidence-driven debugging, we achieved 0.862 composite score
> across 3,645 pharmacovigilance samples, with 100% format compliance.
> We also explored GRPO/DAPO reinforcement learning and documented
> the failure (reward variance collapse) as a published negative result."

---

## Available Datasets from Paper's Pipeline

```python
# EPFL public data (supplementary, not primary for us)
from datasets import load_dataset

guidelines = load_dataset("epfl-llm/guidelines")  # 36K clinical guidelines
# We use these for ICH guideline text only, not for general medical QA
```
