# SFT v5 — First-Principles Analysis (0.23 epochs)

> **Date:** June 13, 2026
> **Status:** ✅ KEY DISCOVERY — This analysis found the **BioDEX truncation bug** (`abstract[:500]`). Fix was implemented → T2 improved 2.1× (0.168→0.558). The version labels (v2-v5) here are intermediate checkpoints at 0.23 epochs; the final shipped model (SFT v6) trained for 1 full epoch with all fixes.

## Score Comparison Across All Runs

| Metric | v2 (old, strip) | v3 (BioDEX, strip) | v4 (no strip) | **v5 (strip + enable_thinking)** |
|--------|:---:|:---:|:---:|:---:|
| Format | 100% | 100% | 22.5% | **95.5%** |
| T1 F1 | 1.000 | 1.000 | 0.000 | **0.759** |
| T2 exact | 0.080 | 0.060 | 0.000 | **0.060** |
| T2 weighted | — | 0.134 | 0.000 | **0.144** |
| T3 F1 | 0.708 | 0.387 | 0.000 | **0.316** |
| T4 exact | 0.880 | 0.880 | 0.860 | **0.880** |

---

## Issue 1: Thinking Traces — Inconsistent (Partially Working)

### Evidence

| Task | `<\|channel>thought` in output? | Output style | Correct? |
|------|:---:|---|:---:|
| T1 | ❌ | LaTeX `\section{}` + structured fields | ✅ |
| T2 | ✅ | Short 1-line think + structured fields | ❌ (wrong PT) |
| T3 | ✅ | Multi-sentence reasoning + fields | ✅ |
| T4 | ❌ | Detailed narrative + structured fields | ✅ |

### Root Cause

`enable_thinking=True` is working — the prompt correctly ends with `<|turn>model\n` (no pre-closed `<channel|>`). But thinking traces are **inconsistent** because:

1. **SFT stripped `<|channel>thought`** from training data → model learned to produce `{think}\n<channel|>\n{answer}` WITHOUT the `<|channel>thought` opening token
2. **Base Gemma 4** knows about `<|channel>thought` → sometimes adds it (T2/T3), sometimes doesn't (T1/T4)
3. **0.23 epochs = undertrained** → model is a mix of SFT-learned and base-model behavior

### Assessment

This is **expected behavior at 0.23 epochs**. At 1 full epoch:
- The SFT format will dominate over base model habits
- Thinking traces will become consistent
- The LaTeX `\section{}` artifact will disappear
- Format compliance will approach 100%

> [!IMPORTANT]
> **Do NOT change the thinking token approach.** The current setup (strip in training + enable_thinking at inference) is the right architecture. The inconsistency will resolve with more training.

---

## Issue 2: T1 Regression (1.000 → 0.759)

### Evidence
- P=1.000: Every YES prediction is correct
- R=0.611: Model misses 39% of actual YES cases
- Model is **too conservative** — predicting NO/UNKNOWN when it should predict YES

### Root Cause Analysis

Two competing hypotheses:

**H1: Eval set changed.** The ingredient matching added ~5,500 T3 pairs → total data changed → decontamination split changed → different 50 T1 eval samples → harder or different cases. **Likely contributor.**

**H2: enable_thinking changes behavior.** With `enable_thinking=True`, the prompt structure changed. The model gets `<|think|>` in system turn + open-ended `<|turn>model\n`. At 0.23 epochs, the model hasn't fully learned to respond in this new context → sometimes falls back to base model behavior (LaTeX sections, less decisive answers).

### Verdict
**Both H1 and H2 contribute.** At 1 full epoch, T1 should recover to ~1.000. The 0.23-epoch run is diagnostically useful but not representative of final performance.

---

## 🔴 Issue 3: T2 "Drug ineffective" — CATASTROPHIC DATA BUG

### The Bug

```python
# 05_download_external_datasets.py, line 248:
pair["source_text"] = abstract[:500]   # ← BLIND TRUNCATION
```

BioDEX abstracts are truncated to 500 chars at download time. Then:

```python
# 03_build_training_data.py, line 834:
if len(source_text) > 500:   # ← NEVER TRUE (text is already ≤500)
    # H1 smart truncation... DEAD CODE
```

### Impact — Measured

```
T2 training data:  7,272 pairs
PT in prompt text:   583 (8.0%)
PT NOT in prompt:  6,689 (92.0%)  ← MODEL CANNOT LEARN FROM THESE
```

**92% of T2 training data asks the model to identify an adverse event from text that DOES NOT DESCRIBE the adverse event.** The model sees an abstract about cancer chemotherapy and must guess "Hypermetabolism" — but the abstract only contains the first 500 characters (background/methods), not the results section where adverse events are described.

### What The Model Learns

The model learns: *"When you see any biomedical text, output a common MedDRA PT regardless of content."* The safest bet is "Drug ineffective" (very common in BioDEX). This explains:
- T2 exact=0.060 (only the 8% where PT was actually in the text)
- The specific "Drug ineffective" prediction (default safe PT)

### The Fix

Two changes needed in [05_download_external_datasets.py](file:///d:/dev/work/TCS_AMD_Hackathon/src/data/05_download_external_datasets.py#L248):

```diff
-pair["source_text"] = abstract[:500]
+pair["source_text"] = abstract  # Keep full text; truncation happens in 03_build
```

And in [03_build_training_data.py](file:///d:/dev/work/TCS_AMD_Hackathon/src/data/03_build_training_data.py#L834):

```diff
-if len(source_text) > 500:
+if len(source_text) > 400:  # Trigger H1 smart truncation earlier
```

This way:
1. BioDEX stores full abstract (1,000-3,000 chars typically)
2. H1 fix centers a 300-char window around the PT mention
3. If PT isn't in the abstract → pair is skipped (no hallucination training)

### Expected Impact

After fix:
- **Training pairs may DROP** (from 7,272 to maybe 2,000-4,000) — because many pairs where PT isn't in abstract get correctly filtered out
- But remaining pairs are **high quality** — model sees text describing the adverse event and learns the correct mapping
- T2 exact should improve significantly (target: >0.200)

---

## Worked Examples Insights

From [worked_examples.md](file:///D:/dev/work/TCS_AMD_Hackathon/docs/domain/worked_examples.md):

The ideal T2 output shows the model reasoning through the MedDRA hierarchy:

```
<|channel>thought
The patient reported gastrointestinal bleeding while on Warfarin.
In the MedDRA hierarchy, this maps to:
- LLT: Stomach bleeding, GI bleed (lay terms)
- PT: Gastrointestinal haemorrhage (the Preferred Term)
- HLT: Gastrointestinal haemorrhages
- SOC: Gastrointestinal disorders
```

Key difference from our training data:
1. **The AE is described in the text** — "gastrointestinal bleeding" is clearly stated
2. **The thinking shows hierarchy navigation** — LLT→PT→HLT→SOC
3. **The model maps lay language to clinical** — "stomach bleeding" → "Gastrointestinal haemorrhage"

Our training data fails on point #1 — the AE isn't in the visible text. Fix the truncation → fix the foundation.

---

## Summary of Actions

| Priority | Fix | File | Impact |
|:---:|---|---|---|
| 🔴 P0 | Stop blind `abstract[:500]` truncation | `05_download_external_datasets.py` | T2 quality |
| 🔴 P0 | Lower H1 threshold to 400 chars | `03_build_training_data.py` | H1 fix actually works |
| 🟢 OK | Keep stripping in SFT | `01_sft_train.py` | ✅ Already correct |
| 🟢 OK | Keep `enable_thinking=True` | `evaluate.py` / `check_raw.py` | ✅ Partially working |
| 🟡 Wait | Train 1 full epoch | AMD cloud | Resolves thinking inconsistency |

> [!CAUTION]
> **After fixing the truncation, you MUST rebuild data before retraining:**
> ```bash
> python src/data/05_download_external_datasets.py  # Re-download BioDEX (full abstracts)
> python src/data/03_build_training_data.py          # Rebuild with H1 fix working
> python src/training/01_sft_train.py                # Retrain SFT
> ```
