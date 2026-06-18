# 🔬 SFT Evaluation Forensic Analysis

> **Run Date:** June 13, 2026
> **Status:** ⚠️ EARLY ANALYSIS — This was eval of SFT v3 (BEFORE BioDEX fix and T3 rebalancing). T2=0.020 here was fixed to 0.667 in final eval. GRPO sections (§4-5) are outdated — GRPO was tested and failed.
> **Value:** Excellent documentation of early debugging methodology, train/eval loss analysis, and data quality diagnosis. The T2 data design problem identified here was the trigger for the BioDEX truncation fix.
> **Model:** Gemma 4 31B (bf16 LoRA, r=64)
> **Training:** 885 steps, 1 epoch, ~100 min on MI300X

---

## 1. Training Dynamics — What the Numbers Tell Us

### Loss Trajectory

| Phase | Epoch | Train Loss | Grad Norm | Learning Rate |
|-------|-------|-----------|-----------|---------------|
| Start | 0.011 | 1.146 | 3.042 | 1e-5 (warmup) |
| Early | 0.023 | 0.585 | 0.678 | 2.1e-5 |
| 25% | 0.260 | 0.020 | 0.031 | 4.4e-5 (peak) |
| 50% | 0.509 | 0.022 | 0.073 | 2.7e-5 |
| 75% | 0.724 | 0.020 | 0.047 | 9.9e-6 |
| 100% | 0.995 | 0.020 | 0.034 | 6.3e-9 |
| **Eval** | 1.0 | — | — | — |

**Eval loss: 0.098**

### Key Observations

**A. Loss plateau at epoch 0.25**
Train loss dropped 57× (1.146 → 0.020) in the first ~220 steps, then flatlined for the remaining 665 steps. The model learned the structural pattern (thinking traces + answer format) extremely fast.

**B. Train/Eval gap: 4.9× (0.020 vs 0.098)**
Previous run (with double-wrap bug): 4.3× (0.027 vs 0.118).
This run is **slightly more overfit** but on **correct data** — the model memorized the correct templates.

> [!NOTE]
> The 4.9× gap does NOT mean the model is broken. It means the model has memorized the exact phrasing of our diversity engine templates (e.g., "Let me assess this case..."). The eval set has different template instantiations from the same engine, so perplexity is higher. The functional performance (100% format, T1=1.0, T4=1.0) proves the model learned the TASK, not just the exact words.

**C. Grad norms collapsed to 0.03-0.09**
By epoch 0.25, gradient norms were tiny. The model was no longer receiving useful learning signal. The remaining 75% of training was essentially no-op — the cosine LR schedule drove the rate to near-zero anyway, which is correct behavior.

**D. Runtime: 6032s (100 min)**
4.69 samples/sec × batch_size 8 × grad_accum 4 = effective batch 32.
885 steps × 32 = 28,320 samples processed. This matches the dataset size (~28,265 train).

---

## 2. The Headline: Double-Wrap Fix WORKED

| Metric | Before Fix | After Fix | Δ |
|--------|-----------|-----------|---|
| Format Compliance | 4% | **100%** | +96% |
| Model output | "own own own own..." | Proper thinking + structured answer | ✅ |
| Train loss | 0.027 | 0.020 | Slightly better |
| Eval loss | 0.118 | 0.098 | Slightly better |

Format compliance going from 4% → 100% is **definitive proof** that the double-wrap tokenization was the root cause of model collapse. The model now generates proper `<|channel>thought...` traces followed by structured answers for every single sample.

---

## 3. Task-by-Task Analysis

### T1 Seriousness: F1=1.000 (P=1.000, R=1.000) ⚠️

**Surface read:** Perfect score.

**Evidence-based interpretation:**
- 50 samples, binary classification (YES/NO).
- The training data embeds explicit outcome narratives: "The patient required hospitalization", "The patient subsequently died." These are unmistakable signals.
- The model just needs to detect these narrative phrases → map to ICH E2A criteria. This is genuinely easy for a 31B model after SFT.
- **Confidence: 7/10 that this is real.** On 50 samples, perfect score is plausible for a straightforward binary task with strong textual signal. Full eval (all samples) will confirm.

**Risk:** Perfect score on TRAINING-DISTRIBUTION data doesn't guarantee performance on novel narratives (ambiguous cases, edge cases where outcome is implied but not stated).

### T2 MedDRA Coding: exact=0.020, fuzzy=0.040 🔴 CRITICAL

**This is the real problem.** 1/50 exact match, 2/50 fuzzy.

**Root cause — confirmed by code evidence:**

The ground truth labels for PHEE and ADE Corpus pairs are set at [03_build_training_data.py:835](file:///d:/dev/work/TCS_AMD_Hackathon/src/data/03_build_training_data.py#L835):
```python
pt = ade.strip()  # Raw entity text from corpus, NOT a real MedDRA PT
```

For BioDEX, the target IS a real MedDRA term ([line 858](file:///d:/dev/work/TCS_AMD_Hackathon/src/data/03_build_training_data.py#L858)):
```python
pt = reaction.strip()  # Actual MedDRA-coded reaction term
```

**The training data is internally inconsistent:**

| Source | Example Target PT | Is Real MedDRA? | % of T2 data |
|--------|------------------|-----------------|--------------|
| PHEE | "severe muscle pain" | ❌ Raw entity text | ~15-25% |
| ADE Corpus | "hepatotoxicity" | ⚠️ Sometimes (medical but not always exact MedDRA) | ~7% |
| BioDEX | "Rhabdomyolysis" | ✅ Real MedDRA PT | ~60-75% |

**What happens during inference:**

1. The model was SFT-trained on a MIX of raw entity text and real MedDRA PTs
2. Gemma 4 31B's base knowledge already knows MedDRA terminology
3. During eval, the model may output the *correct MedDRA PT* (e.g., "Myalgia") but the ground truth says "severe muscle pain" (from PHEE) → **exact match fails**
4. Conversely, for BioDEX eval samples, the model might output the correct BioDEX term but with slight variation → **exact match still fails**

**Additionally:** The input for PHEE/ADE Corpus pairs has the ADE **redacted** as `[ADVERSE EVENT]`. The model must infer the ADE from surrounding context alone. This is a genuinely hard task — even with the correct clinical understanding, producing the exact string that was redacted requires memorization, not reasoning.

**This is a data design problem, not a model problem.**

### T3 Labelling: F1=0.711 (P=0.640, R=0.800) 🟡 MODERATE

**Interpretation:**
- R=0.80: Model catches 80% of labelled events. Good.
- P=0.64: Model says "LABELLED: YES" when it should be "NO" 36% of the time. Overestimates labelling.
- This is expected behavior: the model learned from OnSIDES ground truth, which covers common drugs well but may have gaps for rare drugs in the eval set.
- **T3 is the best candidate for GRPO improvement.** The correctness reward (weight 2.0) directly penalizes wrong YES/NO labels.

### T4 Causality: exact=1.000, weighted=1.000 ⚠️

**Surface read:** Perfect score across all 6 WHO-UMC levels.

**Evidence-based interpretation:**
- The training data generates causality labels from `compute_causality()` — a deterministic rule-based function using FAERS fields (dechallenge, rechallenge, temporal gap, confounders).
- The eval data uses the **same function** to generate ground truth.
- The model learned to reverse-engineer the exact rules from the clinical narrative. Given that the narrative is generated from those same fields (e.g., "When the drug was stopped, symptoms resolved" → positive dechallenge → Probable), the mapping is deterministic.
- **This is circular but correct for SFT.** The model faithfully reproduces the WHO-UMC rule system encoded in our training data.
- **Confidence: 8/10 that this is real.** The narrative→rules mapping is systematic enough that a 31B model can learn it perfectly.

**Risk:** The model learned OUR implementation of WHO-UMC rules, not necessarily the clinical community's interpretation. Edge cases (conflicting evidence, partial data) may break it.

---

## 4. Overall Assessment

```
╔═══════════════════════════════════════════════════════════╗
║  FORMAT COMPLIANCE: 100%  — Double-wrap fix confirmed     ║
║  T1 (Seriousness):  1.000 — Strong, validate on full set  ║
║  T2 (MedDRA):       0.020 — BROKEN (data design issue)    ║
║  T3 (Labelling):    0.711 — Decent, GRPO will improve     ║
║  T4 (Causality):    1.000 — Strong (circular but correct)  ║
╠═══════════════════════════════════════════════════════════╣
║  BLOCKING ISSUE: T2 must be fixed before GRPO             ║
╚═══════════════════════════════════════════════════════════╝
```

### What GRPO Can Fix vs What It Can't

| Issue | Can GRPO fix? | Why |
|-------|--------------|-----|
| T3 P=0.64 | ✅ Yes | `correctness_reward` directly penalizes wrong YES/NO |
| T1 edge cases | ✅ Yes | RL explores beyond training distribution |
| T4 edge cases | ✅ Yes | Partial credit in reward handles ordinal proximity |
| T2 = 2% | ❌ **No** | If ground truth labels are wrong, correctness reward trains on garbage |

> [!WARNING]
> **T2 is the blocking issue.** Running GRPO with T2 data as-is will actively damage the model's medical coding ability — the correctness reward will penalize it for producing correct MedDRA PTs because the ground truth contains raw entity text from PHEE/ADE Corpus.

---

## 5. What Needs to Happen

### Priority 1: Fix T2 Ground Truth (BEFORE GRPO)

**Options (ranked by quality):**

1. **Filter T2 to BioDEX only.** Remove PHEE and ADE Corpus T2 pairs entirely. BioDEX uses real MedDRA PTs. Reduces T2 volume but fixes label quality.

2. **Map PHEE/ADE entities to MedDRA PTs.** Use the model's own base knowledge or a lookup table. Complex but preserves data volume.

3. **Soften T2 eval.** Add medical synonym matching (e.g., "severe muscle pain" → "Myalgia" should get partial credit). Doesn't fix training data but makes eval more fair.

### Priority 2: Run GRPO Smoke Test

After T2 fix:
```bash
python src/training/02_grpo_train.py --smoke
```
Validates pipeline in ~30 min. Check:
- `reward_std > 0` (if zero → no learning signal)
- Loss decreasing
- No crashes

### Priority 3: Full GRPO Run

With validated pipeline and fixed T2 data.
