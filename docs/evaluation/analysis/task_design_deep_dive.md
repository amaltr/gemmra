# 🔬 Deep Dive: Task Design, Thinking Mode & Data Quality

> **Date:** June 10, 2026
> **Status:** ✅ MOSTLY IMPLEMENTED — Key recommendations adopted: thinking ON for all 4 tasks, OnSIDES replaced frequency heuristic for T3, hierarchical MedDRA scoring added (exact→synonym→fuzzy→SOC). DAPO training references are outdated (GRPO failed). T4 Unassessable category was NOT added.
> **Value:** Foundational document for understanding task template design decisions. Good for judge Q&A.

## TL;DR — Quick Decision Table

| Question | Current State | Recommended Change | Impact |
|----------|--------------|-------------------|--------|
| **T1 thinking mode** | OFF | **→ ON** | Major: judges see reasoning |
| **T3 thinking mode** | OFF | **→ ON** | Major: judges see reasoning |
| **T1 rationale evaluation** | Partially rewarded | **Improve faithfulness reward** | Medium |
| **T2 HLT evaluation** | Not evaluated | **Keep as-is** (HLT is in eval, not output) | Low |
| **T3 data quality** | ❌ Frequency heuristic (broken) | **→ Use OnSIDES database** | **Critical** |
| **T4 ground truth** | Scoring function (good) | **Add edge case handling** | Medium |

---

## 1. Should ALL Tasks Have Thinking Mode?

### Current State
```
T1 (Seriousness):  thinking=OFF → direct "SERIOUS: YES/NO"
T2 (MedDRA):       thinking=ON  → <|channel>thought → answer
T3 (Labelling):    thinking=OFF → direct "LABELLED: YES/NO"
T4 (Causality):    thinking=ON  → <|channel>thought → answer
```

### Research Finding
The conventional wisdom says "CoT hurts simple binary classification." But that
research is about **accuracy** — whether CoT helps the model get the *right answer*.

For a **hackathon demo** and for **pharmacovigilance regulatory requirements**, the
calculus is completely different:

> **ICH E2A regulators explicitly require documented reasoning** for seriousness
> assessments. Simply saying "SERIOUS: YES" without explaining which criterion
> was met is **regulatory non-compliance**.

### ✅ Recommendation: Turn Thinking ON for ALL 4 Tasks

**Reasons:**

1. **Hackathon judges** — seeing reasoning traces for every task is dramatically
   more impressive. A model that just says "YES/NO" looks like a classifier.
   A model that explains WHY looks like a clinical expert.

2. **Regulatory alignment** — PV assessments require documented rationale.
   Our model should demonstrate this.

3. **DAPO training** — with thinking ON for all tasks, our faithfulness and
   reasoning quality rewards can evaluate ALL outputs, not just T2/T4.
   This gives DAPO 4x more signal to learn from.

4. **The "think" content is different from the output "Rationale":**
   - `<|channel>thought` = internal step-by-step reasoning (may be hidden in production)
   - `Rationale:` = the externally-facing explanation in the structured output
   - Both should exist. The thinking traces show HOW the model arrived at the rationale.

### Concrete Format (All 4 Tasks):

**Task 1:**
```
<|channel>thought
Evaluating against ICH E2A seriousness criteria.
The reported outcome codes include: HO.
Matching criteria: HO (Hospitalization).
At least one ICH E2A criterion is satisfied — the patient required
inpatient hospital admission for the adverse event.
<channel|>
SERIOUS: YES
Criteria met: HO (Hospitalization)
Rationale: The case meets ICH E2A seriousness definition based on
the reported outcome of hospitalization.
```

**Task 3:**
```
<|channel>thought
Checking whether this adverse event is listed in the drug's approved label.
Reviewing known adverse reaction profile for the drug.
The adverse event is well-documented in the drug's label under
Adverse Reactions section.
<channel|>
LABELLED: YES
Label section: Adverse Reactions
Rationale: This adverse event is documented in the drug's approved
product label as a known adverse reaction.
```

---

## 2. Are We Evaluating Rationale During Fine-Tuning?

### Current State — YES, Partially

Our DAPO reward functions DO evaluate rationale quality, but **only for tasks
with thinking=ON** (T2, T4). Here's what each reward checks:

| Reward Function | What It Checks | T1 | T2 | T3 | T4 |
|-----------------|---------------|-----|-----|-----|-----|
| `format_reward` | Has thinking tokens + structured output | ⚠️ Gets 0.7 (no think) | ✅ 1.0 | ⚠️ Gets 0.7 | ✅ 1.0 |
| `task_structure_reward` | Has Rationale:/Evidence: fields | ✅ Checks Rationale | ✅ | ✅ Checks Rationale | ✅ |
| `reasoning_quality_reward` | Domain terms in thinking block | ❌ No think block | ✅ | ❌ No think block | ✅ |
| `faithfulness_reward` | Case-specific data references | ✅ | ✅ | ✅ | ✅ |

### The Problem
With thinking=OFF for T1/T3, the `reasoning_quality_reward` gives them
**0.0 every time** because there's no thinking block to evaluate.
This means T1/T3 get less reward signal → DAPO learns less about their reasoning.

### ✅ Fix: After turning thinking ON for all tasks, all 4 rewards apply to all 4 tasks.
No code changes needed in the reward functions — they already handle Gemma 4
thinking tokens correctly.

---

## 3. Task 2: Hierarchical F1 and HLT

### Your Question
> "The output format has PT and SOC but no HLT. So how does hierarchical F1 work?"

### Answer
**Hierarchical F1 is about EVALUATION, not OUTPUT FORMAT.**

Here's how it works:

```
Model predicts:  MedDRA PT: "Gastrointestinal bleeding"
Ground truth is: MedDRA PT: "Gastrointestinal haemorrhage"

Exact match?  ❌ No (different PT)
HLT match?    ✅ Yes! Both map to HLT "Gastrointestinal haemorrhages"

Score: Partial credit (e.g., 0.5 instead of 1.0)
```

The model doesn't need to OUTPUT HLT. The **evaluation script** should look up
both the predicted PT and the ground truth PT in the MedDRA hierarchy, and if
they share the same parent HLT, award partial credit.

### Current State in evaluate.py
Our evaluation does **exact string match** on PTs. We don't have a MedDRA
hierarchy lookup to award HLT-level partial credit.

### ✅ Recommendation: Keep current approach (exact PT match)
- We don't have a licensed MedDRA hierarchy file
- Exact PT match is a stricter metric (better for showing model quality)
- In the presentation, we can mention "we use strict exact-match evaluation"
- **This is fine for the hackathon**

### On Rationale in T2
The `Rationale:` field in T2 output IS being evaluated by:
- `task_structure_reward` → checks "Rationale:" exists (+0.5)
- `faithfulness_reward` → checks for case-specific references
- `reasoning_quality_reward` → checks domain terminology in thinking block

So yes, we are rewarding good rationale, not just the correct PT.

---

## 4. Task 3: The Labelling Data Problem (CRITICAL)

### Your Discovery Is Absolutely Correct

You tried string matching between FAERS adverse events and DailyMed label text,
and it failed. This is because:

1. **FAERS uses MedDRA PTs** (standardized: "Gastrointestinal haemorrhage")
2. **Drug labels use free text** (unstructured: "bleeding in the stomach or intestines")
3. **Synonyms, stages, severity modifiers** differ completely
4. **Your NDA 125057 example** (Vemurafenib): FAERS reports "Malignant melanoma stage I"
   but the label says "cutaneous squamous cell carcinoma" — completely different terminology

### Our Current T3 Approach Is WRONG

Our current `build_t3_pairs()` uses a **frequency heuristic**:
```python
# If >5% of FAERS reports for this drug mention this AE, assume it's "labelled"
co_occur['is_labelled'] = co_occur['frequency'] > 0.05
```

This is fundamentally flawed because:
- High-frequency in FAERS ≠ listed in drug label
- Low-frequency in FAERS ≠ not listed in drug label
- Example: Aspirin + headache is reported frequently in FAERS but headache
  IS listed in aspirin's label. The heuristic gets this right by accident.
- Example: A rare cancer drug + nausea might be low-frequency in FAERS
  but nausea IS listed. The heuristic gets this WRONG.

### ✅ Solution: Use the OnSIDES Database

**OnSIDES** (https://github.com/tatonetti-lab/onsides) is exactly what we need:

- **Free, open-source** database of adverse drug events extracted from FDA labels
- **Uses PubMedBERT** to extract ADEs from DailyMed structured product labels  
- **Maps to MedDRA PTs** — same vocabulary as FAERS!
- **Updated quarterly** — much more current than SIDER (which stopped in 2015)
- **Covers FDA-approved drugs** — matches our NDA-based FAERS data

### How to Use OnSIDES for T3:

```
FAERS case: Drug=Warfarin (NDA 009218), AE="Gastrointestinal haemorrhage"
                    ↓
OnSIDES lookup: NDA 009218 → list of labelled MedDRA PTs
                    ↓
Is "Gastrointestinal haemorrhage" in that list?
  → YES → LABELLED: YES
  → NO  → LABELLED: NO (potential unlabelled/unexpected AE)
```

This gives us **real ground truth** instead of frequency heuristics.

### Do We Need LLM-as-Judge?
**Not if we use OnSIDES.** The OnSIDES database already does the hard work of
extracting and standardizing ADEs from drug labels using NLP. Since both OnSIDES
and FAERS use MedDRA PTs, we can do direct lookup without fuzzy matching.

LLM-as-judge would only be needed if:
- OnSIDES doesn't cover some of our drugs (fallback option)
- We need to match non-standard terminology (rare edge case)

### Implementation Plan:

```
Step 1: Download OnSIDES flat files from GitHub
Step 2: Load into DuckDB alongside FAERS
Step 3: Join: FAERS.nda_num → OnSIDES.product_label → OnSIDES.adverse_effects
Step 4: For each FAERS case, check if AE MedDRA PT exists in OnSIDES for that drug
Step 5: Generate T3 training pairs with REAL labelling ground truth
```

---

## 5. Task 4: Ground Truth & Improvements

### Current State
Our scoring function in `compute_causality()` applies WHO-UMC rules:
- Certain: rechallenge + dechallenge + temporal (<90 days)
- Probable: dechallenge + temporal (<180 days) + no confound
- Possible: temporal (<365 days)
- Unlikely: confound OR temporal >730 days

### This Is Actually Good
Since FAERS has no explicit causality field, a scoring function is the
standard approach. Our implementation correctly uses:
- ✅ Dechallenge (dechal field)
- ✅ Rechallenge (rechal field)
- ✅ Temporal gap (event_dt - start_dt)
- ✅ Indication-AE confounding

### Improvements Recommended

1. **Add "Conditional/Unassessable" categories:**
   ```python
   # When key evidence is missing
   if gap_days is None and not dechal and not rechal:
       return 'Unassessable'  # Insufficient data to assess
   ```

2. **Add concomitant drug consideration:**
   - If the case has multiple suspect drugs (role_cod = 'SS' secondary suspect),
     mention this in the causality reasoning
   - Currently we only look at primary suspect

3. **Add dose-response consideration:**
   - The FAERS DRUG file has dose information
   - If dose was increased before AE onset → stronger causal signal
   - This is part of WHO-UMC criteria but we're not using it

4. **Edge case: very short temporal gap (<1 day):**
   - Some AEs occur within hours (anaphylaxis, infusion reactions)
   - Our current code treats gap_days=0 correctly, but the reasoning
     should mention "immediate onset" for stronger causal narrative

---

## 6. Summary of All Changes Needed

### Priority 1 (Critical — Do Before Training)
- [ ] **T3: Replace frequency heuristic with OnSIDES database lookup**
- [ ] **All tasks: Turn thinking=ON** (change `thinking: False` → `thinking: True` in T1, T3)
- [ ] **Update training data builder** to include `<|channel>thought` blocks for T1/T3

### Priority 2 (Improves Quality)
- [ ] **T4: Add Unassessable/Conditional categories** for cases with missing evidence
- [ ] **T4: Include concomitant drug info** in reasoning
- [ ] **Update reward functions** to verify T1 criteria citation in thinking block

### Priority 3 (Nice to Have)
- [ ] **T2: Add hierarchical F1** to evaluation (requires MedDRA hierarchy data)
- [ ] **T4: Add dose-response reasoning** from FAERS DRUG dose fields
