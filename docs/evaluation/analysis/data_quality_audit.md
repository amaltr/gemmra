# Pipeline Output Analysis — Improvements to Integrate

> **Date:** June 11, 2026
> **Status:** ✅ RESOLVED — All issues found here were fixed in subsequent data rebuilds (SFT v5/v6). This audit led to the BioDEX truncation discovery and T3 rebalancing.
> **Value:** Documents our data-first debugging methodology. Good for presentation narrative.

## What's Working Well ✅

Your 3 commits (`cd81bd6`, `9e9f326`, `c995f16`) added excellent improvements:
- **Drug name normalization** (strips dosage forms/strengths) — smart
- **5-strategy cascade** (NDA → exact → normalized → prod_ai → ingredient)
- **Curriculum learning** support in SFT trainer
- **YAML config loading** for hyperparameters
- Dead code cleanup

The pipeline ran end-to-end and produced **28,164 training pairs + 3,124 eval pairs**.

---

## Issues Found — Ranked by Impact

### 🔴 Issue 1: T3 is Severely Under-Represented (CRITICAL)

**Log evidence:**
```
T1: 8,948 pairs
T2: 7,231 pairs
T3: 4,782 pairs   ← 46% less than T1
T4: 7,203 pairs
```

**Root cause:** T3 balance logic caps at `min(max_pairs // 2, YES, NO)` where `max_pairs=10000`. With only 2,644 labelled (YES) and 18,918 unlabelled (NO), the bottleneck is YES. The balance produces `2,644 × 2 = 5,288` pre-decontamination, minus 10% holdout = ~4,782.

**Why it matters:** The model will be weakest on the exact task that has the richest ground truth (OnSIDES). T3 ("is this AE labelled?") is a high-value differentiator — most hackathon teams won't have real label lookup ground truth.

**Fix — Allow asymmetric class ratios for T3:**

Instead of strict 50/50 balance (which wastes 16K NO examples), use a 1:3 ratio (YES:NO). This is standard in pharmacovigilance where unlabelled events are naturally more common. The model should learn the real distribution.

```python
# Current (wastes data):
target = min(max_pairs // 2, len(yes_pairs), len(no_pairs))

# Proposed (uses more data, realistic distribution):
yes_target = min(len(yes_pairs), max_pairs // 4)  # ~25% YES
no_target = min(len(no_pairs), max_pairs * 3 // 4)  # ~75% NO
```

This would give T3 ~8,000+ pairs instead of 5,288.

---

### 🟡 Issue 2: 28,438 Dropped = 57% of FAERS Cases Have No OnSIDES Coverage

**Log evidence:**
```
OnSIDES matched: 2,644 labelled | 18,918 unlabelled | 28,438 dropped (no drug coverage)
```

Only **43%** of sampled FAERS drug entries matched any OnSIDES drug name. This is expected because:
- FAERS `drugname` includes brand names, abbreviations, misspellings
- OnSIDES `source_product_name` is clean FDA label text
- e.g., FAERS: "ADVIL" vs OnSIDES: "ibuprofen tablets, usp 200mg"

**Fix — Add one more matching strategy: prod_ai ↔ OnSIDES drug_name_normalized:**

Currently strategy 5 tries `prod_ai in normalized_drugs`, but `prod_ai` is the raw active ingredient (e.g., "IBUPROFEN") while `drug_name_normalized` still has extra tokens. Adding a **word-containment check** could catch more:

```python
# After all set-based checks fail:
if not has_coverage and prod_ai and prod_ai not in ('', 'nan'):
    # Check if prod_ai is contained in any normalized drug name
    for norm_drug in normalized_drugs:
        if prod_ai in norm_drug or norm_drug in prod_ai:
            has_coverage = True
            # Check all PTs for that drug
            is_labelled = any(
                pt == pt_term for (nd, pt_term) in normalized_set 
                if nd == norm_drug
            )
            break
```

> [!WARNING]
> This substring matching is expensive — only use it on the already-sampled subset, NOT the full 6.9M rows. The current early sampling (`max_pairs * 5`) makes this safe.

---

### 🟡 Issue 3: Ingredient Matching Not Working

**Log evidence:**
```
Match strategies: NDA=True, drug_name=True, normalized=True, ingredient=False
```

The RxNorm ingredient chain **loaded** ("Loading RxNorm ingredient mapping...") but `ingredient=False` in T3 means the `ingredient_name` column ended up all NaN in the lookup table.

**Root cause:** The ingredient mapping code in `04_download_onsides.py` builds the map but the column check `has_ingredient = 'ingredient_name' in onsides.columns and onsides['ingredient_name'].notna().any()` fails because the map didn't join properly (likely column name mismatch in the p2r → i2p → ing chain).

**Fix:** Debug the ingredient chain by printing the intermediate column names. But this is lower priority since drug_name_normalized already provides good coverage. I'd deprioritize this.

---

### 🟡 Issue 4: Section Info Not Used in T3 Training

**Log evidence:**
```
PAE section: label_section
```

OnSIDES has `label_section` (e.g., "AR" for Adverse Reactions, "BW" for Boxed Warning), but T3 always hardcodes `section = 'Adverse Reactions'`:

```python
section = 'Adverse Reactions'  # Default — OnSIDES has section info in lookup
```

The lookup table has this data. Using it would make the training richer and more accurate:
- "AR" → Adverse Reactions
- "BW" → Boxed Warning (more serious!)
- "WP" → Warnings and Precautions

**Fix:** Map the section codes and use them:

```python
SECTION_MAP = {
    'AR': 'Adverse Reactions',
    'BW': 'Boxed Warning',
    'WP': 'Warnings and Precautions',
}
# In the labelled branch:
section_raw = row_section if row_section else 'AR'
section = SECTION_MAP.get(section_raw, section_raw)
```

---

### 🟢 Issue 5: Unnecessary `nda_display` Variable (Minor)

**Code:**
```python
nda_display = row.get('nda_num', 'unknown') if row.get('nda_num') else 'unknown'
```

This is computed but never used (was removed from the input_text template). Dead variable. Harmless but should be cleaned up.

---

### 🟢 Issue 6: T1 Imbalance — 19,690 YES vs 13,192 NO (Minor)

**Log evidence:**
```
YES: 19,690 | NO sampled: 13,192 | Final: 10,000
```

T1 has more YES (serious) than NO cases. The balance logic samples evenly (5K each), which is correct. But the raw ratio (60/40) suggests the quality filter (`HP/LIT` reporter types) biases toward serious cases. This is expected and acceptable — healthcare professionals report more serious cases.

---

## Summary — Priority Matrix

| # | Issue | Priority | Impact | Effort |
|---|-------|----------|--------|--------|
| 1 | T3 under-represented (4,782 vs 8K+ target) | 🔴 High | Training quality | 10 min |
| 2 | 57% FAERS drugs have no OnSIDES coverage | 🟡 Medium | T3 coverage | 30 min |
| 3 | Ingredient mapping all NaN | 🟡 Medium | Minor coverage | Debug |
| 4 | Section info not used in T3 | 🟡 Medium | Training richness | 15 min |
| 5 | Dead `nda_display` variable | 🟢 Low | Code hygiene | 2 min |
| 6 | T1 class ratio 60/40 | 🟢 Low | Expected, OK | N/A |

## Recommendation

**Fix Issues 1 and 4 now** — they're quick wins that directly improve model quality. Issue 2 is a nice-to-have but riskier (substring matching could create false positives). Issue 3 needs debugging time you may not have.
