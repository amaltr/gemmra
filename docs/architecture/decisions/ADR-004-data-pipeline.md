# ADR-004: Data Pipeline & Sources

**Status:** ✅ DECIDED (Updated June 11, 2026 — reflects actual production results)  
**Date:** June 9, 2026

---

## Data Sources

### Primary: FAERS (FDA Adverse Event Reporting System)
- **URL:** https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- **Format:** ASCII files, `$`-delimited, 7 tables linked by `primaryid`
- **Size:** ~60MB zipped per quarter, ~300MB unzipped
- **Quarters used:** 2019 Q1 – 2026 Q1 (29 quarters, ~7M+ cases)
  - *Rationale:* 29 quarters is sufficient (0.3% sampling rate for ~21K pairs). Older quarters add schema complexity and folder structure inconsistencies without changing the output. AllDeletedCases.txt (2019Q1) covers all historical deletions.
- **Storage:** Downloads to ephemeral 3.1 TB overlay, NOT persistent storage
- **License:** Public domain (US Government data)
- **Fields used:** DEMO, DRUG, REAC, OUTC, THER, INDI, RPSR
- **Tasks served:** T1 (seriousness), T2 (MedDRA PT), T4 (causality)

### Secondary: OnSIDES Database (Drug Label ADEs)
- **Source:** https://github.com/tatonetti-lab/onsides (v3.1.1)
- **Method:** PubMedBERT NLP extraction of adverse drug effects from FDA labels → mapped to MedDRA PTs
- **Schema:** Normalized with 7 CSVs; key tables: `product_label`, `product_adverse_effect`, `vocab_meddra_adverse_effect`
- **Join strategy:** 6-strategy cascade matching (see below)
- **Tasks served:** T3 (labelling status ground truth)
- **Why NOT DailyMed:** DailyMed API uses free-text prose in labels; FAERS uses MedDRA PTs. String matching between the two fails empirically (proven via NDA 125057 / Vemurafenib test). OnSIDES bridges this gap via PubMedBERT NLP extraction.

## Critical Data Decisions

### ⚠️ MedDRA Full Hierarchy is NOT Available
- MedDRA is proprietary (ICH). Full LLT→PT→SOC files require subscription.
- **Workaround:** FAERS `REAC` table already has coded PTs. Use those as ground truth.
- For verbatim-to-PT training, use FAERS PTs in clinical context.

### ⚠️ FAERS Narratives are NOT Public
- Free-text patient narratives are stripped from public FAERS data (privacy).
- **Workaround:** Construct synthetic narratives from structured fields.

### ✅ Data Cleaning Strategy
- **DELETED case filtering** — FDA retracts cases via DELETED files; all retracted `primaryid` values are excluded before any processing. Handles both CSV-header and headerless delete file formats (some quarters have plain ID lists, not CSVs).
- **Deduplication** — Keep only the highest `caseversion` per `caseid`
- **Reporter quality** — Filter `rpsr_cod IN ('HP', 'LIT')` (healthcare professional + literature)
- **`rept_cod` NOT filtered** — Both EXP (expedited/serious) and PER (periodic/non-serious) are kept to provide natural class balance for T1 seriousness training. Filtering to EXP-only would create severe bias toward serious cases. `rept_cod` is included in the output for downstream use.
- **UNION ALL BY NAME** — Quarters combined by column name (not position) to handle schema evolution (e.g., DRUG table gained `prod_ai` in 2014Q3)
- **OUTC pre-aggregation** — Outcome codes are aggregated to a comma-separated string per case at the SQL level, preventing row fan-out from the one-to-many OUTC join
- **n_concomitant pre-computation** — Concomitant drug count is computed via SQL subquery, avoiding O(n²) runtime computation in the training data builder
- **Case-insensitive file matching** — File discovery uses lowercase comparison for Linux compatibility on AMD cloud
- **ZIP integrity validation** — Existing downloads verified via `zipfile.is_zipfile()` to detect partial/corrupt downloads
- **Early sampling in training data builder** — All 4 task builders sample before iterrows to prevent OOM on 3.5M+ row datasets
- **Drug name normalization** — Applied in T3 OnSIDES matching via 6-strategy cascade: strips dosage forms, strengths, routes, and punctuation for fuzzy matching (e.g., "ASPIRIN TABLETS, 325 MG" → "aspirin")

### ✅ T3 OnSIDES Matching — 6-Strategy Cascade
OnSIDES v3.1.1 has NO NDA column in its `product_label` table. Matching FAERS drugs to OnSIDES uses a priority cascade:
1. **NDA number** → PT (if `source_product_id` contains NDA-like values)
2. **Drug name (exact)** → PT (lowercase match against `source_product_name`)
3. **Drug name (normalized)** → PT (strips dosage forms, strengths, routes)
4. **Active ingredient (`prod_ai`)** → PT (FAERS `prod_ai` field against OnSIDES normalized names)
5. **RxNorm ingredient** → PT (via `product_to_rxnorm` + `vocab_rxnorm_ingredient` tables)
6. **Substring containment** → PT (last resort: "ibuprofen" matches "ibuprofen tablets")

Cases with no drug coverage in OnSIDES are dropped (not labelled as NO).

### ✅ T3 Class Balance — 1:3 YES:NO Asymmetric Ratio
Unlabelled AEs are naturally more common. Using strict 50/50 balance wastes NO examples and makes T3 artificially small. The 1:3 YES:NO ratio preserves realistic class distribution while keeping the task large enough for learning.

## Pipeline Architecture

```
FAERS ZIP (FDA) ──→ Unzip ──→ 7 TXT files + DELETED files
                                    │
                              DuckDB Load ($-delimited)
                                    │
                              Dedup (max caseversion)
                                    │
                              Filter DELETED cases
                                    │
                              Quality Filter (HP/LIT reporters)
                                    │
                              Master Join
                              ├─ OUTC pre-aggregated (comma-sep)
                              ├─ n_concomitant pre-computed
                              └─ rept_cod included
                                    │
                    ┌───────────────┼───────────────┬───────────────┐
                    │               │               │               │
              Label T1        Label T2        Label T3        Label T4
           (outcome→serious) (REAC.pt)   (OnSIDES lookup)  (WHO-UMC rules)
                    │               │               │               │
                    └───────────────┼───────────────┴───────────────┘
                                    │
                              Build JSONL
                         (Gemma 4 chat format, thinking=ON)
                                    │
                    ┌───────────────┴───────────────┐
              training_data.jsonl            eval_data.jsonl
                                    (MeditronFO decontamination)
```

## Actual Production Results (June 11, 2026)

| Task | Actual Pairs | Source |
|------|-------------|--------|
| T1 Seriousness | 8,949 pairs (60% YES / 40% NO) | FAERS OUTC codes |
| T2 MedDRA | 7,184 pairs | FAERS REAC PTs in clinical context |
| T3 Labelling | 9,035 pairs (1:3 YES:NO) | FAERS DRUG × OnSIDES v3.1.1 (6-strategy cascade) |
| T4 Causality | 7,223 pairs | FAERS DRUG/THER/INDI fields (WHO-UMC rules) |
| **Total** | **32,391 train + 3,609 eval** | **70.8 MB** |

Decontamination: 0 hash overlap between train and eval sets ✅

## Implementation

See:
- [`src/data/01_download_faers.py`](../../src/data/01_download_faers.py) — Downloads 29 quarters of FAERS data
- [`src/data/02_preprocess.py`](../../src/data/02_preprocess.py) — DuckDB load, dedup, filter, master join
- [`src/data/04_download_onsides.py`](../../src/data/04_download_onsides.py) — Builds OnSIDES lookup table (T3 ground truth)
- [`src/data/03_build_training_data.py`](../../src/data/03_build_training_data.py) — Generates JSONL training data for all 4 tasks
