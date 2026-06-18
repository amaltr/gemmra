# Data Sources Research

**Last Updated:** June 9, 2026  
**Researcher:** AI Assistant (web-verified)

---

## Primary: FAERS (FDA Adverse Event Reporting System)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| **Cost** | Free (US Government public data) |
| **Format** | ASCII, `$`-delimited, 7 tables per quarter |
| **Size** | ~200MB zipped per quarter |
| **Coverage** | 20M+ adverse event reports since 2004 |
| **Our selection** | 2024 Q3 + Q4 (most recent complete) |
| **Update frequency** | Now daily (since late 2025); quarterly archives still available |
| **Key limitation** | NO free-text patient narratives (stripped for privacy) |

### FAERS Tables Used
See [`docs/domain/faers_schema.md`](domain/faers_schema.md) for full schema.

| Table | Key Fields | Tasks |
|-------|-----------|-------|
| DEMO | age, gender, event_dt, rept_cod | All |
| DRUG | drugname, nda_num, dechal, rechal, role_cod | T3, T4 |
| REAC | pt (MedDRA Preferred Term) | T2 (ground truth) |
| OUTC | outc_cod (DE, LT, HO, DS, CA) | T1 (ground truth) |
| THER | start_dt, end_dt | T4 (temporal gap) |
| INDI | indi_pt | T4 (confounding) |
| RPSR | rpsr_cod | Quality filtering |

---

## Secondary: CADECv2 (CSIRO Adverse Drug Event Corpus)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://data.csiro.au/collection/csiro:62387 |
| **Released** | November 2024 (use v2, not original CADEC) |
| **Content** | Real patient forum posts (AskAPatient.com) annotated with MedDRA |
| **Size** | ~1,250 posts, 7,101+ annotations |
| **Cost** | Free for research |
| **Key value** | Lay-language → MedDRA PT mapping (exactly what T2 needs) |
| **Citation required** | Yes — see CSIRO data portal |

### Why CADECv2 over original CADEC?
- More diverse drug coverage
- Part of MultiADE benchmark (more credible)
- Released Nov 2024 (more recent)

---

## Auxiliary: DailyMed REST API

| Attribute | Detail |
|-----------|--------|
| **API Base** | `https://dailymed.nlm.nih.gov/dailymed/services/v2/` |
| **Cost** | Free, no API key |
| **Purpose** | Look up drug labels for Task 3 (labelling status) |
| **Workflow** | NDA number → setid → SPL XML → parse section 34084-4 |
| **LOINC for Adverse Reactions** | `34084-4` |

### API Endpoints
```
GET /applicationnumbers.xml?application_number={NDA}    → Get setid
GET /spls/{setid}.xml                                   → Get full label XML
```

---

## Auxiliary: RxNorm API

| Attribute | Detail |
|-----------|--------|
| **API Base** | `https://rxnav.nlm.nih.gov/REST/` |
| **Cost** | Free, no API key (40 req/min) |
| **Purpose** | Normalize messy FAERS drug names |
| **Key endpoint** | `getApproximateMatch?term={drug_name}` |

### Alternative: openFDA API
- **URL:** `https://api.fda.gov/drug/`
- Free, no key
- Can look up NDA numbers, labels, adverse events

---

## ⚠️ NOT Available: MedDRA Full Hierarchy

| Attribute | Detail |
|-----------|--------|
| **Status** | PROPRIETARY (ICH) |
| **Access** | Subscription required (free for academic/regulatory) |
| **Workaround** | Use FAERS REAC.pt as ground truth; CADECv2 for lay-language mapping |

---

## 🆕 Newly Discovered Datasets (June 9, 2026 Research)

### PHEE — Pharmacovigilance Event Extraction Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | ACL Anthology |
| **Content** | 5,000+ annotated pharmacovigilance events from medical case reports |
| **Annotations** | Hierarchical: demographics, treatments, adverse effects |
| **Format** | Structured event schema |
| **Cost** | Free (research) |
| **Key value** | Real clinical case reports with expert annotations — much richer than FAERS structured fields |
| **Use case** | Supplementary training data for T1, T2, T4 |

### BioDEX — Drug Safety Monitoring Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | HuggingFace / GitHub |
| **Content** | PubMed full-text papers bundled with structured drug safety reports |
| **Tasks** | Report extraction + reaction extraction |
| **Cost** | Free |
| **Key value** | Real biomedical text → structured safety output. Directly relevant to our pipeline. |
| **Use case** | Additional training examples, especially for T2 (MedDRA) and T3 (labelling) |

### MultiADE — Multi-Domain ADE Benchmark

| Attribute | Detail |
|-----------|--------|
| **Source** | arXiv |
| **Content** | Aggregated benchmark: n2c2, MADE, PHEE, PsyTAR, CADEC |
| **Cost** | Free |
| **Key value** | Standardized evaluation benchmark — cite this for credibility |
| **Use case** | Evaluation benchmark (not training) |

### openFDA JSON API

| Attribute | Detail |
|-----------|--------|
| **URL** | `https://api.fda.gov/drug/event.json` |
| **Content** | FAERS in machine-readable JSON format, annotated by FDA |
| **Cost** | Free, no key (rate limited) |
| **Key value** | Cleaner than raw FAERS files; pre-processed by FDA |
| **Use case** | Alternative to downloading raw FAERS ZIPs (faster pipeline start) |

### Kaggle Cleaned FAERS

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle (multiple community datasets) |
| **Content** | Pre-deduplicated, cleaned FAERS data covering recent years |
| **Cost** | Free |
| **Key value** | Saves pipeline time — someone already cleaned and deduped the data |
| **Use case** | Faster start if FDA download is slow |

---

## Data Volume Estimates (Updated)

| Source | Raw Records | After Filtering | Training Pairs |
|--------|------------|-----------------|----------------|
| FAERS Q3+Q4 | ~1M cases | ~200K (HP/LIT + dedup) | ~18K (T1+T2+T4) |
| CADECv2 | ~1,250 posts | ~1,000 | ~1,000 (T2) |
| DailyMed | per-drug lookup | ~100 top drugs | ~2,000 (T3) |
| PHEE | ~5,000 events | ~4,000 | ~2,000 (T1+T2+T4) |
| BioDEX | varies | varies | ~1,000 (T2+T3) |
| **Total** | | | **~24,000 pairs** |
