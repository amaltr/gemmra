# Problem Statement — Complete Definition

> **This document contains ONLY the problem definition.** No solutions, no technology
> choices, no implementation details. Use this as the immutable "contract" that
> any solution approach must satisfy.

---

## 1. Official Statement

**ID:** `FINETUNING_005`  
**Track:** Fine-Tuning (Track 3 — Advanced Users)  
**Event:** TCS & AMD AI Hackathon 2026

### Exact Official Text (from hackathon use case CSV)

> *"AI led Medical Review Assistant — Assist medical reviewers to assess seriousness,
> suggest MedDRA codes, evaluate labelling status, and support causality assessment."*

### What This Demands

Build a **fine-tuned AI model** that automates **four** distinct pharmacovigilance
review subtasks. The model must be genuinely fine-tuned (not prompted, not RAG,
not agent-based).

---

## 2. Domain Context: Pharmacovigilance

### What Is Pharmacovigilance?

The science of monitoring drug safety **after** market approval. When a doctor,
pharmacist, or patient suspects a drug caused harm, they file a safety report
(called an **ICSR** — Individual Case Safety Report). A human medical reviewer
then manually processes each report through multiple assessment steps.

### The Scale of the Problem

- **2+ million** adverse event reports filed with the FDA per year
- Each report requires **manual assessment** across multiple criteria
- **$8.3B+** global pharmacovigilance industry (→ $15–18B by 2030)
- Regulatory deadlines: **15-day** expedited reporting for serious unlabelled events
- Growing backlog — human reviewer capacity cannot keep up

### The Available Data

**FAERS** (FDA Adverse Event Reporting System) — the world's largest
crowdsourced pharmacovigilance database:

- **20+ million** reports since 2004
- Updated quarterly (and daily since late 2025)
- **Free** and public (US Government data)
- Structured as 7 relational tables connected by `primaryid`
- Ground truth labels for Tasks 1, 2, and 4 are embedded in the schema
- Download: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

---

## 3. The Four Tasks — Precise Definitions

### Task 1 — Seriousness Assessment

| Attribute | Definition |
|-----------|------------|
| **Question** | Is this adverse event medically serious? |
| **Governing standard** | ICH E2A guideline (International Council for Harmonisation) |
| **Classification** | Binary: SERIOUS / NOT SERIOUS |
| **Criteria** | A case is SERIOUS if it results in **any** of: |

| Code | Criterion | Example |
|------|-----------|---------|
| DE | Death | Patient died from the event |
| LT | Life-threatening condition | Event required emergency intervention to prevent death |
| HO | Hospitalization (initial or prolonged) | Patient admitted or stay extended |
| DS | Disability or permanent damage | Persistent loss of function |
| CA | Congenital anomaly / birth defect | Fetal/neonatal effect |
| OT | Other medically significant event | Catch-all for regulatory judgment |

**Ground truth source in FAERS:** `OUTC.outc_cod` — if any of DE/LT/HO/DS/CA
is present → SERIOUS: YES. No OUTC row or OT-only → NOT SERIOUS.

**Evaluation metric:** F1 score (binary classification)

**Required output format:**
```
SERIOUS: YES/NO
Criteria met: [code + name]
Rationale: [explanation referencing ICH E2A criteria]
```

---

### Task 2 — MedDRA Code Suggestion

| Attribute | Definition |
|-----------|------------|
| **Question** | What is the correct MedDRA Preferred Term for this adverse event? |
| **Governing standard** | MedDRA (Medical Dictionary for Regulatory Activities) |
| **Classification** | Multi-class (~80,000 possible PTs) |

**MedDRA hierarchy (5 levels, top → bottom):**
```
SOC  (System Organ Class)           e.g. "Gastrointestinal disorders"
  └── HLGT (High-Level Group Term)  e.g. "Gastrointestinal haemorrhages"
        └── HLT  (High-Level Term)  e.g. "GI and abdominal pains"
              └── PT  (Preferred Term)        ← MODEL MUST PREDICT THIS
                    └── LLT  (Lowest-Level Term)  ← verbatim synonyms
```

**Why it's hard:** 80,000+ terms, 5 hierarchy levels, complex synonym mapping.
"Stomach bleeding" → "Gastrointestinal haemorrhage" requires understanding
both lay language and medical terminology.

**Ground truth source in FAERS:** `REAC.pt` — already coded to MedDRA PT.

**Key constraint:** MedDRA dictionary is **proprietary** (ICH). Full hierarchy
is NOT freely downloadable. Must work with FAERS PTs directly.

**Evaluation metric:** Hierarchical F1 (partial credit if correct HLT but wrong PT)

**Required output format:**
```
MedDRA PT: [Preferred Term]
SOC: [System Organ Class]
Rationale: [why this term maps to the reported event]
```

---

### Task 3 — Labelling Status Evaluation

| Attribute | Definition |
|-----------|------------|
| **Question** | Is this adverse event listed in the drug's approved product label? |
| **Governing standard** | FDA drug labelling regulations |
| **Classification** | Binary: LABELLED / UNLABELLED |
| **Clinical significance** | Unlabelled = "unexpected" → mandatory 15-day expedited report to FDA |

**How to determine labelling status:**
1. Get the drug's name from `DRUG.drugname`
2. Match against OnSIDES database (PubMedBERT-extracted ADEs from FDA labels)
3. OnSIDES provides MedDRA PT-level adverse effects per drug
4. Check if the adverse event PT appears in OnSIDES for that drug

**Ground truth source:** OnSIDES database (v3.1.1) — PubMedBERT NLP extraction
from FDA labels, mapped to MedDRA PTs. Matched via 6-strategy cascade:
NDA → exact name → normalized name → active ingredient → RxNorm → substring.
Fallback: frequency heuristic if OnSIDES unavailable.

**Evaluation metric:** Precision / Recall / AUC

**Required output format:**
```
LABELLED: YES/NO
Label section reference: [which section of the label]
Rationale: [evidence from label text]
```

---

### Task 4 — Causality Assessment

| Attribute | Definition |
|-----------|------------|
| **Question** | How likely did this drug cause this adverse event? |
| **Governing standard** | WHO-UMC causality assessment scale |
| **Classification** | Ordinal: Certain > Probable > Possible > Unlikely > Unclassified |

**WHO-UMC Scale — Full Criteria:**

| Level | Required Evidence |
|-------|-------------------|
| **Certain** | Plausible time relationship + positive dechallenge + **positive rechallenge** + no alternative explanation |
| **Probable** | Plausible time relationship + positive dechallenge + no alternative explanation (rechallenge not required) |
| **Possible** | Reasonable time sequence; could be explained by disease or other drugs |
| **Unlikely** | Temporal relationship improbable; other explanations more likely |
| **Unclassified** | Insufficient information to assess |

**Evidence available in FAERS:**

| Evidence Type | FAERS Field(s) | Signal |
|---------------|----------------|--------|
| Dechallenge | `DRUG.dechal` | Y = drug stopped → AE resolved (strong evidence) |
| Rechallenge | `DRUG.rechal` | Y = drug restarted → AE recurred (strongest evidence) |
| Temporal gap | `THER.start_dt` vs `DEMO.event_dt` | Days between drug start and event |
| Concomitant drugs | Count of `role_cod = 'C'` in DRUG | Each = alternative explanation |
| Indication confound | `INDI.indi_pt` vs `REAC.pt` | If drug indication matches AE → weakens causality |

**Ground truth source:** No explicit causality field in FAERS. Labels must be
**inferred** from evidence fields using a scoring function.

**Evaluation metric:** Cohen's Kappa (weighted)

**Required output format:**
```
WHO-UMC Causality: [level]
Evidence:
(1) Temporal plausibility — [analysis]
(2) Dechallenge — [analysis]
(3) Rechallenge — [analysis]
(4) Mechanism — [analysis]
(5) Alternative explanations — [analysis]
```

---

## 4. FAERS Schema — The Raw Material

All files are `$`-delimited ASCII text, joined on `primaryid`.

| Table | Description | Tasks |
|-------|-------------|-------|
| **DEMO** | Patient demographics + report metadata | All (context) |
| **DRUG** | All drugs in the case (suspect + concomitant) | T3, T4 |
| **REAC** | Adverse reactions (MedDRA coded PTs) | T2 (ground truth) |
| **OUTC** | Case outcomes (DE, LT, HO, DS, CA) | T1 (ground truth) |
| **THER** | Drug therapy dates (start, end, duration) | T4 (temporal) |
| **INDI** | Drug indications (why prescribed) | T4 (confounding) |
| **RPSR** | Report sources (HP, CSM, LIT) | Quality filter |

### Key Relationships

```
DEMO (1 per case)
  ├── DRUG (many — multiple drugs per case)
  │     ├── THER (1 per drug — therapy dates)
  │     └── INDI (1 per drug — indication)
  ├── REAC (many — multiple reactions per case)
  ├── OUTC (many — multiple outcomes per case)
  └── RPSR (1 per case — report source)
```

### Data Quality Considerations

1. **DELETED cases** — FDA retracts cases (duplicates, manufacturer retractions).
   Must be filtered before any processing.
2. **Duplicate cases** — Same `caseid` can appear in multiple versions.
   Keep only max `caseversion` per `caseid`.
3. **Drug name messiness** — Same drug appears as 50+ string variants.
   `WARFARIN`, `warfarin`, `Coumadin`, `warfarin sodium 5mg`, etc.
4. **Missing data** — Many fields are nullable. Temporal dates often missing.
5. **Reporter quality** — HP (health professional) > LIT (literature) > CSM (consumer).
6. **No free-text narratives** — Patient narrative text is stripped for privacy.

---

## 5. Constraints

| Constraint | Detail |
|-----------|--------|
| **GPU** | AMD MI300X — 192 GB HBM3, 5.3 TB/s bandwidth |
| **Data** | Public datasets only (FAERS, OnSIDES, BioDEX) |
| **Track mandate** | Must demonstrate actual fine-tuning (not prompting/RAG) |
| **Model** | Open-source only — no proprietary API calls |

---

## 6. Success Criteria

For **each task**, a successful implementation must demonstrate:

1. **Measurable improvement** over base model (before/after metrics)
2. **Structured, consistent output format** (not free-form text)
3. **Domain-appropriate reasoning** (references ICH E2A, WHO-UMC, etc.)
4. **Working inference** — can process new cases in real-time
5. **Auditable thinking traces** — visible reasoning for regulatory review

Across all tasks:
- Format compliance rate > 95% (structured output)
- Significant metric improvement over zero-shot baseline
- 10–20 second per-case latency on MI300X
- Human-in-the-Loop design (augment reviewers, not replace them)

---

*This document defines WHAT must be built. For HOW to build it, see the
solution documents in `docs/architecture/decisions/` and `src/`.*
