# Pharmacovigilance 101 — Beginner's Guide

**Purpose:** Everything you need to know about pharmacovigilance to build this project.

---

## What is Pharmacovigilance?

**Pharmacovigilance (PV)** = the science of detecting, assessing, understanding, and preventing adverse effects of medicines.

**In simple terms:** When someone takes a drug and something bad happens, there's a whole system for tracking, reporting, and analyzing those events. That's pharmacovigilance.

## The Process (How It Works)

```
Patient takes drug → Something bad happens → Doctor/patient reports it
                                                        ↓
                                               Safety Report (ICSR)
                                                        ↓
                                         Medical Reviewer assesses it
                                         ├── Is it serious?         ← Task 1
                                         ├── What's the medical code? ← Task 2
                                         ├── Was it already known?   ← Task 3
                                         └── Did the drug cause it?  ← Task 4
                                                        ↓
                                         Regulatory submission (FDA/EMA)
```

## Key Terms Dictionary

| Term | What It Means | Why You Care |
|------|--------------|-------------|
| **Adverse Event (AE)** | Any unwanted medical thing that happens while on a drug | The core unit of pharmacovigilance |
| **Adverse Drug Reaction (ADR)** | An AE where the drug is suspected to be the cause | Subset of AEs — causal link suspected |
| **ICSR** | Individual Case Safety Report — one patient's report | Each ICSR = one training example |
| **FAERS** | FDA Adverse Event Reporting System — US database | Your training data (20M+ cases) |
| **Serious** | ICH E2A definition: Death, Life-threatening, Hospitalization, Disability, Congenital anomaly | Task 1 output |
| **MedDRA** | Medical Dictionary for Regulatory Activities — 80,000 coded terms | Task 2 target vocabulary |
| **Preferred Term (PT)** | A specific MedDRA code (e.g., "Gastrointestinal haemorrhage") | What Task 2 predicts |
| **System Organ Class (SOC)** | Top-level MedDRA category (e.g., "Gastrointestinal disorders") | Hierarchical grouping |
| **Labelled** | Side effect is already listed on the drug's official label | Task 3 output |
| **Unlabelled** | Side effect is NOT on the label = "unexpected" = urgent report needed | Key regulatory concept |
| **WHO-UMC** | World Health Organization causality assessment scale | Task 4 framework |
| **Dechallenge** | Drug stopped → did the problem go away? | Key causality evidence |
| **Rechallenge** | Drug restarted → did the problem come back? | Strongest causality evidence |
| **NDA** | New Drug Application — FDA approval number | Links drug to its label |
| **DailyMed** | FDA database of drug labels (searchable) | Cross-reference for Task 3 |

## Task 1: Seriousness Assessment

### The Rule (ICH E2A)

An adverse event is **SERIOUS** if it results in ANY of these:

| Code | Criterion | Example |
|------|-----------|---------|
| **DE** | Death | Patient died |
| **LT** | Life-threatening | Patient nearly died |
| **HO** | Hospitalization | Patient admitted to hospital |
| **DS** | Disability | Patient permanently disabled |
| **CA** | Congenital anomaly | Birth defect in child |
| **OT** | Other medically significant | Requires intervention to prevent above |

If NONE of these → **NOT SERIOUS**

### In FAERS Data
The `OUTC` table contains outcome codes per case. Check if any code matches `{DE, LT, HO, DS, CA}`.

## Task 2: MedDRA Code Suggestion

### The Problem
Patients describe their symptoms in everyday language. But regulators need standardized codes.

| Patient Says | MedDRA PT (What We Code) |
|-------------|--------------------------|
| "My stomach was bleeding" | Gastrointestinal haemorrhage |
| "I felt dizzy all the time" | Dizziness |
| "My liver was failing" | Hepatic failure |
| "I had a bad rash" | Rash |
| "Heart was racing" | Tachycardia |

### MedDRA Hierarchy
```
SOC (System Organ Class) — 27 top-level categories
  └── HLGT (High Level Group Term)
      └── HLT (High Level Term)
          └── PT (Preferred Term) ← THIS IS WHAT WE PREDICT
              └── LLT (Lowest Level Term) — synonyms
```

### In FAERS Data
The `REAC` table already has the `pt` (Preferred Term) — this IS your ground truth label.

## Task 3: Labelling Status

### The Concept
Every approved drug has an official "label" (package insert) that lists known side effects.

- **Labelled:** The side effect IS listed → expected
- **Unlabelled:** The side effect IS NOT listed → unexpected → 15-DAY EXPEDITED REPORT

### How to Check
1. Get the drug's NDA number from FAERS `DRUG.nda_num`
2. Query DailyMed API: NDA → setid → download SPL XML
3. Parse the "Adverse Reactions" section (LOINC code 34084-4)
4. Check if the MedDRA PT appears in that section

## Task 4: Causality Assessment (WHO-UMC)

### The Scale

| Level | Meaning | Requirements |
|-------|---------|-------------|
| **Certain** | Drug definitely caused it | Rechallenge positive + dechallenge + temporal fit |
| **Probable** | Drug very likely caused it | Dechallenge positive + temporal fit + no better explanation |
| **Possible** | Drug might have caused it | Temporal fit, but alternative causes exist |
| **Unlikely** | Drug probably didn't cause it | Poor temporal fit OR better alternative explanation |
| **Unclassified** | Can't determine | Insufficient data |

### Evidence Types

| Evidence | What It Means | Strength |
|----------|---------------|----------|
| **Temporal plausibility** | AE happened after drug started (within reasonable window) | Basic requirement |
| **Positive dechallenge** | Drug stopped → AE resolved | Strong evidence |
| **Positive rechallenge** | Drug restarted → AE recurred | Strongest evidence |
| **No confounding** | No other drugs or conditions explain it | Strengthens case |
| **Known mechanism** | Drug's pharmacology explains the AE | Supporting evidence |
| **Indication confound** | The disease being treated has same symptoms as AE | Weakens case |

### In FAERS Data
- `DRUG.dechal` = dechallenge (Y/N)
- `DRUG.rechal` = rechallenge (Y/N)
- `THER.start_dt` / `event_dt` = temporal gap
- `DRUG.role_cod` = 'C' for concomitant drugs (alternative explanations)
- `INDI.indi_pt` = indication (check for confounding)
