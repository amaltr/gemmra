# Gemmra-Bench v1.0

**A 3,645-sample evaluation benchmark for pharmacovigilance AI systems.**

---

## Overview

Gemmra-Bench is a curated, decontaminated evaluation benchmark for assessing AI systems on four critical pharmacovigilance (PV) case-processing tasks. It is derived entirely from public FDA adverse event data and designed to measure both accuracy and production-readiness (format compliance, reasoning auditability).

| Property | Value |
|----------|-------|
| **Total samples** | 3,645 |
| **Tasks** | 4 (T1–T4) |
| **Data sources** | FDA FAERS, BioDEX, OnSIDES |
| **Decontamination** | Content-hash splitting (MeditronFO protocol) |
| **Metrics** | Task-specific F1, weighted scoring, composite average |
| **License** | Open (derived from public FDA data) |

---

## Tasks

### T1: Seriousness Classification (1,013 samples)
- **Input:** Structured adverse event case (drug, events, outcomes)
- **Output:** `SERIOUS: YES/NO` + ICH E2A criteria met (DE, LT, HO, DS, CA, OT)
- **Metric:** F1 Score (binary classification)
- **Ground truth:** Derived from FAERS `outc_cod` field — algorithmically deterministic
- **Clinical relevance:** ICH E2A/E2D requires seriousness assessment for every case. Serious cases trigger expedited 15-day reporting.

### T2: MedDRA Preferred Term Coding (845 samples)
- **Input:** Clinical narrative describing adverse drug reaction
- **Output:** MedDRA Preferred Term (PT) code
- **Metric:** Hierarchical weighted scoring (novel):
  - Level 1: Exact PT match (weight 1.0)
  - Level 2: Synonym/LLT match (weight 0.9)
  - Level 3: Fuzzy match >80% (weight 0.75)
  - Level 4: System Organ Class (SOC) match (weight 0.5)
- **Ground truth:** BioDEX biomedical dataset extraction — literature-validated PT mappings
- **Clinical relevance:** MedDRA coding is the most challenging PV task. The dictionary contains 80,000+ Preferred Terms across 27 System Organ Classes.

### T3: Drug Labelling Status Assessment (995 samples)
- **Input:** Drug name + adverse event term
- **Output:** `LABELED: YES/NO` (whether the AE appears on the approved drug label)
- **Metric:** F1 Score (binary classification)
- **Ground truth:** OnSIDES database — 2.7M FDA drug label side-effect pairs covering 1,671 unique drug ingredients
- **Clinical relevance:** Unlabelled serious adverse events trigger ICSRs (Individual Case Safety Reports) and potential label updates.

### T4: WHO-UMC Causality Assessment (792 samples)
- **Input:** Structured case with temporal data, dechallenge/rechallenge info
- **Output:** WHO-UMC causality category (Certain, Probable, Possible, Unlikely, Unassessable)
- **Metric:** Weighted scoring (exact match 1.0, adjacent category 0.5)
- **Ground truth:** Algorithmically derived from FAERS temporal and clinical fields using WHO-UMC decision criteria
- **Clinical relevance:** Causality assessment determines whether a drug caused the adverse event — directly impacts regulatory action.

---

## Decontamination Protocol

Train-eval split uses **content-hash decontamination** inspired by the MeditronFO methodology:

1. Each sample is hashed based on its clinical content (drug, events, outcome)
2. Hash determines assignment to train or eval set — deterministic, reproducible
3. Zero content overlap between training (32,355 pairs) and evaluation (3,645 samples)
4. No temporal splitting (FAERS cases span 2019Q1–2026Q1) — hash-based is more robust than date-based for this data

---

## Scoring

### Per-Task Scores
Each task produces a score in [0, 1]. See task descriptions above for metric details.

### Composite Score
Equal-weighted average of all four task scores:

```
Composite = (T1_score + T2_score + T3_score + T4_score) / 4
```

### Format Compliance
Percentage of outputs that parse into the expected structured format. Production PV systems require 100% parseable output — any format failure means manual intervention.

---

## Baseline Results

| Model | T1 | T2 | T3 | T4 | Composite | Format |
|-------|:---:|:---:|:---:|:---:|:---------:|:------:|
| Gemma 4 31B-IT (base)† | 0.977 | 0.311 | 0.782 | 0.845 | 0.729 | N/A |
| **Gemmra (SFT)** | **0.995** | **0.667** | **0.801** | **0.986** | **0.862** | **100%** |
| Gemmra (WiSE-FT α=0.9) | 0.985 | 0.532 | 0.852 | 0.930 | 0.825 | 99% |
| Gemmra (GRPO) | 0.985 | 0.538 | 0.807 | 0.980 | 0.828 | 100% |

> †Base model evaluated with task-specific format prompts and thinking disabled (250 samples per task)

---

## Intended Use

Gemmra-Bench is designed for:
- **Evaluating fine-tuned models** on pharmacovigilance case processing
- **Comparing PV automation approaches** (LLM-based, rule-based, hybrid)
- **Benchmarking format compliance** alongside accuracy (both matter in production)
- **Assessing reasoning quality** via thinking trace analysis

### Limitations
- T2 scoring depends on MedDRA dictionary coverage — models may know correct terms not in our eval vocabulary
- T3 ground truth covers 1,671 drug ingredients — novel drugs may not be represented
- Samples are derived from US FDA data (FAERS) — may not generalize to other regulatory jurisdictions

---

## Reproducibility

```bash
# Generate evaluation data
python src/data/03_build_training_data.py

# Run evaluation
python src/eval/evaluate.py --checkpoint checkpoints/sft

# Results are saved to eval_results/
```

**Evaluation code:** `src/eval/evaluate.py`
**Scoring functions:** Task-specific scoring with hierarchical MedDRA matching

---

## Citation

```bibtex
@misc{gemmra-bench-2026,
  title={Gemmra-Bench: A Pharmacovigilance Evaluation Benchmark},
  author={Jha, Bhaskar and T R, Amal},
  year={2026},
  howpublished={\url{https://github.com/bhaskarjha-dev/gemmra}},
  note={3,645 decontaminated samples across 4 PV tasks}
}
```
