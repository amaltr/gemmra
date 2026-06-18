# Worked Examples — Walk Through These Before Building

**Purpose:** Concrete examples of all 4 tasks so you understand exactly
what the model's input and output looks like.

> **Format:** Gemma 4 native thinking tokens: `<|channel>thought ... <channel|>`
> All 4 tasks use thinking mode for visible reasoning traces (ADR-006).

---

## Example Case: The Warfarin Patient

**Real-world scenario:** 68-year-old woman on Warfarin (blood thinner) for
atrial fibrillation. After 6 months of treatment, she develops
gastrointestinal bleeding. She is hospitalized. The drug is stopped,
and the bleeding resolves within 5 days.

### FAERS Data Fields for This Case

| Table | Field | Value |
|-------|-------|-------|
| DEMO | age | 68 |
| DEMO | age_cod | YR |
| DEMO | gndr_cod | F |
| DEMO | event_dt | 20240615 |
| DRUG | drugname | WARFARIN SODIUM |
| DRUG | role_cod | PS (Primary Suspect) |
| DRUG | dechal | Y (drug stopped → AE resolved) |
| DRUG | rechal | N (drug not restarted) |
| DRUG | nda_num | 009218 |
| REAC | pt | Gastrointestinal haemorrhage |
| OUTC | outc_cod | HO (Hospitalization) |
| THER | start_dt | 20231215 |
| INDI | indi_pt | Atrial fibrillation |

---

### Task 1 Output: Seriousness Assessment

```
<|channel>thought
Evaluating against ICH E2A seriousness criteria.
The reported outcome codes include: HO (Hospitalization).
Matching criteria: HO — the patient required inpatient hospital
admission for gastrointestinal haemorrhage.
At least one ICH E2A criterion is satisfied.
<channel|>
SERIOUS: YES
Criteria met: HO (Hospitalization)
Rationale: The case meets ICH E2A seriousness definition based on
the reported outcome of hospitalization for gastrointestinal haemorrhage.
```

> **Note:** Task 1 uses thinking=ON. Even though it's a binary classification,
> ICH E2A regulators require documented reasoning for seriousness assessments.
> The thinking trace shows the model evaluated all criteria systematically.

### Task 2 Output: MedDRA Coding

```
<|channel>thought
The patient reported gastrointestinal bleeding while on Warfarin.
In the MedDRA hierarchy, this maps to:
- LLT: Stomach bleeding, GI bleed (lay terms)
- PT: Gastrointestinal haemorrhage (the Preferred Term)
- HLT: Gastrointestinal haemorrhages
- SOC: Gastrointestinal disorders
The term "Gastrointestinal haemorrhage" accurately captures the
clinical event described.
<channel|>
MedDRA PT: Gastrointestinal haemorrhage
SOC: Gastrointestinal disorders
Rationale: Based on the clinical presentation of GI bleeding in a
patient on anticoagulant therapy, "Gastrointestinal haemorrhage" is
the appropriate MedDRA Preferred Term.
```

### Task 3 Output: Labelling Status

```
<|channel>thought
Checking whether 'Gastrointestinal haemorrhage' is listed in the
approved label for Warfarin (NDA 009218).
Reviewing the drug's product label sections.
Gastrointestinal haemorrhage is documented in the Adverse Reactions
and Warnings and Precautions sections.
This is a well-known, expected adverse reaction for anticoagulants.
<channel|>
LABELLED: YES
Drug: WARFARIN SODIUM
Adverse event: Gastrointestinal haemorrhage
Label section: Adverse Reactions, Warnings and Precautions
Rationale: Gastrointestinal haemorrhage is extensively documented in
the Warfarin label as a known, common adverse reaction associated with
anticoagulant therapy.
```

> **Note:** Task 3 uses thinking=ON. The reasoning trace shows the model
> checked the drug label systematically before making a determination.

### Task 4 Output: Causality Assessment

```
<|channel>thought
Applying WHO-UMC causality criteria:
  - Temporal plausibility: event occurred 183 days after drug start —
    within clinically plausible window for anticoagulant-related bleeding
  - Dechallenge: Positive — drug stopped, GI bleeding resolved within 5 days
  - Rechallenge: Not available — drug was not restarted
  - Confounding: No indication-AE overlap (atrial fibrillation ≠ GI bleed)
  - Known mechanism: Warfarin inhibits vitamin K-dependent clotting factors,
    directly increasing bleeding risk
  - Concomitant drugs: No alternative explanations documented
Assessment: Dechallenge positive + temporal plausibility + known mechanism
+ no better alternative = Probable (not Certain because no rechallenge).
<channel|>
WHO-UMC Causality: Probable
Evidence:
  - Temporal plausibility: event 183 days after drug start
  - Dechallenge: Positive — drug stopped, AE resolved
  - Rechallenge: Not available
  - Confounding: No indication-AE overlap
  - Known mechanism: Anticoagulant-related bleeding risk
```

---

## 5 Pre-Loaded Demo Cases

These are the cases to pre-load in the Streamlit demo. Each demonstrates
different aspects of the model's capability.

### Case A — Warfarin / GI Bleed (Standard)
- **Drug:** Warfarin | **AE:** Gastrointestinal haemorrhage
- **T1:** SERIOUS (HO) | **T4:** Probable
- **Demo purpose:** Standard reference case, most intuitive

### Case B — Ibuprofen / Mild Rash (Non-Serious)
- **Drug:** Ibuprofen | **AE:** Rash
- **T1:** NOT SERIOUS (no qualifying outcomes)
- **T4:** Possible (temporal fit, but common AE)
- **Demo purpose:** Shows model correctly identifies NON-serious cases

### Case C — Methotrexate / Liver Failure (Strongest Case)
- **Drug:** Methotrexate | **AE:** Hepatic failure
- **T1:** SERIOUS (LT — Life-threatening)
- **T4:** Certain (rechallenge positive + dechallenge + temporal fit)
- **Demo purpose:** Most severe case — shows strongest causality

### Case D — Atorvastatin / Muscle Pain (MedDRA Challenge)
- **Drug:** Atorvastatin (statin) | **AE:** Myalgia
- **T1:** NOT SERIOUS
- **T4:** Possible
- **Demo purpose:** Tests MedDRA coding — "muscle ache" → "Myalgia" PT

### Case E — Infliximab / Infusion Reaction (Labelling Demo)
- **Drug:** Infliximab (biologic) | **AE:** Infusion related reaction
- **T1:** SERIOUS (HO)
- **T3:** LABELLED: YES (well-known in biologic labels)
- **Demo purpose:** Shows Task 3 labelling lookup capability
