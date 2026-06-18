# FAERS Data Schema Reference

**Purpose:** Complete reference for all FAERS data tables and their fields.

---

## Tables Overview

```
DEMO ──→ DRUG ──→ REAC ──→ OUTC ──→ THER ──→ INDI ──→ RPSR
  │         │        │        │        │        │        │
  └─────────┴────────┴────────┴────────┴────────┴────────┘
                    Linked by: primaryid
```

## File Format
- **Delimiter:** `$` (dollar sign)
- **Encoding:** ASCII
- **Header:** First row is column names
- **Naming:** `{TABLE}{YY}Q{Q}.txt` (e.g., `DEMO24Q3.txt`)

## DEMO (Demographics & Report Info)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Unique report identifier | JOIN KEY |
| `caseid` | INT | Case identifier (may have multiple versions) | Dedup key |
| `caseversion` | INT | Version number (keep max per caseid) | Dedup |
| `i_f_code` | CHAR | Initial (I) or Follow-up (F) | Quality |
| `event_dt` | CHAR | Date of adverse event (YYYYMMDD) | Task 4 (temporal) |
| `mfr_dt` | CHAR | Manufacturer received date | — |
| `fda_dt` | CHAR | FDA received date | — |
| `rept_cod` | CHAR | Report code: EXP=expedited, PER=periodic, DIR=direct | Quality filter |
| `age` | NUM | Patient age | Narrative |
| `age_cod` | CHAR | Age unit: YR, MON, WK, DY, HR | Narrative |
| `gndr_cod` | CHAR | Gender: M, F, UNK | Narrative |
| `wt` | NUM | Weight | — |
| `wt_cod` | CHAR | Weight unit: KG, LBS | — |
| `occp_cod` | CHAR | Reporter type: MD, PH, HP, CN, LW, OT | Quality |
| `reporter_country` | CHAR | Country code | — |

## DRUG (Drug Information)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `drug_seq` | INT | Drug sequence in case | Link to THER/INDI |
| `role_cod` | CHAR | **PS**=Primary Suspect, **SS**=Secondary, **C**=Concomitant, **I**=Interacting | Filter (PS for main drug) |
| `drugname` | CHAR | Drug name (messy — needs normalization) | Narrative |
| `prod_ai` | CHAR | Active ingredient | Normalization |
| `val_vbm` | INT | Valid drug name flag | Quality |
| `route` | CHAR | Route of administration | — |
| `dose_vbm` | CHAR | Dose info | — |
| `dechal` | CHAR | Dechallenge: Y, N, U, D | **Task 4 (causality)** |
| `rechal` | CHAR | Rechallenge: Y, N, U, D | **Task 4 (causality)** |
| `nda_num` | CHAR | NDA application number | **Task 3 (label lookup)** |

## REAC (Reactions / Adverse Events)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `pt` | CHAR | **MedDRA Preferred Term** | **Task 2 GROUND TRUTH** |
| `drug_rec_act` | CHAR | Drug-reaction activity | — |

## OUTC (Patient Outcomes)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `outc_cod` | CHAR | Outcome: DE, LT, HO, DS, CA, RI, OT | **Task 1 GROUND TRUTH** |

**Outcome codes:**
- `DE` = Death
- `LT` = Life-threatening
- `HO` = Hospitalization (initial or prolonged)
- `DS` = Disability
- `CA` = Congenital anomaly
- `RI` = Required intervention
- `OT` = Other serious

## THER (Therapy Dates)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `drug_seq` | INT | Links to DRUG table | JOIN KEY |
| `start_dt` | CHAR | Therapy start date | **Task 4 (temporal gap)** |
| `end_dt` | CHAR | Therapy end date | Task 4 |
| `dur` | NUM | Duration | — |
| `dur_cod` | CHAR | Duration unit | — |

## INDI (Drug Indications)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `drug_seq` | INT | Links to DRUG table | JOIN KEY |
| `indi_pt` | CHAR | Indication (MedDRA PT) | **Task 4 (confound check)** |

## RPSR (Report Sources)

| Column | Type | Description | Used For |
|--------|------|-------------|----------|
| `primaryid` | INT | Report link | JOIN KEY |
| `rpsr_cod` | CHAR | Source: HP=Healthcare Professional, CN=Consumer, LIT=Literature, FGN=Foreign | Quality filter |

## Key Relationships

```sql
-- Join all tables for a complete case view
SELECT d.*, dr.*, r.*, o.*, t.*, i.*, rp.*
FROM demo d
LEFT JOIN drug dr ON d.primaryid = dr.primaryid
LEFT JOIN reac r ON d.primaryid = r.primaryid
LEFT JOIN outc o ON d.primaryid = o.primaryid
LEFT JOIN ther t ON d.primaryid = t.primaryid AND dr.drug_seq = t.drug_seq
LEFT JOIN indi i ON d.primaryid = i.primaryid AND dr.drug_seq = i.drug_seq
LEFT JOIN rpsr rp ON d.primaryid = rp.primaryid
WHERE dr.role_cod = 'PS'  -- Primary suspect drug only
```
