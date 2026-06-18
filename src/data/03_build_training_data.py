"""
Step 3: Build training data (JSONL) from preprocessed FAERS + external datasets.
Run this on CPU (no GPU needed).

GROUND-ZERO REBUILD — All 4 tasks redesigned to be genuinely challenging:

T1 Seriousness: Model must INFER seriousness from clinical narrative
   (outcome codes REMOVED from input — no more trivial IF-check)

T2 MedDRA Coding: Model maps lay/clinical language → MedDRA PT
   (integrates CADECv2-style, PHEE, BioDEX, ADE Corpus data)

T3 Labelling: Model uses drug class + mechanism to REASON about label status
   (not just memorize OnSIDES lookup table)

T4 Causality: Model EXTRACTS evidence from clinical narrative
   (not template-fill from structured fields)

External data integration:
  - PHEE: clinical case report sentences with ADE annotations
  - BioDEX: PubMed paper → drug reaction extraction
  - ADE Corpus v2: sentence-level drug-ADE relations
  - Drug class map: pharmacological class + mechanism for 70+ drugs

Quality assurance (MeditronFO-adopted):
  - 10% decontaminated eval holdout (hash-verified zero overlap)
  - Gemma 4 native thinking tokens for all tasks
"""

import pandas as pd
import numpy as np
import json
import random
import hashlib
import re
from pathlib import Path
from collections import Counter
import sys
import os

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROCESSED_DIR = Path("data/processed")
EXTERNAL_DIR = Path("data/external")
OUTPUT_FILE = PROCESSED_DIR / "training_data.jsonl"
EVAL_FILE = PROCESSED_DIR / "eval_data.jsonl"
DECONTAM_LOG = PROCESSED_DIR / "decontamination_log.txt"
SEED = 42
EVAL_HOLDOUT_RATIO = 0.10


def _safe_str(val, default='Unknown'):
    """Convert pandas value to string, handling NaN/None."""
    if val is None:
        return default
    s = str(val)
    if s.lower() in ('nan', 'none', '', 'nat'):
        return default
    return s


def _age_str(row):
    """Build a clean age string like '51' or 'unknown-age', accounting for age units."""
    age = row.get('age', None)
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return 'unknown-age'
    s = str(age).strip()
    if s.lower() in ('nan', '', 'none', 'nat'):
        return 'unknown-age'
    # Convert to numeric
    try:
        age_val = float(s)
    except (ValueError, OverflowError):
        return s
    
    # Handle age_cod units (FAERS: YR, MON, DEC, WK, DY, HR)
    age_cod = str(row.get('age_cod', 'YR')).strip().upper()
    if age_cod in ('', 'NAN', 'NONE', 'NAT'):
        age_cod = 'YR'
    
    if age_cod == 'MON':
        if age_val < 12:
            return f"{int(age_val)}-month-old"  # e.g. "6-month-old"
        else:
            return str(int(age_val / 12))  # Convert to years
    elif age_cod == 'DEC':
        return str(int(age_val * 10))  # Decades to years
    elif age_cod == 'WK':
        if age_val < 52:
            return f"{int(age_val)}-week-old"
        else:
            return str(int(age_val / 52))
    elif age_cod == 'DY':
        if age_val < 365:
            return f"{int(age_val)}-day-old"
        else:
            return str(int(age_val / 365))
    elif age_cod == 'HR':
        return f"{int(age_val)}-hour-old"
    else:  # YR or unknown
        return str(int(age_val))


# ============================================================
# TASK 1: Seriousness Assessment — NARRATIVE-BASED (no outcome codes in input)
# ============================================================

SERIOUS_CODES = {'DE', 'LT', 'HO', 'DS', 'CA'}
SERIOUS_NAMES = {
    'DE': 'Death', 'LT': 'Life-threatening',
    'HO': 'Hospitalization', 'DS': 'Disability',
    'CA': 'Congenital anomaly', 'RI': 'Required intervention',
    'OT': 'Other medically significant'
}

# Outcome code → natural language narrative variants (model must INFER from these)
# H1 FIX: 12-15 templates per outcome for diversity. Includes ambiguous edge cases.
OUTCOME_NARRATIVES = {
    'DE': [
        "The patient subsequently died.",
        "The adverse event resulted in the patient's death.",
        "The patient passed away following this adverse event.",
        "A fatal outcome was reported.",
        "The patient did not survive the clinical episode.",
        "The patient expired during treatment.",
        "Death occurred during the course of the adverse event.",
        "The treating physician confirmed the patient's death following the event.",
        "The clinical team reported a fatal outcome.",
        "The patient was pronounced dead after the event.",
        "Despite resuscitation efforts, the patient succumbed to the condition.",
        "The patient's death was attributed to the reported adverse event.",
    ],
    'LT': [
        "The patient's condition was assessed as life-threatening.",
        "The event was considered life-threatening by the reporting physician.",
        "The adverse event placed the patient in immediate danger of death.",
        "A life-threatening situation developed.",
        "The patient was transferred to the ICU due to the severity of the event.",
        "Emergency resuscitation measures were initiated.",
        "The clinical team assessed the situation as imminently life-threatening.",
        "The patient required emergency intervention to prevent death.",
        "Vital signs were critically unstable following the event.",
        "The event was classified as presenting an immediate threat to the patient's life.",
        "The attending physician documented immediate danger of death.",
        "Cardiopulmonary compromise was noted.",
    ],
    'HO': [
        "The patient required hospitalization.",
        "The patient was admitted to the hospital for treatment.",
        "The adverse event led to an emergency hospital admission.",
        "Inpatient hospitalization was required.",
        "The patient's hospital stay was prolonged due to the event.",
        "The patient was admitted to the emergency department and subsequently hospitalized.",
        "The event necessitated an overnight hospital stay.",
        "The patient's existing hospitalization was prolonged as a consequence of the event.",
        "Emergency department presentation led to hospital admission.",
        "The patient required inpatient observation for several days.",
        "Hospital admission was arranged following the clinical deterioration.",
        "The patient was kept in the hospital for monitoring after the event.",
        "An unplanned hospital admission resulted from the adverse event.",
        "The severity of the event required the patient to be hospitalized.",
        "The event led to extended medical care in an inpatient setting.",
    ],
    'DS': [
        "The patient experienced persistent or significant disability.",
        "The event resulted in substantial functional impairment.",
        "The adverse event caused lasting disability.",
        "Significant incapacity was reported following the event.",
        "The patient suffered permanent functional limitation.",
        "The event left the patient with a substantial disruption of daily activities.",
        "Long-term disability resulted from the adverse event.",
        "The patient's ability to carry out normal activities was significantly impaired.",
        "The event resulted in loss of an important body function.",
        "Permanent impairment was documented as a consequence of the event.",
        "The patient required ongoing assistance with daily living after the event.",
        "A chronic functional deficit was attributed to the adverse event.",
    ],
    'CA': [
        "A congenital anomaly was reported in association with this event.",
        "The event was associated with a birth defect.",
        "A congenital malformation was identified.",
        "Exposure during pregnancy was reported, and a birth defect was documented.",
        "The infant was born with a congenital anomaly potentially linked to the drug exposure.",
        "A developmental abnormality was identified in the offspring.",
        "The pregnancy resulted in a child with a congenital defect.",
        "Prenatal drug exposure was followed by identification of a birth defect.",
        "A structural malformation was reported in the newborn.",
        "The newborn presented with a congenital abnormality.",
    ],
}

# For non-serious cases — describe outcomes that DON'T meet ICH E2A criteria
# H1 FIX: More diverse templates including edge/ambiguous cases
NON_SERIOUS_NARRATIVES = [
    "The patient recovered without any lasting effects.",
    "The adverse event resolved on its own without medical intervention.",
    "The symptoms were mild and self-limiting.",
    "The patient reported improvement after symptomatic treatment.",
    "The event was considered medically non-significant by the attending physician.",
    "The patient continued treatment and the symptoms subsided.",
    "No hospitalization, disability, or life-threatening conditions were reported.",
    "The adverse event was transient and resolved completely.",
    # Ambiguous / edge cases (still non-serious per ICH E2A)
    "The patient visited the emergency room but was not admitted.",
    "Outpatient treatment was sufficient to manage the adverse event.",
    "The patient required a brief observation period but was discharged the same day.",
    "The event caused temporary discomfort but no lasting impairment.",
    "The patient experienced a noticeable but self-resolving reaction.",
    "Follow-up showed complete resolution of all symptoms.",
    "The patient required dose adjustment but no hospitalization.",
    "The event was documented as non-serious by the reporting healthcare professional.",
]


# ============================================================
# COMBINATORIAL DIVERSITY ENGINE
# Generates near-infinite unique completions by assembling
# randomly-selected phrase fragments with synonym substitution.
# No teacher LLM needed — runs in milliseconds.
# ============================================================

VOCAB = {
    'assess': ['assess', 'evaluate', 'examine', 'analyze', 'review', 'appraise'],
    'Assess': ['Assess', 'Evaluate', 'Examine', 'Analyze', 'Review', 'Appraise'],
    'indicates': ['indicates', 'suggests', 'points to', 'is consistent with', 'demonstrates', 'reveals'],
    'criteria': ['criteria', 'classification standards', 'regulatory thresholds', 'defined criteria'],
    'resolved': ['resolved', 'improved', 'subsided', 'abated', 'cleared', 'remitted'],
    'developed': ['developed', 'experienced', 'presented with', 'manifested', 'reported', 'exhibited'],
    'documented': ['documented', 'recorded', 'noted', 'reported', 'listed'],
    'recognized': ['recognized', 'established', 'well-known', 'acknowledged', 'documented'],
    'corresponds': ['corresponds', 'maps', 'aligns', 'matches', 'translates'],
    'reviewing': ['reviewing', 'examining', 'checking', 'inspecting', 'surveying', 'consulting'],
    'suggests_v': ['suggests', 'indicates', 'implies', 'points toward', 'is suggestive of'],
    'supports': ['supports', 'strengthens', 'reinforces', 'bolsters', 'corroborates'],
    'appears': ['appears', 'seems', 'is found', 'is present', 'is identified'],
    'assessed': ['assessed', 'evaluated', 'examined', 'analyzed', 'reviewed'],
    'support': ['support', 'strengthen', 'reinforce', 'bolster', 'corroborate'],
}


def _diversify(template: str) -> str:
    """Replace {vocab_key} markers with random synonyms from VOCAB."""
    for key, synonyms in VOCAB.items():
        marker = '{' + key + '}'
        while marker in template:
            template = template.replace(marker, random.choice(synonyms), 1)
    return template


# ---- T1 THINKING TRACE BANKS ----

T1_THINK_OPENERS = [
    "Let me {assess} this case against ICH E2A seriousness {criteria}.",
    "{Assess}ing the clinical presentation for seriousness indicators.",
    "I need to determine if any ICH E2A seriousness {criteria} are met.",
    "Checking this adverse event report against the five seriousness categories.",
    "The key question: does this case meet any serious outcome threshold?",
    "Step 1: Identify outcome indicators. Step 2: Map to ICH E2A categories.",
    "Looking at the clinical narrative for death, hospitalization, disability, life-threatening, or congenital anomaly.",
    "{Assess}ing whether the reported outcome satisfies ICH E2A serious {criteria}.",
    "Reviewing the case details to classify seriousness per regulatory guidelines.",
    "Applying ICH E2A seriousness classification to this pharmacovigilance report.",
    "Clinical seriousness assessment: checking five standard categories.",
    "Reading through the case narrative to identify any seriousness triggers.",
    "Beginning seriousness classification under ICH E2A framework.",
    "Determining seriousness level by checking each ICH E2A criterion.",
    "First, I need to identify what clinical outcome occurred and whether it meets regulatory seriousness {criteria}.",
]

T1_THINK_EVIDENCE_YES = {
    'DE': [
        "The narrative {indicates} a fatal outcome. Death is the most severe ICH E2A criterion.",
        "Patient death is confirmed in the report. This automatically classifies as serious.",
        "The clinical record describes a fatal event — this meets the death criterion (DE).",
        "A lethal outcome was reported. Under ICH E2A, death makes this unambiguously serious.",
        "The patient did not survive. Death satisfies the highest severity ICH E2A criterion.",
        "Fatal outcome {documented}. This is the clearest seriousness indicator.",
    ],
    'LT': [
        "The event was assessed as life-threatening, satisfying the LT criterion.",
        "Life-threatening severity is {documented}, meeting ICH E2A {criteria} for serious classification.",
        "The clinical situation posed an immediate danger of death — life-threatening (LT) criterion met.",
        "Emergency intervention was required to prevent death, indicating life-threatening severity.",
        "The report describes an immediately dangerous situation, consistent with LT classification.",
        "Vital instability or imminent risk of death is described, satisfying the life-threatening criterion.",
    ],
    'HO': [
        "Hospitalization was required, meeting the HO criterion under ICH E2A.",
        "The patient was admitted to hospital — this satisfies the hospitalization seriousness criterion.",
        "Inpatient treatment was needed, directly meeting ICH E2A hospitalization {criteria}.",
        "Hospital admission is {documented}, qualifying this as a serious case under ICH E2A.",
        "The adverse event led to hospitalization or prolonged an existing stay — serious per HO criterion.",
        "The narrative describes a hospital admission, meeting one of the five ICH E2A serious outcomes.",
    ],
    'DS': [
        "Persistent or significant disability is {documented}, satisfying the DS criterion.",
        "The patient suffered lasting functional impairment — serious under the disability criterion.",
        "Substantial incapacity is reported, meeting the ICH E2A disability/incapacity threshold.",
        "Long-term disability resulting from the event meets the DS seriousness criterion.",
        "Significant disruption to the patient's normal functioning is described, satisfying DS.",
        "Permanent functional limitation {documented} — this qualifies under ICH E2A disability.",
    ],
    'CA': [
        "A congenital anomaly/birth defect is reported, meeting the CA criterion.",
        "The report describes a birth defect potentially linked to drug exposure — serious per CA.",
        "Congenital malformation identified, directly satisfying the ICH E2A CA criterion.",
        "Birth defect {documented} in association with the drug — this is automatically serious.",
        "A developmental abnormality is described, meeting the congenital anomaly seriousness {criteria}.",
    ],
}

T1_THINK_CONCLUSIONS_YES = [
    "At least one ICH E2A criterion is met — this case is serious.",
    "Based on the evidence, this qualifies as a serious adverse event.",
    "Conclusion: this meets seriousness {criteria} and requires expedited reporting.",
    "The case is serious. The clinical outcome meets regulatory reporting thresholds.",
    "Seriousness confirmed — the described outcome falls within ICH E2A serious categories.",
    "This is a reportable serious adverse event based on the clinical presentation.",
    "The outcome described satisfies at least one serious category under ICH E2A.",
    "Classification: SERIOUS. The adverse event outcome triggers regulatory reporting obligations.",
    "In summary, this case meets the definition of a serious adverse event per ICH E2A.",
    "The presence of this outcome makes this a clear-cut serious case.",
    "Final assessment: serious adverse event. Expedited reporting is required.",
    "One or more seriousness {criteria} are satisfied in this case.",
]

T1_THINK_EVIDENCE_NO = [
    "The narrative describes a self-limiting event with no indicators of serious outcome.",
    "No death, hospitalization, life-threatening situation, disability, or congenital anomaly is {documented}.",
    "The clinical outcome does not appear to meet any of the five ICH E2A seriousness {criteria}.",
    "The event {resolved} without lasting effects and did not require hospitalization.",
    "The described outcome is mild and does not trigger any ICH E2A serious category.",
    "None of the five seriousness categories appear to be satisfied.",
    "The patient's condition {resolved} and no serious clinical outcome is {documented}.",
    "The adverse event was transient and medically non-significant by ICH E2A standards.",
    "Reviewing each criterion: death (no), life-threatening (no), hospitalization (no), disability (no), CA (no).",
    "The outcome described is consistent with a non-serious adverse event.",
]

T1_THINK_CONCLUSIONS_NO = [
    "None of the ICH E2A seriousness {criteria} are met.",
    "This case does not qualify as serious under ICH E2A.",
    "Conclusion: non-serious adverse event. Routine reporting applies.",
    "The case is classified as non-serious based on the clinical outcome.",
    "No seriousness {criteria} are triggered — standard reporting timeline applies.",
    "Final assessment: non-serious. No expedited reporting required.",
    "Classification: NOT SERIOUS. The outcome does not meet any ICH E2A threshold.",
    "In summary, this adverse event does not reach the seriousness threshold.",
    "The clinical outcome is below the regulatory seriousness cutoff.",
    "None of the five serious outcome categories are applicable here.",
]

T1_ANSWER_RATIONALE_YES = [
    "Based on the clinical presentation, this case meets ICH E2A seriousness criteria.",
    "The reported outcome satisfies the seriousness threshold for expedited reporting.",
    "This adverse event is classified as serious. Expedited regulatory reporting is required.",
    "ICH E2A criteria met. The case requires expedited regulatory reporting.",
    "The clinical outcome meets one or more seriousness categories, confirming serious classification.",
    "Serious classification confirmed based on the described clinical outcome.",
    "This case qualifies for 15-day expedited reporting based on the seriousness criteria met.",
]

T1_ANSWER_RATIONALE_NO = [
    "The case does not meet any ICH E2A seriousness criteria.",
    "No seriousness criteria are triggered by the reported clinical outcome.",
    "This adverse event is non-serious per ICH E2A. Routine reporting applies.",
    "None of the five ICH E2A serious outcome categories are met.",
    "The clinical outcome does not satisfy any seriousness threshold.",
    "Assessment: non-serious. The adverse event resolved without meeting any serious criteria.",
    "No death, hospitalization, disability, life-threatening event, or birth defect is reported.",
    "Standard reporting timeline applies — no expedited reporting required.",
]

T1_INPUT_OPENERS = [
    "A {age} year-old {gender} was prescribed {drug} for {indication}.",
    "Patient: {age}-year-old {gender}, currently taking {drug} for {indication}.",
    "Clinical report for a {age} year-old {gender} receiving {drug} therapy ({indication}).",
    "{drug} was initiated for {indication} in a {age} year-old {gender}.",
    "A {gender} patient, age {age}, was started on {drug} to treat {indication}.",
    "The patient is a {age} year-old {gender} on {drug} for the management of {indication}.",
    "History: {age} year-old {gender} prescribed {drug} ({indication}).",
    "A {age}-year-old {gender} patient began treatment with {drug} for {indication}.",
]

T1_AE_SENTENCES = [
    "The patient {developed} the following adverse event(s): {ae}.",
    "The patient reported: {ae}.",
    "During treatment, the patient {developed} {ae}.",
    "An adverse event was reported: {ae}.",
    "The following adverse reaction was observed: {ae}.",
    "{ae} was reported during {drug} therapy.",
    "The patient subsequently {developed} {ae}.",
    "Clinical presentation included {ae}.",
]


# ---- T4 THINKING TRACE BANKS ----

T4_THINK_OPENERS = [
    "Applying WHO-UMC causality criteria to this clinical case.",
    "{Assess}ing the causal relationship using the WHO-UMC framework.",
    "To determine causality, I need to {assess} temporal relationship, dechallenge, rechallenge, and alternative explanations.",
    "WHO-UMC causality assessment requires checking several key factors.",
    "Let me systematically {assess} each component of the WHO-UMC criteria.",
    "Performing structured causality analysis under WHO-UMC guidelines.",
    "Beginning causality evaluation — I'll {assess} temporal plausibility, dechallenge, rechallenge, and confounders.",
    "Clinical causality assessment: applying the WHO-UMC scale to this drug-event pair.",
    "Step-by-step WHO-UMC analysis for this case.",
    "I need to weigh the evidence for and against a causal link between this drug and event.",
]

T4_TEMPORAL = {
    'strong': [
        "The {gap}-day onset aligns well with the drug's expected pharmacokinetic profile.",
        "Symptom emergence within {gap} days of drug initiation {supports} a temporal link.",
        "A {gap}-day latency is within the expected window for this type of reaction.",
        "The close temporal proximity ({gap} days) strengthens the causal hypothesis.",
        "The {gap}-day interval between drug start and event onset is temporally plausible.",
        "Temporal relationship: strong — onset {gap} days after drug initiation.",
        "The short interval ({gap} days) between starting the drug and developing the event {suggests_v} a temporal connection.",
        "Drug-event timing ({gap} days) is consistent with a pharmacological effect.",
    ],
    'plausible': [
        "The {gap}-day latency is plausible but not definitive for establishing temporal causation.",
        "Temporal relationship: event appeared {gap} days into therapy — plausible but not conclusive.",
        "A {gap}-day onset window is within the range of plausible temporal association.",
        "The timing ({gap} days) is consistent with a possible drug-related effect.",
        "Temporal plausibility: moderate — the {gap}-day interval is reasonable for this class of drug.",
        "The {gap}-day gap between drug start and event is within an acceptable temporal window.",
    ],
    'weak': [
        "The extended latency of {gap} days makes a direct temporal connection less certain.",
        "A {gap}-day gap is longer than typically expected, weakening temporal support.",
        "Temporal relationship: uncertain — {gap} days is an unusually long latency period.",
        "The {gap}-day interval is extended; temporal plausibility is weak.",
        "With {gap} days between drug start and event, the temporal link is tenuous.",
    ],
    'missing': [
        "No temporal data is available, making it impossible to {assess} the time relationship.",
        "The exact timing between drug initiation and event onset is not {documented}.",
        "Temporal plausibility cannot be assessed — dates are not recorded.",
        "Without temporal information, this key causality criterion cannot be evaluated.",
    ],
}

T4_DECHALLENGE = {
    'positive': [
        "Positive dechallenge: the adverse event {resolved} after drug withdrawal.",
        "When the drug was stopped, symptoms {resolved} — this {supports} a causal link.",
        "The resolution of symptoms after drug discontinuation provides supportive evidence.",
        "Dechallenge was positive — the event cleared when the drug was withdrawn.",
        "The drug-event link is strengthened by the observed improvement upon discontinuation.",
        "Symptoms {resolved} following drug cessation, which is a significant positive finding.",
    ],
    'negative': [
        "Negative dechallenge: symptoms persisted despite drug discontinuation.",
        "The adverse event continued even after the drug was stopped, weakening the causal link.",
        "Dechallenge was negative — no improvement was observed after drug withdrawal.",
        "The persistence of symptoms after drug cessation argues against a direct causal relationship.",
    ],
    'unknown': [
        "No dechallenge information is available — this key criterion cannot be {assessed}.",
        "It is unknown whether the drug was discontinued or what happened to symptoms afterward.",
        "Dechallenge data is missing, limiting the causality assessment.",
        "Without dechallenge information, one important piece of the causal puzzle is absent.",
    ],
}

T4_RECHALLENGE = {
    'positive': [
        "Positive rechallenge: the event recurred when the drug was reintroduced — strong causal evidence.",
        "The recurrence of the adverse event upon drug re-exposure is highly suggestive of causality.",
        "Rechallenge was positive, providing the strongest evidence for a causal relationship.",
        "Drug re-exposure provoked the same reaction, which is compelling evidence of causation.",
    ],
    'none': [
        "No rechallenge was performed or reported.",
        "Rechallenge data is not available.",
        "The drug was not reintroduced, so rechallenge evidence is absent.",
    ],
}

T4_CONFOUND = {
    'yes': [
        "Notably, the adverse event ({ae}) overlaps with the indication ({indication}), creating a confounding factor.",
        "Potential confound: the reported event is similar to the condition being treated.",
        "The indication-event overlap complicates causality assessment — the event could be disease-related.",
        "A confounding factor exists: the adverse event resembles the underlying condition.",
    ],
    'no': [
        "No indication-event overlap detected — no major confounding factor.",
        "The adverse event is distinct from the treated condition, reducing confounding risk.",
        "No obvious confounders identified between the drug indication and the adverse event.",
    ],
}

T4_CONCOMITANT = {
    'many': [
        "The patient was taking {n} other medications, providing alternative explanations for the event.",
        "With {n} concomitant drugs, attribution to a single agent is challenging.",
        "Multiple concomitant medications ({n}) are potential alternative causes.",
    ],
    'few': [
        "Only {n} concomitant medication was reported, providing limited alternative explanations.",
        "Few concomitant drugs ({n}) — alternative drug-related causes are limited.",
    ],
    'none': [
        "No concomitant medications were reported — the suspect drug is the sole agent.",
        "The absence of other medications strengthens attribution to the suspect drug.",
        "Monotherapy — no other drugs could explain the adverse event.",
    ],
}

T4_VERDICTS = {
    'Certain': [
        "All WHO-UMC criteria for 'Certain' are met: plausible time, positive dechallenge, positive rechallenge, no alternatives.",
        "This case satisfies the highest causality level. The evidence chain is complete.",
        "The combination of temporal plausibility, positive dechallenge, positive rechallenge, and absence of confounders {indicates} Certain causality.",
        "Causality: Certain. Every criterion is fulfilled with no contradictory evidence.",
    ],
    'Probable': [
        "The evidence {supports} 'Probable' causality: plausible timing, positive dechallenge, and no strong alternative explanations.",
        "This case meets Probable/Likely level — strong temporal link with positive dechallenge but no rechallenge.",
        "Causality assessment: Probable. The evidence is strong but falls short of Certain due to absence of rechallenge.",
        "With plausible timing and positive dechallenge in the absence of confounders, Probable is the appropriate level.",
    ],
    'Possible': [
        "The evidence {suggests_v} 'Possible' causality — temporal plausibility exists but confounders or incomplete evidence limit certainty.",
        "Causality: Possible. There is a reasonable temporal relationship but alternative explanations cannot be ruled out.",
        "This case fits the Possible category — some evidence supports causation but it is not conclusive.",
        "The temporal link exists but the presence of confounders or missing dechallenge data limits the assessment to Possible.",
    ],
    'Unlikely': [
        "The evidence does not {support} a causal relationship — causality is classified as Unlikely.",
        "Causality: Unlikely. The temporal relationship is implausible or alternative explanations are more likely.",
        "Weak or absent temporal link combined with plausible alternative explanations makes this Unlikely.",
        "Assessment: Unlikely. The evidence weighs against a causal relationship between the drug and the event.",
    ],
    'Conditional': [
        "Insufficient data for a definitive assessment — classified as Conditional/Unclassified.",
        "More information is needed to make a firm causality determination. Conditional classification applies.",
        "Causality: Conditional. Some data exists but a complete assessment is not possible.",
    ],
    'Unassessable': [
        "The available information is insufficient or contradictory — causality is Unassessable.",
        "Causality cannot be assessed due to missing essential data (temporal, dechallenge, rechallenge).",
        "Unassessable: insufficient information to apply the WHO-UMC criteria meaningfully.",
        "Without adequate clinical data, no causality determination can be made.",
    ],
}


# ---- T2 THINKING TRACE BANKS ----

T2_THINK_PATTERNS = [
    "Analyzing the clinical description for the core adverse event. The text {indicates} the MedDRA Preferred Term '{pt}'.",
    "The medical terminology in this passage {corresponds} to the MedDRA term '{pt}'.",
    "Based on the clinical context and drug involvement, the appropriate coding is '{pt}'.",
    "Cross-referencing the described symptoms with MedDRA hierarchy: '{pt}' at the PT level.",
    "{Assess}ing the adverse event description. The standardized MedDRA coding for this is '{pt}'.",
    "The clinical language used {corresponds} to the MedDRA Preferred Term '{pt}'.",
    "After {reviewing} the clinical text, the adverse reaction described maps to '{pt}' in MedDRA terminology.",
    "Medical coding analysis: the described adverse event aligns with MedDRA PT '{pt}'.",
    "Reading the clinical context carefully. The adverse drug reaction described here is best coded as '{pt}'.",
    "The reported reaction, in standardized medical terminology, is classified as '{pt}' under MedDRA.",
]

T2_ANSWER_RATIONALES = [
    "Based on the clinical description and pharmacological context, '{pt}' is the appropriate MedDRA Preferred Term for coding this adverse event.",
    "The adverse event described in the clinical text maps to '{pt}' under MedDRA coding standards.",
    "After analyzing the clinical context, '{pt}' is the correct standardized terminology.",
    "The described reaction corresponds to MedDRA PT '{pt}' based on the clinical presentation.",
    "'{pt}' is the most appropriate MedDRA Preferred Term for this adverse drug reaction.",
    "Clinical-to-MedDRA mapping: the described event codes to '{pt}'.",
]


# ---- T3 THINKING TRACE BANKS ----

T3_THINK_LABELLED_YES = [
    "Checking whether '{ae}' is {documented} in the approved label for {drug}. Searching the product label sections: '{ae}' {appears} in the {section} section. This adverse event is {recognized} for this drug.",
    "Reviewing the safety profile of {drug}. The adverse event '{ae}' is {documented} in the drug's label under {section}.",
    "After {reviewing} all label sections for {drug}, '{ae}' is found in the {section} section as a known adverse reaction.",
    "Label check for {drug}: '{ae}' is listed in the {section} section of the approved product label.",
    "{Assess}ing whether '{ae}' is a known reaction for {drug}. Yes — it is {documented} in the {section} section.",
    "The adverse event '{ae}' is a {recognized} adverse reaction for {drug}, found in the product label's {section} section.",
    "Consulting the approved label for {drug}: '{ae}' {appears} under {section}. This is a known adverse reaction.",
    "Searching the drug label for {drug}. Result: '{ae}' is listed in {section}. Labelled: yes.",
]

T3_THINK_LABELLED_YES_WITH_CLASS = [
    "Checking '{ae}' against the safety profile of {drug}. Drug class: {drug_class}. Mechanism: {mechanism}. The event '{ae}' {appears} in the {section} section of the approved label.",
    "{Assess}ing '{ae}' for {drug} ({drug_class}, {mechanism}). This adverse event is {documented} in the label's {section} section.",
    "Given {drug}'s pharmacological profile ({drug_class}), '{ae}' is a {recognized} adverse reaction found in the {section} section.",
    "Reviewing {drug} ({drug_class}) label. '{ae}' is listed in {section} as a known adverse reaction.",
]

T3_THINK_UNLABELLED = [
    "Checking whether '{ae}' is {documented} in the approved label for {drug}. After {reviewing} all sections, '{ae}' is NOT found. This may be an unexpected adverse reaction.",
    "Reviewing the safety profile of {drug}. The adverse event '{ae}' does not appear in any section of the drug's product label.",
    "After searching all label sections for {drug}, '{ae}' is not {documented} as a known adverse reaction.",
    "Label check for {drug}: '{ae}' is not found in any section. This could represent an unlabelled/unexpected reaction.",
    "{Assess}ing whether '{ae}' is a known reaction for {drug}. No — it is not {documented} in the product label.",
    "The adverse event '{ae}' is NOT found in the approved label for {drug}. This may require expedited reporting.",
    "Searching the drug label for {drug}. Result: '{ae}' not found. This may be an unlabelled adverse reaction.",
]

T3_THINK_UNLABELLED_WITH_CLASS = [
    "Checking '{ae}' against the safety profile of {drug}. Drug class: {drug_class}. Mechanism: {mechanism}. After {reviewing} all sections, '{ae}' is NOT found in the approved label.",
    "{Assess}ing '{ae}' for {drug} ({drug_class}, {mechanism}). This adverse event is not {documented} in the product label.",
    "Given {drug}'s pharmacological profile ({drug_class}), '{ae}' is not a {recognized} adverse reaction in the label. This may be unexpected.",
]

T3_ANSWER_RATIONALE_YES = [
    "This adverse event is {documented} in the drug's approved product label under the {section} section as a known adverse reaction.",
    "The adverse event is a {recognized} reaction for this drug, listed in the {section} section.",
    "'{ae}' is a known adverse reaction for {drug}, found in the product label ({section}).",
    "Label check confirms: this is a {documented} adverse reaction for this drug.",
    "The drug's label includes '{ae}' in its {section} section.",
]

T3_ANSWER_RATIONALE_NO = [
    "This adverse event is not {documented} in the drug's approved product label. It may represent an unlabelled/unexpected adverse reaction requiring expedited regulatory reporting.",
    "'{ae}' is not found in any section of the drug's label. This could be an unexpected reaction.",
    "The adverse event is not listed as a known reaction for {drug}. Expedited 15-day reporting may be required.",
    "After reviewing all label sections, this event is not {documented}. It may represent an unlabelled reaction.",
    "The drug's product label does not include '{ae}'. This may be an unexpected adverse event.",
]


def build_t1_pairs(df: pd.DataFrame, max_pairs: int = 10000) -> list[dict]:
    """Build Task 1 (Seriousness) training pairs — NARRATIVE-BASED + DIVERSIFIED.

    Uses Combinatorial Diversity Engine for unique thinking traces and answers.
    Expected unique completions: 8,000+ (was 299 with old templates).
    """
    print("\n  📋 Task 1: Seriousness Assessment (DIVERSIFIED)...")

    # Aggregate per case
    case_outcomes = df.groupby('primaryid').agg({
        'outc_cod': lambda x: ','.join(set(x.dropna().astype(str))),
        'age': 'first', 'age_cod': 'first', 'gndr_cod': 'first',
        'drugname': 'first',
        'meddra_pt': lambda x: ', '.join(sorted(set(x.dropna().astype(str)))),
        'indi_pt': 'first', 'occp_cod': 'first',
    }).reset_index()

    # Early sampling
    sample_cap = max_pairs * 5
    if len(case_outcomes) > sample_cap:
        case_outcomes = case_outcomes.sample(n=sample_cap, random_state=SEED)

    pairs = []
    random.seed(SEED)

    for _, row in case_outcomes.iterrows():
        outc = str(row.get('outc_cod', ''))
        codes = set(outc.split(',')) if outc else set()
        serious_codes_found = codes & SERIOUS_CODES
        is_serious = len(serious_codes_found) > 0

        # Build patient context
        age = _age_str(row)
        gender = {'M': 'male', 'F': 'female'}.get(str(row.get('gndr_cod', '')), 'unknown gender')
        gender_abbrev = {'M': 'M', 'F': 'F'}.get(str(row.get('gndr_cod', '')), '?')
        drug = _safe_str(row.get('drugname'), 'Unknown drug')
        ae = _safe_str(row.get('meddra_pt'), 'Unknown event')
        indication = _safe_str(row.get('indi_pt'), 'unknown indication')

        # === Convert outcome codes to NARRATIVE ===
        if is_serious:
            outcome_parts = []
            for code in serious_codes_found:
                if code in OUTCOME_NARRATIVES:
                    outcome_parts.append(random.choice(OUTCOME_NARRATIVES[code]))
            outcome_narrative = ' '.join(outcome_parts) if outcome_parts else "A serious clinical outcome was reported."
        else:
            outcome_narrative = random.choice(NON_SERIOUS_NARRATIVES)

        # === DIVERSIFIED INPUT ===
        opener_template = random.choice(T1_INPUT_OPENERS)
        opener = opener_template.replace('{age}', age).replace('{gender}', gender)
        opener = opener.replace('{gender_abbrev}', gender_abbrev)
        opener = opener.replace('{drug}', drug).replace('{indication}', indication)

        ae_template = random.choice(T1_AE_SENTENCES)
        ae_sentence = ae_template.replace('{ae}', ae).replace('{drug}', drug)
        ae_sentence = _diversify(ae_sentence)

        input_text = f"{opener} {ae_sentence} {outcome_narrative}"

        # === DIVERSIFIED THINKING TRACE ===
        if is_serious:
            criteria_str = ', '.join(f"{c} ({SERIOUS_NAMES.get(c, c)})" for c in serious_codes_found)

            # Assemble: opener + evidence per code + conclusion
            think_opener = _diversify(random.choice(T1_THINK_OPENERS))
            think_evidence_parts = []
            for code in serious_codes_found:
                if code in T1_THINK_EVIDENCE_YES:
                    think_evidence_parts.append(_diversify(random.choice(T1_THINK_EVIDENCE_YES[code])))
            think_evidence = ' '.join(think_evidence_parts) if think_evidence_parts else f"The narrative {_diversify('{indicates}')} seriousness criteria: {criteria_str}."
            think_conclusion = _diversify(random.choice(T1_THINK_CONCLUSIONS_YES))

            think = f"{think_opener} {think_evidence} {think_conclusion}"

            # Diversified answer
            rationale = _diversify(random.choice(T1_ANSWER_RATIONALE_YES))
            answer = (
                f"SERIOUS: YES\n"
                f"Criteria met: {criteria_str}\n"
                f"Rationale: {rationale}"
            )
        else:
            think_opener = _diversify(random.choice(T1_THINK_OPENERS))
            think_evidence = _diversify(random.choice(T1_THINK_EVIDENCE_NO))
            think_conclusion = _diversify(random.choice(T1_THINK_CONCLUSIONS_NO))

            think = f"{think_opener} {think_evidence} {think_conclusion}"

            rationale = _diversify(random.choice(T1_ANSWER_RATIONALE_NO))
            answer = (
                f"SERIOUS: NO\n"
                f"Criteria met: None\n"
                f"Rationale: {rationale}"
            )

        pairs.append({
            "task": "T1",
            "primaryid": str(row.get('primaryid', '')),
            "thinking": True,
            "messages": [
                {"role": "system", "content": "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). Base your assessment on the clinical narrative provided. Think step by step."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": f"<|channel>thought\n{think}\n<channel|>\n{answer}"}
            ]
        })

    # Balance classes (60% YES, 40% NO)
    yes_pairs = [p for p in pairs if "SERIOUS: YES" in p['messages'][-1]['content']]
    no_pairs = [p for p in pairs if "SERIOUS: NO" in p['messages'][-1]['content']]

    target_no = int(len(yes_pairs) * 0.67)
    random.seed(SEED)
    no_sampled = random.sample(no_pairs, min(target_no, len(no_pairs)))

    balanced = yes_pairs + no_sampled
    random.shuffle(balanced)
    balanced = balanced[:max_pairs]

    print(f"     YES: {len(yes_pairs):,} | NO sampled: {len(no_sampled):,} | Final: {len(balanced):,}")
    return balanced


# ============================================================
# TASK 2: MedDRA Coding — LAY/CLINICAL LANGUAGE → PT (not copy task)
# ============================================================

def _load_external_t2_data() -> list[dict]:
    """Load external datasets for T2 (lay/clinical language → MedDRA PT)."""
    external_file = EXTERNAL_DIR / "external_pairs.jsonl"
    if not external_file.exists():
        print("     ⚠️ No external data found. Run: python src/data/05_download_external_datasets.py")
        return []

    pairs = []
    with open(str(external_file), 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('task_type') == 'T2':
                    pairs.append(data)
            except:
                continue

    print(f"     External T2 data loaded: {len(pairs):,} pairs")
    return pairs


def build_t2_pairs(df: pd.DataFrame, max_pairs: int = 8000) -> list[dict]:
    """Build Task 2 (MedDRA Coding) training pairs — GENUINE CODING TASK.

    Uses EXTERNAL data only (BioDEX, ADE Corpus v2):
    - BioDEX: biomedical literature abstracts → MedDRA-coded reactions
    - ADE Corpus: clinical sentences with drug-ADE relation annotations

    C3 FIX: FAERS T2 pairs REMOVED — they put the MedDRA PT directly in the
    input text ("Adverse event to code: {pt}"), making it a trivial copy task.
    External data provides genuine medical language → MedDRA mapping.
    """
    print("\n  📋 Task 2: MedDRA Code Suggestion (GENUINE CODING)...")

    pairs = []
    random.seed(SEED)

    # === TIER 1: External datasets (lay/clinical language → PT) ===
    external_t2 = _load_external_t2_data()

    for ext in external_t2:
        source = ext.get('source', 'Unknown')
        drug = ext.get('drug', 'Unknown drug')

        if source in ('PHEE', 'ADE_corpus'):
            # T2-FIX: SKIP non-MedDRA sources entirely.
            # PHEE/ADE Corpus use raw entity text as target PT (e.g., "tense bullae",
            # "15-kg weight gain", "100 mg/m2"). These are NOT real MedDRA Preferred Terms.
            # Training on mixed labels (real MedDRA from BioDEX + raw text from ADE Corpus)
            # causes the model to produce outputs that don't match either vocabulary.
            # Evidence: T2 eval scored 2% exact match with mixed data.
            # BioDEX alone has 92,202 pairs with 7,206 verified MedDRA PTs — sufficient.
            continue

        elif source == 'BioDEX':
            # Has drug + reaction term (already MedDRA-coded)
            reaction = ext.get('reaction_term', '')
            source_text = ext.get('source_text', '')
            if not reaction or len(reaction) < 3:
                continue

            # H2 FIX: Skip BioDEX entries without an abstract entirely.
            # Without clinical text, the prompt has no details — the model would
            # be forced to hallucinate a random reaction, corrupting gradients.
            if not source_text or len(source_text.strip()) < 30:
                continue

            pt = reaction.strip()

            # H1 FIX: Smart truncation — ensure the target reaction term appears
            # in the truncated text. Blind [:300] can cut BEFORE the AE mention,
            # forcing the model to hallucinate (corrupting SFT gradients).
            if len(source_text) > 400:
                pt_lower = pt.lower()
                src_lower = source_text.lower()
                pt_pos = src_lower.find(pt_lower)
                if pt_pos >= 0:
                    # Center window around the reaction mention
                    window_start = max(0, pt_pos - 150)
                    window_end = min(len(source_text), pt_pos + len(pt) + 150)
                    source_excerpt = source_text[window_start:window_end]
                    if window_start > 0:
                        source_excerpt = '...' + source_excerpt
                    if window_end < len(source_text):
                        source_excerpt = source_excerpt + '...'
                else:
                    # PT not found in text — skip this pair entirely
                    # Training on it would teach hallucination
                    continue
            else:
                source_excerpt = source_text

            input_text = (
                f"From biomedical literature about {drug}: \"{source_excerpt}\"\n\n"
                f"Identify and code the adverse drug reaction described in this text "
                f"using MedDRA Preferred Terms."
            )

            think_template = random.choice(T2_THINK_PATTERNS)
            think = _diversify(think_template.replace('{pt}', pt).replace('{drug}', drug))

            # H3 FIX: content-based ID
            source_hash = hashlib.sha256(source_text.encode()).hexdigest()[:12]
            pair_id = f"ext_{source}_{source_hash}"
        else:
            continue

        rationale_template = random.choice(T2_ANSWER_RATIONALES)
        rationale = _diversify(rationale_template.replace('{pt}', pt))
        answer = (
            f"MedDRA PT: {pt}\n"
            f"Drug context: {drug}\n"
            f"Rationale: {rationale}"
        )

        pairs.append({
            "task": "T2",
            "primaryid": pair_id,  # H3 FIX: content-based, not sequential
            "thinking": True,
            "messages": [
                {"role": "system", "content": "You are a medical coder specializing in MedDRA terminology. Given an adverse event description from clinical or patient-reported text, map it to the correct MedDRA Preferred Term (PT). Think step by step about the medical terminology and clinical context."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": f"<|channel>thought\n{think}\n<channel|>\n{answer}"}
            ]
        })

    ext_count = len(pairs)
    print(f"     Tier 1 (external): {ext_count:,} pairs")

    # C3 FIX: FAERS T2 pairs REMOVED — they leaked the answer (PT in input).
    # T2-FIX: PHEE/ADE Corpus REMOVED — raw entity text, not real MedDRA PTs.
    # BioDEX alone provides verified MedDRA PTs from PubMed literature.
    print(f"     PHEE/ADE Corpus: REMOVED (T2-FIX — raw entity text, not MedDRA PTs)")
    print(f"     FAERS T2 pairs: REMOVED (C3 fix — answer was in input)")

    random.seed(SEED)
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    print(f"     Total T2 pairs: {len(pairs):,}")
    return pairs


# ============================================================
# TASK 3: Labelling Status — CLASS-AWARE REASONING
# ============================================================

def _load_drug_class_map() -> dict:
    """Load drug class / mechanism map for T3 reasoning."""
    map_file = EXTERNAL_DIR / "drug_class_map.json"
    if map_file.exists():
        with open(str(map_file), 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _lookup_drug_info(drugname: str, drug_class_map: dict) -> dict | None:
    """Look up drug class and mechanism."""
    if not drugname:
        return None
    name = drugname.strip().lower()
    # Direct match
    if name in drug_class_map:
        return drug_class_map[name]
    # Substring match
    for key, info in drug_class_map.items():
        if key in name or name.startswith(key):
            return info
    return None


# OnSIDES label_section codes → human-readable names
_SECTION_MAP = {
    'AR': 'Adverse Reactions',
    'BW': 'Boxed Warning',
    'WP': 'Warnings and Precautions',
    'WA': 'Warnings',
    'PR': 'Precautions',
}


def build_t3_pairs(df: pd.DataFrame, max_pairs: int = 10000) -> list[dict]:
    """Build Task 3 (Labelling) training pairs — CLASS-AWARE REASONING.

    KEY CHANGE: Add drug class and mechanism of action to the input so the
    model can REASON about whether an AE is expected for a drug class, not
    just memorize a lookup table.
    """
    print("\n  📋 Task 3: Labelling Status (CLASS-AWARE REASONING)...")

    drug_class_map = _load_drug_class_map()
    if drug_class_map:
        print(f"     Drug class map loaded: {len(drug_class_map)} drugs")
    else:
        print(f"     ⚠️ No drug class map. Run: python src/data/05_download_external_datasets.py")

    # Load OnSIDES lookup
    onsides_parquet = Path("data/external/onsides_lookup.parquet")
    use_onsides = False
    onsides_lookup = None

    if onsides_parquet.exists():
        try:
            onsides_lookup = pd.read_parquet(onsides_parquet)
            if len(onsides_lookup) > 0:
                use_onsides = True
                print(f"     ✅ OnSIDES loaded: {len(onsides_lookup):,} drug-AE pairs")
        except Exception as e:
            print(f"     ⚠️ OnSIDES load failed: {e}")

    if not use_onsides:
        print(f"     ⚠️ OnSIDES not found. Falling back to frequency heuristic.")
        return _build_t3_with_heuristic(df, max_pairs, drug_class_map)

    return _build_t3_with_onsides(df, onsides_lookup, max_pairs, drug_class_map)


def _build_t3_with_onsides(df: pd.DataFrame, onsides: pd.DataFrame,
                           max_pairs: int, drug_class_map: dict) -> list[dict]:
    """Build T3 pairs with OnSIDES ground truth + drug class reasoning."""
    import re as _re

    # Drug name normalizer
    _DOSAGE_FORMS = _re.compile(
        r'\b(tablet|capsule|injection|solution|suspension|cream|ointment|gel|'
        r'patch|spray|inhaler|drops|syrup|powder|granules|oral|topical|'
        r'intravenous|subcutaneous|intramuscular|for\s+injection)\b',
        _re.IGNORECASE
    )
    _STRENGTH_PATTERN = _re.compile(
        r'\b\d+[\.,]?\d*\s*(?:mg|mcg|g|ml|%|iu|units?|meq|mmol)\b',
        _re.IGNORECASE
    )
    def _normalize(name):
        if not name or name == 'nan': return ''
        name = name.lower().strip()
        name = _STRENGTH_PATTERN.sub('', name)
        name = _DOSAGE_FORMS.sub('', name)
        name = _re.sub(r'[,/\(\)\[\]\{\}]', ' ', name)
        return _re.sub(r'\s+', ' ', name).strip()

    print("     Using OnSIDES + drug class reasoning")

    # Build OnSIDES lookup sets
    has_drug_name = 'drug_name' in onsides.columns
    drugname_set = set()
    drugname_drugs = set()
    normalized_set = set()
    normalized_drugs = set()

    if has_drug_name:
        dn = onsides[onsides['drug_name'].notna()].copy()
        drugname_set = set(zip(dn['drug_name'].astype(str).str.lower(), dn['pt_meddra_term'].astype(str).str.lower()))
        drugname_drugs = set(dn['drug_name'].astype(str).str.lower())
        dn['_norm'] = dn['drug_name'].astype(str).apply(_normalize)
        dn = dn[dn['_norm'] != '']
        normalized_set = set(zip(dn['_norm'], dn['pt_meddra_term'].astype(str).str.lower()))
        normalized_drugs = set(dn['_norm'])

    # Strategy 3: Active ingredient matching
    # OnSIDES has ingredient_name column with 1,671 unique ingredients
    # FAERS has prod_ai (active ingredient) — matching these adds 10× drug coverage
    ingredient_set = set()  # (ingredient, pt) pairs
    ingredient_drugs = set()  # ingredients that have OnSIDES coverage
    has_ingredient = 'ingredient_name' in onsides.columns
    if has_ingredient:
        ig = onsides[onsides['ingredient_name'].notna()].copy()
        ingredient_set = set(zip(ig['ingredient_name'].astype(str).str.lower(), ig['pt_meddra_term'].astype(str).str.lower()))
        ingredient_drugs = set(ig['ingredient_name'].astype(str).str.lower())
        print(f"     Ingredient lookup: {len(ingredient_drugs):,} unique ingredients")

    # Section lookup (by drug name)
    section_lookup = {}
    if 'label_section' in onsides.columns and has_drug_name:
        sect_df = onsides[['drug_name', 'pt_meddra_term', 'label_section']].drop_duplicates().dropna()
        sect_df['section_mapped'] = sect_df['label_section'].str.upper().str.strip().map(_SECTION_MAP)
        sect_df['section_mapped'] = sect_df['section_mapped'].fillna(sect_df['label_section'])
        section_lookup = dict(zip(
            zip(sect_df['drug_name'].astype(str).str.lower(), sect_df['pt_meddra_term'].astype(str).str.lower()),
            sect_df['section_mapped']
        ))

    # Section lookup (by ingredient — fallback for Strategy 3 matches)
    ingredient_section_lookup = {}
    if 'label_section' in onsides.columns and has_ingredient:
        isect = onsides[['ingredient_name', 'pt_meddra_term', 'label_section']].drop_duplicates().dropna()
        isect['section_mapped'] = isect['label_section'].str.upper().str.strip().map(_SECTION_MAP)
        isect['section_mapped'] = isect['section_mapped'].fillna(isect['label_section'])
        ingredient_section_lookup = dict(zip(
            zip(isect['ingredient_name'].astype(str).str.lower(), isect['pt_meddra_term'].astype(str).str.lower()),
            isect['section_mapped']
        ))

    # Prepare FAERS data
    drug_pt = df[df['drugname'].notna() & df['meddra_pt'].notna()].copy()
    drug_pt['pt_clean'] = drug_pt['meddra_pt'].str.strip().str.lower()
    drug_pt['drugname_clean'] = drug_pt['drugname'].str.strip().str.lower()
    drug_pt['drugname_normalized'] = drug_pt['drugname_clean'].apply(_normalize)
    # Active ingredient: FAERS prod_ai may contain semicolon-separated ingredients
    drug_pt['ai_clean'] = drug_pt.get('prod_ai', pd.Series(dtype=str)).fillna('').str.strip().str.lower()

    case_data = drug_pt.drop_duplicates(subset=['primaryid', 'drugname_clean', 'pt_clean'])
    if len(case_data) > max_pairs * 5:
        case_data = case_data.sample(n=max_pairs * 5, random_state=SEED)

    pairs = []
    labelled_count = 0
    unlabelled_count = 0
    dropped_count = 0
    class_enriched = 0

    random.seed(SEED)

    for _, row in case_data.iterrows():
        drug = row.get('drugname', 'Unknown drug')
        ae = row.get('meddra_pt', 'Unknown event')
        pt = str(row['pt_clean'])
        drugname = str(row['drugname_clean'])
        drugname_norm = str(row.get('drugname_normalized', ''))
        ai_raw = str(row.get('ai_clean', ''))
        indication = row.get('indi_pt', 'unknown indication')

        # OnSIDES matching (3-strategy cascade)
        # Strategy 1: exact drugname → Strategy 2: normalized → Strategy 3: active ingredient
        has_coverage = False
        is_labelled = False

        if drugname in drugname_drugs:
            has_coverage = True
            is_labelled = (drugname, pt) in drugname_set
        elif drugname_norm and drugname_norm in normalized_drugs:
            has_coverage = True
            is_labelled = (drugname_norm, pt) in normalized_set
        elif ai_raw and has_ingredient:
            # Strategy 3: Match each active ingredient in prod_ai
            # FAERS uses backslash separator: "ETHINYL ESTRADIOL\\LEVONORGESTREL"
            for ing in _re.split(r'[;\\]+', ai_raw):
                ing = ing.strip()
                if ing and ing in ingredient_drugs:
                    has_coverage = True
                    is_labelled = (ing, pt) in ingredient_set
                    break

        if not has_coverage:
            dropped_count += 1
            continue

        # === KEY CHANGE: Add drug class + mechanism for reasoning ===
        drug_info = _lookup_drug_info(drug, drug_class_map)
        if drug_info:
            drug_class = drug_info.get('class', 'Unknown class')
            mechanism = drug_info.get('mechanism', 'Unknown mechanism')
            class_enriched += 1

            input_text = (
                f"Drug: {drug}\n"
                f"Pharmacological class: {drug_class}\n"
                f"Mechanism of action: {mechanism}\n"
                f"Indication: {indication}\n"
                f"Reported adverse event: {ae}\n\n"
                f"Based on the drug's pharmacological class, mechanism of action, "
                f"and known safety profile, determine whether this adverse event "
                f"is listed in the drug's approved product label."
            )
        else:
            input_text = (
                f"Drug: {drug} (prescribed for {indication}). "
                f"Adverse event reported: {ae}. "
                f"Is this adverse event listed in the drug's approved product label?"
            )

        if is_labelled:
            labelled_count += 1
            section = section_lookup.get((drugname, pt),
                      section_lookup.get((drugname_norm, pt),
                      ingredient_section_lookup.get((ai_raw, pt), 'Adverse Reactions')))

            # DIVERSIFIED T3 thinking trace
            if drug_info:
                think_template = random.choice(T3_THINK_LABELLED_YES_WITH_CLASS)
                think = think_template.replace('{ae}', ae).replace('{drug}', drug)
                think = think.replace('{drug_class}', drug_info['class'])
                think = think.replace('{mechanism}', drug_info['mechanism'])
                think = think.replace('{section}', section)
                think = _diversify(think)
            else:
                think_template = random.choice(T3_THINK_LABELLED_YES)
                think = think_template.replace('{ae}', ae).replace('{drug}', drug)
                think = think.replace('{section}', section)
                think = _diversify(think)

            rationale_template = random.choice(T3_ANSWER_RATIONALE_YES)
            rationale = rationale_template.replace('{ae}', ae).replace('{drug}', drug)
            rationale = rationale.replace('{section}', section)
            rationale = _diversify(rationale)
            answer = (
                f"LABELLED: YES\n"
                f"Drug: {drug}\n"
                f"Adverse event: {ae}\n"
                f"Label section: {section}\n"
                f"Rationale: {rationale}"
            )
        else:
            unlabelled_count += 1
            # DIVERSIFIED T3 unlabelled thinking trace
            if drug_info:
                think_template = random.choice(T3_THINK_UNLABELLED_WITH_CLASS)
                think = think_template.replace('{ae}', ae).replace('{drug}', drug)
                think = think.replace('{drug_class}', drug_info['class'])
                think = think.replace('{mechanism}', drug_info['mechanism'])
                think = _diversify(think)
            else:
                think_template = random.choice(T3_THINK_UNLABELLED)
                think = think_template.replace('{ae}', ae).replace('{drug}', drug)
                think = _diversify(think)

            rationale_template = random.choice(T3_ANSWER_RATIONALE_NO)
            rationale = rationale_template.replace('{ae}', ae).replace('{drug}', drug)
            rationale = _diversify(rationale)
            answer = (
                f"LABELLED: NO\n"
                f"Drug: {drug}\n"
                f"Adverse event: {ae}\n"
                f"Label section: Not found\n"
                f"Rationale: {rationale}"
            )

        pairs.append({
            "task": "T3",
            "primaryid": str(row.get('primaryid', '')),
            "thinking": True,
            "messages": [
                {"role": "system", "content": "You are a pharmacovigilance expert. Determine if the reported adverse event is listed in the drug's approved product label. Consider the drug's pharmacological class and mechanism of action in your assessment. An unlabelled adverse event requires expedited 15-day reporting. Think step by step."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": f"<|channel>thought\n{think}\n<channel|>\n{answer}"}
            ]
        })

    # T3 FIX: Changed from 40:60 to 50:50 YES:NO ratio.
    # Evidence: 62% of eval-YES drug-AE pairs are UNSEEN in training.
    # SIMPLE prompts were 38% YES → model learned "unknown pair = NO".
    # With 50:50, model has no prior bias — must reason about the pair.
    yes_pairs = [p for p in pairs if "LABELLED: YES" in p['messages'][-1]['content']]
    no_pairs = [p for p in pairs if "LABELLED: NO" in p['messages'][-1]['content']]

    yes_target = min(len(yes_pairs), int(max_pairs * 0.5))
    no_target = min(len(no_pairs), int(max_pairs * 0.5), yes_target)

    if yes_target > 0 and no_target > 0:
        balanced = random.sample(yes_pairs, yes_target) + random.sample(no_pairs, no_target)
    else:
        balanced = pairs[:max_pairs]
    random.shuffle(balanced)

    print(f"     OnSIDES: {labelled_count:,} labelled | {unlabelled_count:,} unlabelled | {dropped_count:,} dropped")
    print(f"     Class-enriched: {class_enriched:,} pairs have drug class + mechanism")
    print(f"     Balanced: {yes_target:,} YES + {no_target:,} NO = {len(balanced):,} total")
    return balanced


def _build_t3_with_heuristic(df: pd.DataFrame, max_pairs: int, drug_class_map: dict) -> list[dict]:
    """Fallback: frequency heuristic for T3 when OnSIDES unavailable."""
    print("     ⚠️ Using frequency heuristic (less accurate)")

    drug_pt = df[df['drugname'].notna() & df['meddra_pt'].notna()].copy()
    drug_pt['drugname_lower'] = drug_pt['drugname'].str.upper().str.strip()
    drug_pt['pt_lower'] = drug_pt['meddra_pt'].str.lower().str.strip()

    co_occur = drug_pt.groupby(['drugname_lower', 'pt_lower']).size().reset_index(name='count')
    drug_total = drug_pt.groupby('drugname_lower').size().reset_index(name='drug_total')
    co_occur = co_occur.merge(drug_total, on='drugname_lower')
    co_occur['frequency'] = co_occur['count'] / co_occur['drug_total']
    co_occur['is_labelled'] = co_occur['frequency'] > 0.05

    case_data = drug_pt.drop_duplicates(subset=['primaryid', 'drugname_lower', 'pt_lower'])
    case_data = case_data.merge(
        co_occur[['drugname_lower', 'pt_lower', 'is_labelled', 'frequency']],
        on=['drugname_lower', 'pt_lower'], how='left'
    )
    case_data = case_data[case_data['is_labelled'].notna()]

    if len(case_data) > max_pairs * 3:
        case_data = case_data.sample(n=max_pairs * 3, random_state=SEED)

    pairs = []
    random.seed(SEED)

    for _, row in case_data.iterrows():
        drug = row.get('drugname', 'Unknown drug')
        ae = row.get('meddra_pt', 'Unknown event')
        is_labelled = row.get('is_labelled', False)
        indication = row.get('indi_pt', 'unknown indication')

        drug_info = _lookup_drug_info(drug, drug_class_map)
        if drug_info:
            input_text = (
                f"Drug: {drug}\n"
                f"Pharmacological class: {drug_info['class']}\n"
                f"Mechanism: {drug_info['mechanism']}\n"
                f"Adverse event: {ae}\n\n"
                f"Is this adverse event listed in the drug's approved product label?"
            )
        else:
            input_text = (
                f"Drug: {drug} (prescribed for {indication}). "
                f"Adverse event reported: {ae}. "
                f"Is this adverse event listed in the drug's approved product label?"
            )

        if is_labelled:
            think_template = random.choice(T3_THINK_LABELLED_YES)
            think = think_template.replace('{ae}', ae).replace('{drug}', drug)
            think = think.replace('{section}', 'Adverse Reactions')
            think = _diversify(think)
            rationale_template = random.choice(T3_ANSWER_RATIONALE_YES)
            rationale = rationale_template.replace('{ae}', ae).replace('{drug}', drug)
            rationale = rationale.replace('{section}', 'Adverse Reactions')
            rationale = _diversify(rationale)
            answer = f"LABELLED: YES\nDrug: {drug}\nAdverse event: {ae}\nLabel section: Adverse Reactions\nRationale: {rationale}"
        else:
            think_template = random.choice(T3_THINK_UNLABELLED)
            think = think_template.replace('{ae}', ae).replace('{drug}', drug)
            think = _diversify(think)
            rationale_template = random.choice(T3_ANSWER_RATIONALE_NO)
            rationale = rationale_template.replace('{ae}', ae).replace('{drug}', drug)
            rationale = _diversify(rationale)
            answer = f"LABELLED: NO\nDrug: {drug}\nAdverse event: {ae}\nLabel section: Not found\nRationale: {rationale}"

        pairs.append({
            "task": "T3",
            "primaryid": str(row.get('primaryid', '')),
            "thinking": True,
            "messages": [
                {"role": "system", "content": "You are a pharmacovigilance expert. Determine if the reported adverse event is listed in the drug's approved product label. Think step by step."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": f"<|channel>thought\n{think}\n<channel|>\n{answer}"}
            ]
        })

    yes_p = [p for p in pairs if "LABELLED: YES" in p['messages'][-1]['content']]
    no_p = [p for p in pairs if "LABELLED: NO" in p['messages'][-1]['content']]
    target = min(max_pairs // 2, len(yes_p), len(no_p))
    if target > 0:
        balanced = random.sample(yes_p, min(target, len(yes_p))) + random.sample(no_p, min(target, len(no_p)))
    else:
        balanced = pairs[:max_pairs]
    random.shuffle(balanced)
    print(f"     Heuristic: {len(yes_p):,} labelled | {len(no_p):,} unlabelled | Final: {len(balanced):,}")
    return balanced


# ============================================================
# TASK 4: Causality Assessment — CLINICAL NARRATIVE (evidence extraction)
# ============================================================

def compute_causality(row) -> str | None:
    """Apply WHO-UMC causality rules to a case.
    
    S3 FIX: Complete rewrite to check ALL criteria simultaneously per WHO-UMC scale.
    Previous version had: (1) Certain didn't check for alternative explanations,
    (2) Unlikely overrode positive dechallenge+rechallenge, (3) missing Conditional level.
    
    WHO-UMC levels:
    - Certain: plausible time + dechallenge + rechallenge + no alternatives
    - Probable/Likely: plausible time + dechallenge + no alternatives (rechallenge not needed)
    - Possible: plausible time + could be other causes
    - Unlikely: improbable time OR more likely alternative
    - Conditional/Unclassified: more data needed for assessment
    - Unassessable/Unclassifiable: insufficient or contradictory information
    """
    dechal_raw = str(row.get('dechal', '')).upper()
    dechal = dechal_raw == 'Y'  # Positive dechallenge
    neg_dechal = dechal_raw == 'N'  # Negative dechallenge (evidence AGAINST causality)
    rechal = str(row.get('rechal', '')).upper() == 'Y'

    try:
        event_dt = pd.to_datetime(str(row.get('event_dt', '')), format='%Y%m%d', errors='coerce')
        start_dt = pd.to_datetime(str(row.get('start_dt', '')), format='%Y%m%d', errors='coerce')
        if pd.notna(event_dt) and pd.notna(start_dt):
            gap_days = (event_dt - start_dt).days
        else:
            gap_days = None
    except:
        gap_days = None

    indi = str(row.get('indi_pt', '')).lower().strip()
    ae = str(row.get('meddra_pt', '')).lower().strip()
    confound = (indi == ae) and indi != '' and indi != 'nan'
    
    # S3 FIX: Check concomitant drugs as alternative explanation
    n_concomitant = 0
    if 'n_concomitant' in row.index:
        try:
            n_concomitant = int(row.get('n_concomitant', 0) or 0)
        except (ValueError, TypeError):
            n_concomitant = 0
    has_alternative = confound or (n_concomitant > 3)
    
    # Plausible temporal relationship
    plausible_time = gap_days is not None and 0 <= gap_days < 365
    strong_time = gap_days is not None and 0 <= gap_days < 90
    improbable_time = gap_days is not None and gap_days > 730
    
    # H3 FIX: Negative dechallenge is active evidence AGAINST causality.
    # Previously treated same as missing — now downgrades classification.
    if neg_dechal and not rechal and (not plausible_time or has_alternative):
        return 'Unlikely'
    
    # Insufficient data
    if gap_days is None and not dechal and not rechal and not neg_dechal:
        return 'Unassessable'
    
    # S3 FIX: Certain requires ALL of: plausible time + dechal + rechal + NO alternatives
    if rechal and dechal and strong_time and not has_alternative:
        return 'Certain'
    
    # Probable: plausible time + dechal (or rechal) + no strong alternatives
    if dechal and plausible_time and not has_alternative:
        return 'Probable'
    
    # S3 FIX: Even with confound, positive dechal+rechal overrides to at least Possible
    # (Previous bug: confound immediately made everything Unlikely)
    if (dechal or rechal) and plausible_time and has_alternative:
        return 'Possible'  # Evidence for AND against — "Possible" is correct
    
    # Unlikely: improbable temporal relationship OR strong alternative AND no supporting evidence
    if improbable_time and not dechal and not rechal:
        return 'Unlikely'
    if has_alternative and not dechal and not rechal and not strong_time:
        return 'Unlikely'
    
    # Possible: plausible time but incomplete evidence
    if plausible_time:
        return 'Possible'
    
    # Conditional: some data exists but not enough for assessment
    if gap_days is not None or dechal or rechal:
        return 'Conditional'
    
    return None


def build_t4_pairs(df: pd.DataFrame, max_pairs: int = 8000) -> list[dict]:
    """Build Task 4 (Causality) training pairs — CLINICAL NARRATIVE.

    KEY CHANGE: Convert structured fields to clinical narrative. The model
    must EXTRACT evidence from the text, not read pre-formatted fields.

    Input: "A 68-year-old woman was started on warfarin... When the drug was
           discontinued, the symptoms resolved..."
    NOT: "Dechallenge: Y. Rechallenge: N. Temporal: 45 days."
    """
    print("\n  📋 Task 4: Causality Assessment (CLINICAL NARRATIVE)...")

    # H2 FIX: Don't dedup on drug_seq — fillna('') collapses all NaN entries
    # into one per primaryid, silently discarding concomitant drugs.
    # Instead, dedup on primaryid+drugname (the actual unique drug per case).
    case_drugs = df.drop_duplicates(subset=['primaryid', 'drugname'])
    case_drugs = case_drugs[case_drugs['drugname'].notna() & case_drugs['meddra_pt'].notna()]

    case_drugs['causality'] = case_drugs.apply(compute_causality, axis=1)
    labeled = case_drugs[case_drugs['causality'].notna()]

    print(f"     Causality distribution:")
    for level, count in labeled['causality'].value_counts().items():
        print(f"       {level}: {count:,}")

    if len(labeled) > max_pairs * 3:
        labeled = labeled.sample(n=max_pairs * 3, random_state=SEED)

    pairs = []
    random.seed(SEED)

    for _, row in labeled.iterrows():
        causality = row['causality']
        drug = row.get('drugname', 'Unknown')
        ae = row.get('meddra_pt', 'Unknown')
        dechal = str(row.get('dechal', '')).upper()
        rechal = str(row.get('rechal', '')).upper()
        indication = row.get('indi_pt', 'Unknown')
        age = _age_str(row)
        gender = {'M': 'male', 'F': 'female'}.get(str(row.get('gndr_cod', '')), 'patient')

        # Temporal gap computation
        try:
            event_dt = pd.to_datetime(str(row.get('event_dt', '')), format='%Y%m%d', errors='coerce')
            start_dt = pd.to_datetime(str(row.get('start_dt', '')), format='%Y%m%d', errors='coerce')
            gap_days = (event_dt - start_dt).days if pd.notna(event_dt) and pd.notna(start_dt) else None
        except:
            gap_days = None

        # Concomitant drugs
        concomitant_count = int(row.get('n_concomitant', 0)) if 'n_concomitant' in row.index else 0

        # Indication-AE overlap
        indi_lower = str(indication).lower().strip()
        ae_lower = str(ae).lower().strip()
        confound = (indi_lower == ae_lower) and indi_lower != '' and indi_lower != 'nan'

        # === KEY CHANGE: Build clinical narrative (not structured fields) ===
        age_prefix = f"A {age} year-old" if age != 'unknown-age' else "A"
        narrative_parts = [
            f"{age_prefix} {gender} was prescribed {drug} for the treatment of {indication}."
        ]

        # Temporal relationship as narrative
        if gap_days is not None:
            if gap_days < 1:
                narrative_parts.append(f"On the same day that {drug} was started, the patient developed {ae}.")
            elif gap_days < 7:
                narrative_parts.append(f"Within {gap_days} days of starting {drug}, the patient developed {ae}.")
            elif gap_days < 30:
                weeks = gap_days // 7
                narrative_parts.append(f"Approximately {weeks} week{'s' if weeks > 1 else ''} after initiating {drug} therapy, the patient experienced {ae}.")
            elif gap_days < 180:
                months = gap_days // 30
                narrative_parts.append(f"After approximately {months} month{'s' if months > 1 else ''} on {drug}, the patient developed {ae}.")
            elif gap_days < 365:
                narrative_parts.append(f"Several months into {drug} treatment, the patient reported {ae}.")
            else:
                years = gap_days // 365
                narrative_parts.append(f"After more than {years} year{'s' if years > 1 else ''} of {drug} therapy, the patient experienced {ae}.")
        else:
            narrative_parts.append(f"The patient developed {ae} during treatment with {drug}. The exact temporal relationship is not documented.")

        # Dechallenge as narrative
        if dechal == 'Y':
            narrative_parts.append(random.choice([
                f"When {drug} was discontinued, the symptoms of {ae} gradually resolved.",
                f"The adverse event resolved after {drug} was stopped.",
                f"Following discontinuation of {drug}, the patient's {ae.lower()} improved significantly.",
            ]))
        elif dechal == 'N':
            narrative_parts.append(random.choice([
                f"Despite discontinuation of {drug}, the symptoms persisted.",
                f"The adverse event continued even after {drug} was stopped.",
            ]))
        else:
            narrative_parts.append("No information is available regarding whether the drug was discontinued.")

        # Rechallenge as narrative
        if rechal == 'Y':
            narrative_parts.append(random.choice([
                f"Notably, when {drug} was later reintroduced, the {ae.lower()} recurred.",
                f"Upon rechallenge with {drug}, the same adverse event reappeared.",
            ]))

        # Concomitant drugs
        if concomitant_count > 0:
            narrative_parts.append(f"The patient was also receiving {concomitant_count} other concomitant medication{'s' if concomitant_count > 1 else ''}.")
        else:
            narrative_parts.append("No concomitant medications were reported.")

        # Indication confound
        if confound:
            narrative_parts.append(f"Of note, the reported adverse event ({ae}) is similar to the condition for which {drug} was prescribed ({indication}).")

        input_text = ' '.join(narrative_parts) + f"\n\nAssess the causal relationship between {drug} and {ae} using WHO-UMC criteria."

        # === DIVERSIFIED THINKING TRACE (narrative, not checklist) ===
        think_opener = _diversify(random.choice(T4_THINK_OPENERS))

        # Temporal evidence
        if gap_days is not None:
            gap_str = str(gap_days)
            if gap_days < 30:
                temporal_template = random.choice(T4_TEMPORAL['strong'])
            elif gap_days < 365:
                temporal_template = random.choice(T4_TEMPORAL['plausible'])
            else:
                temporal_template = random.choice(T4_TEMPORAL['weak'])
            temporal_text = _diversify(temporal_template.replace('{gap}', gap_str))
        else:
            temporal_text = _diversify(random.choice(T4_TEMPORAL['missing']))

        # Dechallenge evidence
        if dechal == 'Y':
            dechal_text = _diversify(random.choice(T4_DECHALLENGE['positive']).replace('{drug}', drug))
        elif dechal == 'N':
            dechal_text = _diversify(random.choice(T4_DECHALLENGE['negative']).replace('{drug}', drug))
        else:
            dechal_text = _diversify(random.choice(T4_DECHALLENGE['unknown']))

        # Rechallenge evidence
        if rechal == 'Y':
            rechal_text = _diversify(random.choice(T4_RECHALLENGE['positive']).replace('{drug}', drug))
        else:
            rechal_text = _diversify(random.choice(T4_RECHALLENGE['none']))

        # Confounding
        if confound:
            confound_text = _diversify(random.choice(T4_CONFOUND['yes']).replace('{ae}', ae).replace('{indication}', str(indication)))
        else:
            confound_text = _diversify(random.choice(T4_CONFOUND['no']))

        # Concomitant
        if concomitant_count > 3:
            concom_text = _diversify(random.choice(T4_CONCOMITANT['many']).replace('{n}', str(concomitant_count)))
        elif concomitant_count > 0:
            concom_text = _diversify(random.choice(T4_CONCOMITANT['few']).replace('{n}', str(concomitant_count)))
        else:
            concom_text = _diversify(random.choice(T4_CONCOMITANT['none']))

        # Verdict
        verdict_text = _diversify(random.choice(T4_VERDICTS.get(causality, T4_VERDICTS['Unassessable'])).replace('{support}', random.choice(VOCAB['supports'])))

        # Assemble thinking trace as flowing narrative
        think = f"{think_opener} {temporal_text} {dechal_text} {rechal_text} {confound_text} {concom_text} {verdict_text}"

        # Build structured answer with evidence summary
        evidence_lines = [
            f"Temporal: {temporal_text}",
            f"Dechallenge: {dechal_text}",
            f"Rechallenge: {rechal_text}",
            f"Confounders: {confound_text}",
            f"Alternatives: {concom_text}",
        ]
        answer = f"WHO-UMC Causality: {causality}\nEvidence:\n" + "\n".join(f"  - {e}" for e in evidence_lines)

        pairs.append({
            "task": "T4",
            "primaryid": str(row.get('primaryid', '')),
            "thinking": True,
            "messages": [
                {"role": "system", "content": "You are a pharmacovigilance expert. Read the clinical case narrative carefully, extract the relevant evidence (temporal relationship, dechallenge, rechallenge, concomitant medications, confounding factors), and assess drug-event causality using WHO-UMC criteria: Certain, Probable, Possible, Unlikely, Conditional, Unassessable. Think step by step."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": f"<|channel>thought\n{think}\n<channel|>\n{answer}"}
            ]
        })

    random.seed(SEED)
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    print(f"     Final T4 pairs: {len(pairs):,}")
    return pairs


# ============================================================
# TEXT FIELD BUILDER (for SFTTrainer compatibility)
# ============================================================

def build_text_field(pair: dict) -> str:
    """Convert messages format to a single text string for SFTTrainer."""
    messages = pair.get('messages', [])
    parts = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
        elif role == 'user':
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == 'assistant':
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    return "\n".join(parts)


# ============================================================
# DECONTAMINATION (MeditronFO-adopted)
# ============================================================

def decontaminate(pairs: list[dict], eval_ratio: float = EVAL_HOLDOUT_RATIO) -> tuple[list[dict], list[dict]]:
    """Split data into train/eval with hash-verified decontamination."""
    print(f"\n  🔒 Decontamination (MeditronFO-adopted)...")

    all_ids = list(set(p.get('primaryid', str(i)) for i, p in enumerate(pairs)))
    random.shuffle(all_ids)

    eval_count = max(1, int(len(all_ids) * eval_ratio))
    eval_ids = set(all_ids[:eval_count])

    eval_hashes = {hashlib.sha256(pid.encode()).hexdigest()[:16] for pid in eval_ids}

    train_pairs = [p for p in pairs if p.get('primaryid', '') not in eval_ids]
    eval_pairs = [p for p in pairs if p.get('primaryid', '') in eval_ids]

    train_hashes = {hashlib.sha256(p.get('primaryid', '').encode()).hexdigest()[:16] for p in train_pairs}
    overlap = train_hashes & eval_hashes

    print(f"     Train cases: {len(all_ids) - eval_count:,} | Eval cases: {eval_count:,}")
    print(f"     Train pairs: {len(train_pairs):,} | Eval pairs: {len(eval_pairs):,}")
    print(f"     Hash overlap: {len(overlap)} (must be 0) {'✅' if len(overlap) == 0 else '❌ CONTAMINATION DETECTED'}")

    return train_pairs, eval_pairs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Training Data Builder — GROUND-ZERO REBUILD")
    print("  All 4 tasks redesigned for genuine difficulty")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    # Load preprocessed FAERS data
    parquet_path = PROCESSED_DIR / "faers_master.parquet"
    csv_path = PROCESSED_DIR / "faers_master.csv"

    if parquet_path.exists():
        df = pd.read_parquet(str(parquet_path))
        print(f"\n  📂 Loaded: {len(df):,} rows from {parquet_path.name}")
    elif csv_path.exists():
        df = pd.read_csv(str(csv_path), dtype=str)
        print(f"\n  📂 Loaded: {len(df):,} rows from {csv_path.name}")
    else:
        print(f"  ❌ Not found: {parquet_path} or {csv_path}")
        print(f"     Run first: python src/data/02_preprocess.py")
        return 1

    # Build pairs for each redesigned task
    all_pairs = []

    t1_pairs = build_t1_pairs(df)
    all_pairs.extend(t1_pairs)

    t2_pairs = build_t2_pairs(df)
    all_pairs.extend(t2_pairs)

    t3_pairs = build_t3_pairs(df)
    all_pairs.extend(t3_pairs)

    t4_pairs = build_t4_pairs(df)
    all_pairs.extend(t4_pairs)

    # Decontamination
    train_pairs, eval_pairs = decontaminate(all_pairs)

    random.shuffle(train_pairs)

    # Add 'text' field for SFTTrainer
    for pair in train_pairs:
        pair['text'] = build_text_field(pair)
    for pair in eval_pairs:
        pair['text'] = build_text_field(pair)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_FILE), 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    with open(str(EVAL_FILE), 'w', encoding='utf-8') as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # Decontamination log
    with open(str(DECONTAM_LOG), 'w', encoding='utf-8') as f:
        f.write(f"Decontamination Report — {pd.Timestamp.now()}\n")
        f.write(f"GROUND-ZERO REBUILD — All tasks redesigned\n")
        f.write(f"Total pairs: {len(all_pairs)}\n")
        f.write(f"Training pairs: {len(train_pairs)}\n")
        f.write(f"Evaluation pairs: {len(eval_pairs)}\n")
        f.write(f"Holdout ratio: {EVAL_HOLDOUT_RATIO}\n")
        eval_ids = set(p.get('primaryid', '') for p in eval_pairs)
        f.write(f"Eval case IDs: {len(eval_ids)}\n")
        for pid in sorted(eval_ids):
            h = hashlib.sha256(pid.encode()).hexdigest()[:16]
            f.write(f"  {pid} → {h}\n")

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING DATA SUMMARY (GROUND-ZERO REBUILD)")
    print("=" * 60)

    task_counts = Counter()
    for p in train_pairs:
        task_counts[p.get('task', 'unknown')] += 1

    task_descriptions = {
        'T1': 'Seriousness (DIVERSIFIED — narrative-based, combinatorial think/answer)',
        'T2': 'MedDRA Coding (DIVERSIFIED — lay/clinical language, varied rationales)',
        'T3': 'Labelling (DIVERSIFIED — class-aware, combinatorial reasoning)',
        'T4': 'Causality (DIVERSIFIED — narrative evidence, combinatorial assembly)',
    }

    for task, count in sorted(task_counts.items()):
        desc = task_descriptions.get(task, task)
        print(f"  {task}: {count:,} pairs — {desc}")
    print(f"  {'─' * 50}")
    print(f"  TOTAL training: {len(train_pairs):,} pairs")
    print(f"  TOTAL eval:     {len(eval_pairs):,} pairs")

    if OUTPUT_FILE.exists():
        print(f"\n  💾 Training: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1e6:.1f} MB)")
    print(f"  💾 Eval:     {EVAL_FILE}")
    print(f"  📋 Decontam: {DECONTAM_LOG}")

    # Diversity report — verify combinatorial engine effectiveness
    print(f"\n  📊 DIVERSITY REPORT (Combinatorial Engine)")
    for task in ['T1', 'T2', 'T3', 'T4']:
        task_pairs = [p for p in train_pairs if p.get('task') == task]
        if not task_pairs:
            continue
        completions = [p['messages'][-1]['content'] for p in task_pairs]
        unique = len(set(completions))
        pct = 100 * unique / len(completions) if completions else 0
        print(f"     {task}: {unique:,} unique / {len(completions):,} total ({pct:.1f}%)")

    print(f"\n  ✅ Training data ready (DIVERSIFIED)! Next step:")
    print(f"     python src/training/01_sft_train.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
