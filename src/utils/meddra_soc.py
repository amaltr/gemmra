"""
MedDRA SOC (System Organ Class) approximate classifier.

MedDRA's hierarchy is proprietary (MSSO license). This module provides a
keyword-based heuristic that maps a MedDRA Preferred Term string to its
most likely SOC. Used for:
  - T2 eval: partial credit when model picks correct organ system
  - GRPO correctness_reward: softer gradient signal for T2

Accuracy: ~85% on common PTs. Good enough for partial-credit scoring.
Not a substitute for the real MedDRA hierarchy.

Also contains medical synonym groups for common PT equivalences.
"""

import re

# ============================================================
# 27 MedDRA SOCs with keyword patterns
# ============================================================
# Each SOC has a list of regex patterns. Matched case-insensitively.
# Order matters: first match wins. More specific patterns first.

SOC_PATTERNS: dict[str, list[str]] = {
    "Hepatobiliary disorders": [
        r"hepat", r"liver", r"jaundice", r"cholest", r"biliary", r"bilirubin",
        r"cirrhosis", r"cholangitis", r"gallbladder", r"gallstone",
    ],
    "Cardiac disorders": [
        r"cardiac", r"cardiomyopath", r"arrhythmi", r"tachycardi", r"bradycardi",
        r"atrial", r"ventricular", r"palpitation", r"myocardial", r"angina",
        r"coronary", r"heart\s+failure", r"qt\s+prolong", r"torsade",
        r"fibrillation", r"flutter", r"heart", r"cardio",
    ],
    "Nervous system disorders": [
        r"headache", r"seizure", r"convulsion", r"neuropath", r"neuralgi",
        r"tremor", r"ataxia", r"dyskines", r"encephalopath", r"\bcoma\b",
        r"syncope", r"dizziness", r"paraesthes", r"paresthes", r"paralys",
        r"stroke", r"cerebr", r"neurotoxic", r"cognitive", r"amnesia",
        r"somnolence", r"stupor", r"meningism", r"aphasia", r"dysarthria",
        r"hemipar", r"quadripar", r"guillain",
    ],
    "Gastrointestinal disorders": [
        r"nausea", r"vomiting", r"diarrh", r"constipation", r"abdominal",
        r"gastric", r"gastrointestinal", r"colitis", r"pancreatit",
        r"dyspepsia", r"gastritis", r"intestinal", r"bowel", r"rectal",
        r"oesophag", r"esophag", r"ileus", r"stomatitis", r"dysphagia",
        r"\boral\b", r"mouth\s+ulcer", r"peritonit", r"ascites",
    ],
    "Respiratory, thoracic and mediastinal disorders": [
        r"respiratory", r"pulmonary", r"dyspn[oe]", r"\bcough\b", r"bronch",
        r"pneumonitis", r"pleural", r"\basthma\b", r"apn[oe]a",
        r"interstitial\s+lung", r"wheez", r"stridor", r"pharyngit",
        r"rhinit", r"epistaxis", r"h[ae]moptysis", r"pulmonary\s+fibros",
        r"acute\s+respiratory", r"ards",
    ],
    "Skin and subcutaneous tissue disorders": [
        r"\brash\b", r"pruritus", r"dermatit", r"alopecia", r"erythema",
        r"vesicl", r"blister", r"bull[oa]", r"stevens.johnson", r"toxic\s+epidermal",
        r"photosensitiv", r"\bskin\b", r"cutaneous", r"\bacne\b", r"psoriasis",
        r"eczema", r"exanthem", r"desquamat", r"pemphig", r"lichenoid",
        r"purpur[ai]", r"petechiae",
    ],
    "Renal and urinary disorders": [
        r"renal", r"kidney", r"nephr", r"urinary", r"oliguri", r"anuria",
        r"proteinuri", r"h[ae]maturi", r"cystitis", r"incontinen",
        r"retention", r"dialysis", r"glomerul", r"nephrotic", r"tubul",
    ],
    "Blood and lymphatic system disorders": [
        r"an[ae]mi", r"thrombocytopen", r"neutropen", r"leukopen",
        r"pancytopen", r"lymphopen", r"agranulocyt", r"coagulopath",
        r"platelet", r"leukocyt", r"granulocyt", r"lymphocyt",
        r"h[ae]molyt", r"myelosuppress", r"bone\s+marrow", r"splen",
        r"lymphaden", r"disseminated\s+intravascular", r"\bdic\b",
        r"aplastic", r"polycyth", r"eosinophil",
    ],
    "Immune system disorders": [
        r"anaphyla", r"hypersensitiv", r"allerg", r"autoimmune",
        r"angioedema", r"serum\s+sickness", r"cytokine\s+release",
        r"graft.vs.host", r"immune\s+reconstitution",
    ],
    "Infections and infestations": [
        r"infection", r"\bsepsis\b", r"septic", r"pneumonia", r"meningitis",
        r"cellulitis", r"abscess", r"tuberculosis", r"\bherpes\b",
        r"candid", r"fungal", r"bacterial", r"viral", r"mycobacter",
        r"endocardit", r"osteomyelit", r"histoplasm",
    ],
    "Musculoskeletal and connective tissue disorders": [
        r"muscle", r"myalgia", r"myopath", r"rhabdomyolysis", r"arthralg",
        r"arthrit", r"tendon", r"\bbone\b", r"osteo", r"musculoskeletal",
        r"back\s+pain", r"\bjoint\b", r"spasm", r"myosit", r"lupus",
        r"fibromyalg",
    ],
    "Metabolism and nutrition disorders": [
        r"hypokala?emi", r"hyperkala?emi", r"hyponatr", r"hypernatr",
        r"hypoglycemi", r"hyperglycemi", r"acidosis", r"alkalosis",
        r"dehydrat", r"anorexia", r"appetite", r"metabol",
        r"lactic", r"ketoacidosis", r"hypercalc", r"hypocalc",
        r"\bgout\b", r"hypomagnes", r"hypermagnes", r"hypophosphat",
    ],
    "Eye disorders": [
        r"\beye\b", r"ocular", r"visual", r"vision", r"cataract",
        r"glaucoma", r"retinal", r"retinopathy", r"macular", r"optic",
        r"blindness", r"diplopia", r"photophobia", r"uveitis",
        r"conjunctiv", r"corneal", r"keratit", r"intraocular", r"vitreous",
    ],
    "Psychiatric disorders": [
        r"depression", r"anxiety", r"insomnia", r"psychos[ie]",
        r"hallucination", r"confusion", r"delirium", r"agitat",
        r"suicid", r"\bmania\b", r"bipolar", r"psychiatric",
        r"nightmare", r"psychotic", r"paranoi", r"obsessive",
    ],
    "Endocrine disorders": [
        r"thyroid", r"hypothyroid", r"hyperthyroid", r"adrenal",
        r"cushing", r"pituitary", r"endocrine", r"goit[re]",
        r"thyrotoxic", r"addison", r"hyperaldoster",
    ],
    "Vascular disorders": [
        r"hypertension", r"hypotension", r"thrombos[ie]", r"emboli",
        r"phlebitis", r"vasculitis", r"raynaud", r"vasoconstric",
        r"vasodilat", r"isch[ae]mi", r"deep\s+vein", r"\bdvt\b",
        r"vascular", r"h[ae]morrhag", r"bleeding", r"haematoma",
        r"hematoma",
    ],
    "Neoplasms benign, malignant and unspecified": [
        r"neoplasm", r"cancer", r"carcinoma", r"lymphoma", r"leuk[ae]mi",
        r"tumo[ur]", r"malignant", r"benign", r"sarcoma", r"melanoma",
        r"myeloma", r"mesothelioma", r"metast",
    ],
    "Pregnancy, puerperium and perinatal conditions": [
        r"pregnan", r"prenatal", r"perinatal", r"abortion",
        r"miscarriage", r"stillbirth", r"gestational", r"ectopic",
        r"preeclampsia", r"f[oe]+tal", r"neonatal",
    ],
    "Congenital, familial and genetic disorders": [
        r"congenital", r"teratogen", r"birth\s+defect", r"malformation",
        r"\bcleft\b", r"spina\s+bifida", r"chromosom", r"syndrome.*fetal",
        r"fetal.*syndrome",
    ],
    "Ear and labyrinth disorders": [
        r"\bear\b", r"tinnitus", r"hearing", r"deafness", r"otitis",
        r"ototoxic", r"labyrinth", r"vestibular", r"acoustic",
    ],
    "Reproductive system and breast disorders": [
        r"\bbreast\b", r"gynaecoma", r"gynecoma", r"sexual",
        r"menstrual", r"amenorrh", r"erectile", r"impotence",
        r"vaginal", r"uterine", r"ovarian", r"testicular", r"prostat",
        r"infertilit", r"galactorrh", r"dysmenorrh", r"swelling",
    ],
    "General disorders and administration site conditions": [
        r"fatigue", r"pyrexia", r"\bfever\b", r"malaise", r"[oe]edema",
        r"asthenia", r"\bdeath\b", r"injection\s+site", r"infusion\s+site",
        r"chills", r"peripheral\s+(o?edema|swelling)", r"drug\s+ineffect",
        r"therapeutic.*response", r"condition\s+aggravat",
        r"growth\s+retard", r"failure\s+to\s+thrive", r"disease\s+progression",
    ],
    "Investigations": [
        r"increased$", r"decreased$", r"elevated", r"abnormal",
        r"transaminase", r"\balt\b", r"\bast\b", r"creatinine",
        r"\bcount\b", r"laboratory", r"\becg\b", r"\beeg\b",
        r"blood\s+pressure", r"weight\s+(gain|loss|increas|decreas)",
        r"enzyme\s+increas",
    ],
    "Injury, poisoning and procedural complications": [
        r"overdose", r"poisoning", r"\bfall\b", r"fracture",
        r"procedural", r"complication", r"\bwound\b", r"\bburn\b",
        r"contusion", r"laceration", r"drug\s+interaction",
    ],
    "Surgical and medical procedures": [
        r"transplant", r"surgery", r"amputation", r"biopsy",
        r"implant.*remov", r"catheter", r"transfusion",
    ],
    "Social circumstances": [
        r"disability", r"bedridden", r"unemploy",
    ],
    "Product issues": [
        r"product", r"contamination", r"device\s+malfunction",
    ],
}

# Compile patterns for performance
_COMPILED_SOC: list[tuple[str, re.Pattern]] = []
for soc, patterns in SOC_PATTERNS.items():
    combined = "|".join(f"(?:{p})" for p in patterns)
    _COMPILED_SOC.append((soc, re.compile(combined, re.IGNORECASE)))


def classify_soc(pt: str) -> str | None:
    """Map a MedDRA PT string to its approximate SOC.
    
    Returns SOC name or None if no match found.
    """
    if not pt or len(pt.strip()) < 2:
        return None
    pt_clean = pt.strip()
    for soc, pattern in _COMPILED_SOC:
        if pattern.search(pt_clean):
            return soc
    return None


def soc_match(pt1: str, pt2: str) -> bool:
    """Check if two MedDRA PTs belong to the same SOC."""
    soc1 = classify_soc(pt1)
    soc2 = classify_soc(pt2)
    if soc1 is None or soc2 is None:
        return False
    return soc1 == soc2


# ============================================================
# Medical synonym groups (common PT equivalences)
# ============================================================
# Each group contains PTs that describe the same clinical concept.
# If model outputs any term in the same group as GT, it's a match.

SYNONYM_GROUPS: list[set[str]] = [
    # Hepatic
    {"hepatotoxicity", "drug-induced liver injury", "liver injury", "hepatic injury",
     "toxic hepatitis", "hepatic failure", "liver failure", "hepatic damage", "liver damage"},
    {"hepatic enzyme increased", "transaminases increased", "liver function test abnormal",
     "alt increased", "ast increased", "liver enzymes elevated"},
    {"jaundice", "icterus", "hyperbilirubinaemia", "hyperbilirubinemia"},
    {"cholestasis", "cholestatic hepatitis"},
    
    # Cardiac
    {"cardiac arrest", "cardiopulmonary arrest", "asystole"},
    {"myocardial infarction", "heart attack", "acute myocardial infarction", "acute coronary syndrome"},
    {"heart failure", "cardiac failure", "congestive heart failure", "congestive cardiac failure"},
    {"atrial fibrillation", "auricular fibrillation"},
    
    # Nervous
    {"headache", "cephalgia", "cephalalgia"},
    {"seizure", "convulsion", "epilepsy", "epileptic seizure"},
    {"cerebrovascular accident", "stroke", "cerebral infarction", "brain infarction"},
    {"peripheral neuropathy", "polyneuropathy", "neuritis peripheral"},
    {"dizziness", "vertigo", "lightheadedness"},
    {"syncope", "loss of consciousness", "fainting"},
    
    # GI
    {"nausea", "nausea and vomiting"},
    {"diarrhoea", "diarrhea"},
    {"oedema", "edema", "swelling"},
    {"abdominal pain", "abdominal pain upper", "abdominal pain lower", "stomach ache"},
    {"gastrointestinal haemorrhage", "gastrointestinal hemorrhage", "gi bleed", "gi bleeding"},
    
    # Respiratory
    {"dyspnoea", "dyspnea", "breathlessness", "shortness of breath"},
    {"pulmonary embolism", "lung embolism"},
    {"pneumonia", "lung infection", "pulmonary infection"},
    {"interstitial lung disease", "pulmonary fibrosis", "lung fibrosis"},
    
    # Skin
    {"rash", "skin rash", "exanthem", "eruption"},
    {"urticaria", "hives"},
    {"stevens-johnson syndrome", "sjs", "toxic epidermal necrolysis", "ten",
     "stevens johnson syndrome"},
    {"alopecia", "hair loss"},
    
    # Blood
    {"anaemia", "anemia"},
    {"thrombocytopenia", "low platelet count", "platelet count decreased"},
    {"neutropenia", "neutrophil count decreased"},
    {"pancytopenia", "bone marrow failure"},
    {"disseminated intravascular coagulation", "dic", "coagulopathy"},
    
    # Renal
    {"renal failure", "kidney failure", "renal insufficiency", "kidney injury",
     "acute kidney injury", "acute renal failure"},
    {"nephrotoxicity", "renal toxicity", "kidney damage"},
    
    # Immune
    {"anaphylaxis", "anaphylactic reaction", "anaphylactic shock", "anaphylactoid reaction"},
    {"hypersensitivity", "allergic reaction", "drug hypersensitivity"},
    {"angioedema", "angioneurotic oedema", "angioneurotic edema"},
    
    # Metabolic
    {"hypoglycaemia", "hypoglycemia", "low blood sugar"},
    {"hyperglycaemia", "hyperglycemia", "high blood sugar", "blood glucose increased"},
    {"diabetic ketoacidosis", "ketoacidosis", "dka"},
    {"lactic acidosis", "lactate increased"},
    
    # Vascular
    {"deep vein thrombosis", "dvt", "venous thrombosis"},
    {"hypertension", "blood pressure increased", "high blood pressure"},
    {"hypotension", "blood pressure decreased", "low blood pressure"},
    {"haemorrhage", "hemorrhage", "bleeding"},
    
    # Musculoskeletal
    {"myalgia", "muscle pain", "muscular pain"},
    {"arthralgia", "joint pain"},
    {"rhabdomyolysis", "myoglobinuria"},
    
    # Pregnancy/fetal
    {"foetal exposure during pregnancy", "fetal exposure during pregnancy",
     "foetal exposure", "drug exposure during pregnancy"},
    {"fetal valproate syndrome", "valproate syndrome", "foetal valproate syndrome"},
    {"growth retardation", "intrauterine growth retardation", "fetal growth restriction",
     "growth restriction"},
    
    # Psychiatric
    {"depression", "depressed mood", "major depression"},
    {"insomnia", "sleep disorder", "sleeplessness"},
    {"suicidal ideation", "suicidal thoughts", "suicidality"},
    
    # Eye
    {"cataract", "lens opacity"},
    {"macular oedema", "macular edema", "cystoid macular oedema", "cystoid macular edema"},
    
    # General
    {"death", "fatal outcome", "fatal"},
    {"drug ineffective", "lack of efficacy", "therapeutic response decreased",
     "drug ineffective for unapproved indication"},
    {"pyrexia", "fever", "body temperature increased"},
]

# Build lookup: term → group index for O(1) matching
_SYNONYM_LOOKUP: dict[str, int] = {}
for idx, group in enumerate(SYNONYM_GROUPS):
    for term in group:
        _SYNONYM_LOOKUP[term.lower().strip()] = idx


def synonym_match(pt1: str, pt2: str) -> bool:
    """Check if two PTs are medical synonyms."""
    p1 = pt1.lower().strip()
    p2 = pt2.lower().strip()
    if p1 == p2:
        return True
    g1 = _SYNONYM_LOOKUP.get(p1)
    g2 = _SYNONYM_LOOKUP.get(p2)
    if g1 is not None and g2 is not None and g1 == g2:
        return True
    return False


def compute_t2_similarity(pred_pt: str, true_pt: str) -> float:
    """Compute similarity score between predicted and true MedDRA PT.
    
    Returns:
        1.0 — exact match
        0.9 — medical synonym match
        0.7 — substring/word-overlap match (existing fuzzy)
        0.5 — same SOC (correct organ system)
        0.0 — no match
    """
    p = pred_pt.lower().strip()
    t = true_pt.lower().strip()
    
    if not p or not t:
        return 0.0
    
    # Level 1: Exact match
    if p == t:
        return 1.0
    
    # Level 2: Synonym match
    if synonym_match(p, t):
        return 0.9
    
    # Level 3: Substring or word overlap ≥ 50%
    if t in p or p in t:
        return 0.7
    t_words = set(t.split())
    p_words = set(p.split())
    overlap = len(t_words & p_words)
    total = max(len(t_words | p_words), 1)
    if overlap / total >= 0.5:
        return 0.7
    
    # Level 4: Same SOC
    if soc_match(p, t):
        return 0.5
    
    return 0.0
