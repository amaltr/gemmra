"""
Step 5: Download and parse external datasets (CADECv2, PHEE, BioDEX).
Run this on CPU (no GPU needed). Requires: pip install datasets

These are the datasets we CLAIM in our documentation but never actually used.
This script makes them real.

Output: data/external/external_pairs.jsonl
  - Unified format for integration into 03_build_training_data.py
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

EXTERNAL_DIR = Path("data/external")
OUTPUT_FILE = EXTERNAL_DIR / "external_pairs.jsonl"

# ============================================================
# PHEE — Pharmacovigilance Event Extraction (HuggingFace)
# ============================================================

def download_phee() -> list[dict]:
    """Download and parse PHEE dataset from HuggingFace.
    
    PHEE contains 5,000+ annotated pharmacovigilance events from medical
    case reports with hierarchical annotations for Subject, Treatment, Effect.
    
    Source: sarus-tech/phee (HuggingFace)
    Paper: EMNLP 2022 — "PHEE: A Dataset for Pharmacovigilance Event Extraction from Text"
    """
    print("\n  📦 Downloading PHEE dataset...")
    
    try:
        from datasets import load_dataset
        # sarus-tech/phee uses deprecated script loading.
        # Use chufangao/PHEE-NER which has Parquet format.
        ds = load_dataset("chufangao/PHEE-NER")
    except Exception as e:
        print(f"     ❌ Failed to load PHEE-NER: {e}")
        try:
            # Fallback: try direct parquet loading
            ds = load_dataset("sarus-tech/phee")
        except Exception as e2:
            print(f"     ❌ Also failed: {e2}")
            return []
    
    pairs = []
    stats = Counter()
    
    for split_name in ['train', 'validation', 'test']:
        if split_name not in ds:
            continue
        split = ds[split_name]
        
        for example in split:
            try:
                # PHEE structure: each example has 'text' (sentence) and 'entities'/'events'
                text = example.get('text', '') or example.get('sentence', '')
                if not text or len(text.strip()) < 20:
                    continue
                
                # Extract entities from PHEE's annotation format
                entities = example.get('entities', [])
                events = example.get('events', [])
                
                # Try to extract drug and ADE from entities
                drugs = []
                effects = []
                subjects = []
                
                if isinstance(entities, list):
                    for ent in entities:
                        if isinstance(ent, dict):
                            etype = ent.get('type', '').lower()
                            etext = ent.get('text', '') or ent.get('span', '')
                            if 'treatment' in etype or 'drug' in etype:
                                if etext: drugs.append(etext)
                            elif 'effect' in etype or 'adverse' in etype:
                                if etext: effects.append(etext)
                            elif 'subject' in etype:
                                if etext: subjects.append(etext)
                
                # Also try flat field extraction (PHEE-NER format uses BIO tags)
                if not drugs and not effects:
                    # PHEE-NER has 'tokens' and 'ner_tags' columns
                    tokens = example.get('tokens', [])
                    ner_tags = example.get('ner_tags', []) or example.get('labels', [])
                    
                    if tokens and ner_tags and len(tokens) == len(ner_tags):
                        # BIO tag extraction
                        current_entity = []
                        current_type = None
                        for tok, tag in zip(tokens, ner_tags):
                            # Handle both int and string tags
                            tag_str = str(tag)
                            if tag_str.startswith('B-') or (isinstance(tag, int) and tag > 0 and tag % 2 == 1):
                                if current_entity and current_type:
                                    entity_text = ' '.join(current_entity)
                                    if 'drug' in str(current_type).lower() or 'treatment' in str(current_type).lower():
                                        drugs.append(entity_text)
                                    elif 'effect' in str(current_type).lower() or 'adverse' in str(current_type).lower():
                                        effects.append(entity_text)
                                current_entity = [tok]
                                current_type = tag_str
                            elif tag_str.startswith('I-') or (isinstance(tag, int) and tag > 0 and tag % 2 == 0):
                                current_entity.append(tok)
                            else:
                                if current_entity and current_type:
                                    entity_text = ' '.join(current_entity)
                                    if 'drug' in str(current_type).lower() or 'treatment' in str(current_type).lower():
                                        drugs.append(entity_text)
                                    elif 'effect' in str(current_type).lower() or 'adverse' in str(current_type).lower():
                                        effects.append(entity_text)
                                current_entity = []
                                current_type = None
                        # Flush last entity
                        if current_entity and current_type:
                            entity_text = ' '.join(current_entity)
                            if 'drug' in str(current_type).lower():
                                drugs.append(entity_text)
                            elif 'effect' in str(current_type).lower():
                                effects.append(entity_text)
                    
                    # Reconstruct text from tokens if not available
                    if not text and tokens:
                        text = ' '.join(str(t) for t in tokens)
                
                if not text:
                    continue
                
                # Create pairs for different tasks
                # T2: Clinical text → ADE extraction (useful for MedDRA coding)
                if effects:
                    for effect in effects:
                        if isinstance(effect, str) and len(effect.strip()) > 2:
                            pairs.append({
                                "source": "PHEE",
                                "task_type": "T2",
                                "text": text.strip(),
                                "ade_text": effect.strip(),
                                "drug": drugs[0] if drugs else "Unknown",
                                "split": split_name,
                            })
                            stats['T2'] += 1
                
                # T1/T4: Full case sentence (useful for seriousness/causality if enough context)
                if drugs and effects:
                    pairs.append({
                        "source": "PHEE",
                        "task_type": "case_sentence",
                        "text": text.strip(),
                        "drugs": drugs,
                        "effects": [e for e in effects if isinstance(e, str)],
                        "subjects": subjects,
                        "split": split_name,
                    })
                    stats['case_sentence'] += 1
                    
            except Exception as e:
                continue
    
    print(f"     ✅ PHEE loaded: {len(pairs)} pairs")
    for task, count in sorted(stats.items()):
        print(f"        {task}: {count}")
    return pairs


# ============================================================
# BioDEX — Biomedical Drug Event Extraction (HuggingFace)
# ============================================================

def download_biodex() -> list[dict]:
    """Download and parse BioDEX-Reactions from HuggingFace.
    
    BioDEX contains PubMed papers linked to drug safety reports.
    BioDEX-Reactions specifically has drug → reaction mappings with source text.
    
    Source: BioDEX/BioDEX-Reactions (HuggingFace)
    Paper: EMNLP 2023 — "BioDEX: Large-Scale Biomedical Adverse Drug Event Extraction"
    """
    print("\n  📦 Downloading BioDEX-Reactions dataset...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("BioDEX/BioDEX-Reactions")
    except Exception as e:
        print(f"     ❌ Failed to load BioDEX-Reactions: {e}")
        try:
            ds = load_dataset("BioDEX/raw_dataset")
        except Exception as e2:
            print(f"     ❌ Also failed with raw_dataset: {e2}")
            return []
    
    pairs = []
    stats = Counter()
    
    for split_name in ds.keys():
        split = ds[split_name]
        
        for example in split:
            try:
                # BioDEX-Reactions actual columns: title, abstract, reactions, reactions_unmerged
                # reactions is a comma-separated string of MedDRA terms
                reactions_str = example.get('reactions', '')
                if not reactions_str:
                    continue
                
                # Parse comma-separated reactions
                reactions = [r.strip() for r in reactions_str.split(',') if r.strip()]
                if not reactions:
                    continue
                
                # Source text from abstract
                abstract = example.get('abstract', '')
                title = example.get('title', '')
                
                # Try to extract drug name from title (common pattern: "Drug-induced...")
                drug = 'Unknown'
                if title:
                    # Simple heuristic: look for common drug-related patterns in title
                    import re as _re
                    drug_match = _re.search(r'(\w+)[-\s](?:induced|associated|related|caused)', title, _re.I)
                    if drug_match:
                        drug = drug_match.group(1)
                
                for reaction in reactions:
                    if len(reaction) < 3:
                        continue
                    
                    # T2: biomedical text → MedDRA reaction
                    pair = {
                        "source": "BioDEX",
                        "task_type": "T2",
                        "drug": drug,
                        "reaction_term": reaction,
                        "split": split_name,
                    }
                    if abstract:
                        # Keep full abstract — smart truncation happens in
                        # 03_build_training_data.py (H1 fix centers window
                        # around PT mention). Blind [:500] here cut off the
                        # adverse event description in 92% of cases.
                        pair["source_text"] = abstract
                    if title:
                        pair["title"] = title[:200]
                    pairs.append(pair)
                    stats['T2'] += 1
                    
                    # T3: drug-reaction pair (from literature = likely labelled)
                    pairs.append({
                        "source": "BioDEX",
                        "task_type": "T3",
                        "drug": drug,
                        "reaction_term": reaction,
                        "is_labelled": True,
                        "evidence": "Published in biomedical literature",
                        "split": split_name,
                    })
                    stats['T3'] += 1
                    
            except Exception as e:
                continue
    
    print(f"     ✅ BioDEX loaded: {len(pairs)} pairs")
    for task, count in sorted(stats.items()):
        print(f"        {task}: {count}")
    return pairs


# ============================================================
# ADE Corpus v2 — Adverse Drug Event sentences (HuggingFace)
# ============================================================

def download_ade_corpus() -> list[dict]:
    """Download ADE Corpus v2 — sentence-level ADE annotations.
    
    Contains sentences from MEDLINE case reports classified as 
    ADE-related or not, with drug-ADE relation annotations.
    This provides clinical-register text for T2 training.
    
    Source: ade_corpus_v2 (HuggingFace)
    """
    print("\n  📦 Downloading ADE Corpus v2...")
    
    try:
        from datasets import load_dataset
        # Try the relation extraction split which has drug-ADE pairs
        ds = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
    except Exception as e:
        print(f"     ❌ Failed to load ADE Corpus v2: {e}")
        try:
            ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2",
                            "Ade_corpus_v2_drug_ade_relation")
        except Exception as e2:
            print(f"     ❌ Also failed: {e2}")
            return []
    
    pairs = []
    
    for split_name in ds.keys():
        split = ds[split_name]
        
        for example in split:
            try:
                text = example.get('text', '') or example.get('sentence', '')
                drug = example.get('drug', '') or example.get('Drug', '')
                effect = example.get('effect', '') or example.get('Adverse-Effect', '')
                
                if not text or not effect or len(effect.strip()) < 3:
                    continue
                
                pairs.append({
                    "source": "ADE_corpus",
                    "task_type": "T2",
                    "text": text.strip(),
                    "ade_text": effect.strip(),
                    "drug": drug.strip() if drug else "Unknown",
                    "split": split_name,
                })
                
            except Exception:
                continue
    
    print(f"     ✅ ADE Corpus v2: {len(pairs)} drug-ADE pairs")
    return pairs


# ============================================================
# Drug Class / Mechanism Mapping (Built-in)
# ============================================================

# Top ~100 FAERS drugs with pharmacological class and mechanism
# Used for T3 (labelling) to enable class-aware reasoning
DRUG_CLASS_MAP = {
    # Anticoagulants
    'warfarin': {'class': 'Anticoagulant (Vitamin K antagonist)', 'mechanism': 'Inhibits vitamin K-dependent clotting factors (II, VII, IX, X)', 'common_aes': ['haemorrhage', 'bleeding', 'bruising']},
    'rivaroxaban': {'class': 'Direct Oral Anticoagulant (Factor Xa inhibitor)', 'mechanism': 'Directly inhibits Factor Xa in the coagulation cascade', 'common_aes': ['bleeding', 'haemorrhage']},
    'apixaban': {'class': 'Direct Oral Anticoagulant (Factor Xa inhibitor)', 'mechanism': 'Directly inhibits Factor Xa', 'common_aes': ['bleeding', 'anaemia']},
    'heparin': {'class': 'Anticoagulant (Indirect thrombin inhibitor)', 'mechanism': 'Activates antithrombin III', 'common_aes': ['bleeding', 'thrombocytopenia']},
    # NSAIDs
    'aspirin': {'class': 'NSAID / Antiplatelet', 'mechanism': 'Irreversible COX-1/COX-2 inhibition', 'common_aes': ['gi bleeding', 'ulceration', 'tinnitus']},
    'ibuprofen': {'class': 'NSAID', 'mechanism': 'Non-selective COX-1/COX-2 inhibition', 'common_aes': ['gi upset', 'renal impairment', 'hypertension']},
    'naproxen': {'class': 'NSAID', 'mechanism': 'Non-selective COX inhibition', 'common_aes': ['gi bleeding', 'oedema']},
    'celecoxib': {'class': 'NSAID (COX-2 selective)', 'mechanism': 'Selective COX-2 inhibition', 'common_aes': ['cardiovascular events', 'gi upset']},
    'diclofenac': {'class': 'NSAID', 'mechanism': 'Non-selective COX inhibition', 'common_aes': ['gi bleeding', 'hepatotoxicity']},
    # SSRIs
    'sertraline': {'class': 'SSRI Antidepressant', 'mechanism': 'Selective serotonin reuptake inhibition', 'common_aes': ['nausea', 'insomnia', 'sexual dysfunction', 'serotonin syndrome']},
    'fluoxetine': {'class': 'SSRI Antidepressant', 'mechanism': 'Selective serotonin reuptake inhibition', 'common_aes': ['anxiety', 'insomnia', 'weight changes']},
    'escitalopram': {'class': 'SSRI Antidepressant', 'mechanism': 'Selective serotonin reuptake inhibition', 'common_aes': ['nausea', 'headache', 'sexual dysfunction']},
    'paroxetine': {'class': 'SSRI Antidepressant', 'mechanism': 'Selective serotonin reuptake inhibition', 'common_aes': ['weight gain', 'withdrawal syndrome', 'sexual dysfunction']},
    'citalopram': {'class': 'SSRI Antidepressant', 'mechanism': 'Selective serotonin reuptake inhibition', 'common_aes': ['qt prolongation', 'nausea']},
    # Statins
    'atorvastatin': {'class': 'Statin (HMG-CoA reductase inhibitor)', 'mechanism': 'Competitively inhibits HMG-CoA reductase in cholesterol biosynthesis', 'common_aes': ['myalgia', 'rhabdomyolysis', 'hepatotoxicity']},
    'rosuvastatin': {'class': 'Statin', 'mechanism': 'HMG-CoA reductase inhibition', 'common_aes': ['myalgia', 'elevated liver enzymes']},
    'simvastatin': {'class': 'Statin', 'mechanism': 'HMG-CoA reductase inhibition', 'common_aes': ['myopathy', 'rhabdomyolysis']},
    'pravastatin': {'class': 'Statin', 'mechanism': 'HMG-CoA reductase inhibition', 'common_aes': ['myalgia', 'gi upset']},
    # Antihypertensives
    'lisinopril': {'class': 'ACE Inhibitor', 'mechanism': 'Blocks angiotensin-converting enzyme', 'common_aes': ['cough', 'angioedema', 'hyperkalaemia']},
    'enalapril': {'class': 'ACE Inhibitor', 'mechanism': 'Blocks angiotensin-converting enzyme', 'common_aes': ['cough', 'hypotension']},
    'losartan': {'class': 'ARB (Angiotensin II Receptor Blocker)', 'mechanism': 'Blocks AT1 receptor', 'common_aes': ['dizziness', 'hyperkalaemia']},
    'valsartan': {'class': 'ARB', 'mechanism': 'AT1 receptor antagonism', 'common_aes': ['dizziness', 'renal impairment']},
    'amlodipine': {'class': 'Calcium Channel Blocker (DHP)', 'mechanism': 'Blocks L-type calcium channels in vascular smooth muscle', 'common_aes': ['oedema', 'flushing', 'dizziness']},
    'metoprolol': {'class': 'Beta-1 Selective Blocker', 'mechanism': 'Selective beta-1 adrenergic receptor antagonism', 'common_aes': ['bradycardia', 'fatigue', 'hypotension']},
    'atenolol': {'class': 'Beta-1 Selective Blocker', 'mechanism': 'Beta-1 adrenergic blockade', 'common_aes': ['bradycardia', 'fatigue']},
    'hydrochlorothiazide': {'class': 'Thiazide Diuretic', 'mechanism': 'Inhibits Na-Cl symporter in distal convoluted tubule', 'common_aes': ['hypokalaemia', 'hyponatraemia', 'hyperuricaemia']},
    # Diabetes
    'metformin': {'class': 'Biguanide Antidiabetic', 'mechanism': 'Decreases hepatic glucose production, increases insulin sensitivity', 'common_aes': ['lactic acidosis', 'gi upset', 'vitamin b12 deficiency']},
    'insulin': {'class': 'Insulin', 'mechanism': 'Exogenous insulin replacement, activates insulin receptors', 'common_aes': ['hypoglycaemia', 'weight gain', 'lipodystrophy']},
    'glipizide': {'class': 'Sulfonylurea', 'mechanism': 'Stimulates pancreatic beta-cell insulin secretion via K-ATP channel closure', 'common_aes': ['hypoglycaemia', 'weight gain']},
    'sitagliptin': {'class': 'DPP-4 Inhibitor', 'mechanism': 'Inhibits dipeptidyl peptidase-4, increasing incretin levels', 'common_aes': ['pancreatitis', 'nasopharyngitis']},
    'empagliflozin': {'class': 'SGLT2 Inhibitor', 'mechanism': 'Inhibits sodium-glucose co-transporter 2 in proximal tubule', 'common_aes': ['uti', 'genital mycotic infections', 'diabetic ketoacidosis']},
    # Opioids
    'oxycodone': {'class': 'Opioid Analgesic (mu-agonist)', 'mechanism': 'Full agonist at mu-opioid receptors', 'common_aes': ['constipation', 'respiratory depression', 'dependence']},
    'morphine': {'class': 'Opioid Analgesic', 'mechanism': 'Mu-opioid receptor agonism', 'common_aes': ['constipation', 'respiratory depression', 'nausea']},
    'fentanyl': {'class': 'Synthetic Opioid', 'mechanism': 'Potent mu-opioid receptor agonist', 'common_aes': ['respiratory depression', 'sedation']},
    'tramadol': {'class': 'Opioid Analgesic (atypical)', 'mechanism': 'Weak mu-opioid agonism + serotonin/norepinephrine reuptake inhibition', 'common_aes': ['seizures', 'serotonin syndrome', 'nausea']},
    # Antibiotics
    'amoxicillin': {'class': 'Penicillin Antibiotic', 'mechanism': 'Inhibits bacterial cell wall synthesis (PBP binding)', 'common_aes': ['rash', 'diarrhoea', 'anaphylaxis']},
    'azithromycin': {'class': 'Macrolide Antibiotic', 'mechanism': 'Binds 50S ribosomal subunit, inhibits bacterial protein synthesis', 'common_aes': ['qt prolongation', 'gi upset', 'hepatotoxicity']},
    'ciprofloxacin': {'class': 'Fluoroquinolone Antibiotic', 'mechanism': 'Inhibits bacterial DNA gyrase and topoisomerase IV', 'common_aes': ['tendon rupture', 'neuropathy', 'qt prolongation']},
    'levofloxacin': {'class': 'Fluoroquinolone', 'mechanism': 'DNA gyrase / topoisomerase IV inhibition', 'common_aes': ['tendon disorders', 'neuropathy']},
    'doxycycline': {'class': 'Tetracycline Antibiotic', 'mechanism': 'Inhibits 30S ribosomal subunit', 'common_aes': ['photosensitivity', 'oesophageal ulceration']},
    # PPI
    'omeprazole': {'class': 'Proton Pump Inhibitor', 'mechanism': 'Irreversibly inhibits H+/K+ ATPase in gastric parietal cells', 'common_aes': ['hypomagnesaemia', 'fracture risk', 'c. difficile infection']},
    'pantoprazole': {'class': 'Proton Pump Inhibitor', 'mechanism': 'H+/K+ ATPase inhibition', 'common_aes': ['hypomagnesaemia', 'vitamin b12 deficiency']},
    'esomeprazole': {'class': 'Proton Pump Inhibitor', 'mechanism': 'H+/K+ ATPase inhibition', 'common_aes': ['headache', 'gi upset']},
    # Anticonvulsants
    'gabapentin': {'class': 'Gabapentinoid Anticonvulsant', 'mechanism': 'Binds alpha-2-delta subunit of voltage-gated calcium channels', 'common_aes': ['dizziness', 'somnolence', 'peripheral oedema']},
    'pregabalin': {'class': 'Gabapentinoid', 'mechanism': 'Alpha-2-delta calcium channel modulation', 'common_aes': ['weight gain', 'dizziness', 'somnolence']},
    'levetiracetam': {'class': 'Anticonvulsant', 'mechanism': 'Binds synaptic vesicle protein SV2A', 'common_aes': ['behavioural changes', 'somnolence']},
    'valproic acid': {'class': 'Anticonvulsant / Mood Stabilizer', 'mechanism': 'Multiple: GABA potentiation, sodium channel blockade, HDAC inhibition', 'common_aes': ['hepatotoxicity', 'pancreatitis', 'teratogenicity']},
    'carbamazepine': {'class': 'Anticonvulsant', 'mechanism': 'Sodium channel blockade', 'common_aes': ['sjs/ten', 'agranulocytosis', 'hyponatraemia']},
    # Oncology
    'methotrexate': {'class': 'Antimetabolite / DMARD', 'mechanism': 'Dihydrofolate reductase inhibition; inhibits purine synthesis', 'common_aes': ['hepatotoxicity', 'pancytopenia', 'pneumonitis']},
    'pembrolizumab': {'class': 'Immune Checkpoint Inhibitor (anti-PD-1)', 'mechanism': 'Blocks PD-1 receptor, restoring T-cell anti-tumour response', 'common_aes': ['immune-mediated colitis', 'hepatitis', 'pneumonitis', 'thyroiditis']},
    'nivolumab': {'class': 'Immune Checkpoint Inhibitor (anti-PD-1)', 'mechanism': 'PD-1 blockade', 'common_aes': ['immune-mediated adverse reactions', 'fatigue']},
    'tamoxifen': {'class': 'Selective Estrogen Receptor Modulator (SERM)', 'mechanism': 'Competitive estrogen receptor antagonist in breast tissue', 'common_aes': ['endometrial cancer risk', 'venous thromboembolism', 'hot flashes']},
    'lenalidomide': {'class': 'Immunomodulatory (IMiD)', 'mechanism': 'Cereblon binding, modulates ubiquitin ligase activity', 'common_aes': ['neutropenia', 'thrombocytopenia', 'venous thromboembolism']},
    # Biologics
    'adalimumab': {'class': 'TNF-alpha Inhibitor (Biologic DMARD)', 'mechanism': 'Monoclonal antibody neutralizing TNF-alpha', 'common_aes': ['infections', 'injection site reactions', 'lymphoma risk']},
    'infliximab': {'class': 'TNF-alpha Inhibitor', 'mechanism': 'Chimeric monoclonal antibody against TNF-alpha', 'common_aes': ['infusion reactions', 'serious infections', 'hepatotoxicity']},
    'rituximab': {'class': 'Anti-CD20 Monoclonal Antibody', 'mechanism': 'Depletes CD20+ B lymphocytes', 'common_aes': ['infusion reactions', 'infections', 'progressive multifocal leukoencephalopathy']},
    # Cardiovascular
    'clopidogrel': {'class': 'Antiplatelet (P2Y12 inhibitor)', 'mechanism': 'Irreversibly blocks P2Y12 ADP receptor on platelets', 'common_aes': ['bleeding', 'ttp']},
    'digoxin': {'class': 'Cardiac Glycoside', 'mechanism': 'Inhibits Na+/K+ ATPase, increases intracellular calcium', 'common_aes': ['arrhythmia', 'nausea', 'visual disturbances']},
    'amiodarone': {'class': 'Class III Antiarrhythmic', 'mechanism': 'Potassium channel blockade (+ sodium, calcium, beta-blocking)', 'common_aes': ['pulmonary toxicity', 'thyroid dysfunction', 'hepatotoxicity', 'corneal deposits']},
    # Misc
    'prednisone': {'class': 'Systemic Corticosteroid', 'mechanism': 'Glucocorticoid receptor agonism; suppresses inflammatory gene transcription', 'common_aes': ['osteoporosis', 'hyperglycaemia', 'adrenal suppression', 'immunosuppression']},
    'prednisolone': {'class': 'Systemic Corticosteroid', 'mechanism': 'Glucocorticoid receptor activation', 'common_aes': ['cushingoid features', 'osteoporosis']},
    'dexamethasone': {'class': 'Systemic Corticosteroid', 'mechanism': 'Potent glucocorticoid receptor agonist', 'common_aes': ['hyperglycaemia', 'immunosuppression']},
    'levothyroxine': {'class': 'Thyroid Hormone Replacement', 'mechanism': 'Exogenous T4 supplementation', 'common_aes': ['tachycardia', 'osteoporosis (if overreplaced)']},
    'allopurinol': {'class': 'Xanthine Oxidase Inhibitor', 'mechanism': 'Inhibits xanthine oxidase, reducing uric acid production', 'common_aes': ['sjs/ten', 'dress syndrome', 'gout flare']},
    'montelukast': {'class': 'Leukotriene Receptor Antagonist', 'mechanism': 'Blocks CysLT1 receptor', 'common_aes': ['neuropsychiatric events', 'suicidality (boxed warning)']},
    'sildenafil': {'class': 'PDE5 Inhibitor', 'mechanism': 'Inhibits phosphodiesterase type 5, increasing cGMP', 'common_aes': ['headache', 'flushing', 'visual disturbances', 'priapism']},
    'duloxetine': {'class': 'SNRI Antidepressant', 'mechanism': 'Serotonin and norepinephrine reuptake inhibition', 'common_aes': ['nausea', 'dry mouth', 'hepatotoxicity', 'withdrawal syndrome']},
    'venlafaxine': {'class': 'SNRI Antidepressant', 'mechanism': 'Serotonin and norepinephrine reuptake inhibition', 'common_aes': ['hypertension', 'withdrawal syndrome', 'serotonin syndrome']},
    'quetiapine': {'class': 'Atypical Antipsychotic', 'mechanism': 'Dopamine D2 and serotonin 5-HT2A receptor antagonism', 'common_aes': ['metabolic syndrome', 'sedation', 'qt prolongation']},
    'olanzapine': {'class': 'Atypical Antipsychotic', 'mechanism': 'Multi-receptor antagonist (D2, 5-HT2A, H1, M1)', 'common_aes': ['weight gain', 'metabolic syndrome', 'sedation']},
    'aripiprazole': {'class': 'Atypical Antipsychotic (partial agonist)', 'mechanism': 'Partial D2/5-HT1A agonist, 5-HT2A antagonist', 'common_aes': ['akathisia', 'insomnia', 'compulsive behaviours']},
    'lithium': {'class': 'Mood Stabilizer', 'mechanism': 'Multiple: GSK-3 inhibition, inositol depletion, neuroprotection', 'common_aes': ['nephrotoxicity', 'thyroid dysfunction', 'tremor', 'narrow therapeutic index']},
    'zolpidem': {'class': 'Non-benzodiazepine Hypnotic (Z-drug)', 'mechanism': 'Selective GABA-A alpha-1 subunit agonism', 'common_aes': ['complex sleep behaviours', 'amnesia', 'dependence']},
    'alprazolam': {'class': 'Benzodiazepine', 'mechanism': 'GABA-A receptor positive allosteric modulator', 'common_aes': ['dependence', 'withdrawal seizures', 'sedation']},
    'lorazepam': {'class': 'Benzodiazepine', 'mechanism': 'GABA-A receptor modulation', 'common_aes': ['sedation', 'dependence', 'respiratory depression']},
}


def lookup_drug_class(drugname: str) -> dict | None:
    """Look up drug class and mechanism for a given drug name."""
    if not drugname:
        return None
    name = drugname.strip().lower()
    # Direct match
    if name in DRUG_CLASS_MAP:
        return DRUG_CLASS_MAP[name]
    # Substring match (e.g., "WARFARIN SODIUM" → "warfarin")
    for key, info in DRUG_CLASS_MAP.items():
        if key in name or name.startswith(key):
            return info
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  External Dataset Downloader")
    print("  Sources: PHEE, BioDEX, ADE Corpus v2")
    print("=" * 60)
    
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    
    # Download each dataset
    phee_pairs = download_phee()
    all_pairs.extend(phee_pairs)
    
    biodex_pairs = download_biodex()
    all_pairs.extend(biodex_pairs)
    
    ade_pairs = download_ade_corpus()
    all_pairs.extend(ade_pairs)
    
    # Save drug class map as JSON for use by 03_build_training_data.py
    drug_class_file = EXTERNAL_DIR / "drug_class_map.json"
    with open(str(drug_class_file), 'w', encoding='utf-8') as f:
        json.dump(DRUG_CLASS_MAP, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Drug class map: {drug_class_file} ({len(DRUG_CLASS_MAP)} drugs)")
    
    # Save unified pairs
    with open(str(OUTPUT_FILE), 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EXTERNAL DATA SUMMARY")
    print(f"{'=' * 60}")
    
    source_counts = Counter(p['source'] for p in all_pairs)
    task_counts = Counter(p['task_type'] for p in all_pairs)
    
    print(f"\n  By source:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count:,}")
    
    print(f"\n  By task type:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count:,}")
    
    print(f"\n  💾 Output: {OUTPUT_FILE} ({len(all_pairs):,} total pairs)")
    print(f"  💾 Drug classes: {drug_class_file}")
    
    if len(all_pairs) == 0:
        print(f"\n  ⚠️  No external data downloaded!")
        print(f"     This likely means 'datasets' package is not installed.")
        print(f"     Run: pip install datasets")
        return 1
    
    print(f"\n  ✅ External datasets ready!")
    print(f"     Next: python src/data/03_build_training_data.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
