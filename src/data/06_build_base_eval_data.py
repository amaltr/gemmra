"""
Script to build base model specific evaluation dataset from decontaminated eval_data.jsonl.
Samples 250 stratified samples (62 for T1, 63 for T2, 62 for T3, 63 for T4)
and updates their system prompts to include strict formatting instructions for the base model.
"""

import json
import random
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EVAL_FILE = Path("data/processed/eval_data.jsonl")
OUTPUT_FILE = Path("data/processed/eval_data_base_250.jsonl")
SEED = 42

T1_SYSTEM_PROMPT = (
    "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: "
    "Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). "
    "Base your assessment on the clinical narrative provided.\n\n"
    "CRITICAL formatting instruction: You must output your final assessment EXACTLY in the following format. "
    "Do NOT write any thinking process, preambles, intro, or conversational filler. Start directly with 'SERIOUS:'.\n\n"
    "Format:\n"
    "SERIOUS: [YES or NO]\n"
    "Criteria met: [List codes like HO, DE, etc., or None]\n"
    "Rationale: [Brief explanation]"
)

T2_SYSTEM_PROMPT = (
    "You are a medical coder specializing in MedDRA terminology. Given an adverse event description from "
    "clinical or patient-reported text, map it to the correct MedDRA Preferred Term (PT).\n\n"
    "CRITICAL formatting instruction: You must output your mapped term EXACTLY in the following format. "
    "Do NOT write any thinking process, preambles, intro, or conversational filler. Start directly with 'MedDRA PT:'.\n\n"
    "Format:\n"
    "MedDRA PT: [Preferred Term]\n"
    "Drug context: [Drug name or Unknown]\n"
    "Rationale: [Brief explanation]"
)

T3_SYSTEM_PROMPT = (
    "You are a pharmacovigilance expert. Determine if the reported adverse event is listed in the drug's "
    "approved product label. Consider the drug's pharmacological class and mechanism of action in your assessment.\n\n"
    "CRITICAL formatting instruction: You must output your assessment EXACTLY in the following format. "
    "Do NOT write any thinking process, preambles, intro, or conversational filler. Start directly with 'LABELLED:'.\n\n"
    "Format:\n"
    "LABELLED: [YES or NO]\n"
    "Drug: [Drug name]\n"
    "Adverse event: [Adverse event name]\n"
    "Label section: [Section name, or Not found]\n"
    "Rationale: [Brief explanation]"
)

T4_SYSTEM_PROMPT = (
    "You are a pharmacovigilance expert. Read the clinical case narrative carefully, extract the relevant evidence "
    "(temporal relationship, dechallenge, rechallenge, concomitant medications, confounding factors), "
    "and assess drug-event causality using WHO-UMC criteria: Certain, Probable, Possible, Unlikely, Conditional, Unassessable.\n\n"
    "CRITICAL formatting instruction: You must output your causality assessment EXACTLY in the following format. "
    "Do NOT write any thinking process, preambles, intro, or conversational filler. Start directly with 'WHO-UMC Causality:'.\n\n"
    "Format:\n"
    "WHO-UMC Causality: [Certain/Probable/Possible/Unlikely/Conditional/Unassessable]\n"
    "Evidence:\n"
    "  - Temporal: [analysis of timing]\n"
    "  - Dechallenge: [analysis of dechallenge]\n"
    "  - Rechallenge: [analysis of rechallenge]\n"
    "  - Confounders: [analysis of confounding factors]\n"
    "  - Alternatives: [analysis of concomitant drugs/alternative causes]"
)

def format_system_prompt(task):
    if task == 'T1':
        return T1_SYSTEM_PROMPT
    elif task == 'T2':
        return T2_SYSTEM_PROMPT
    elif task == 'T3':
        return T3_SYSTEM_PROMPT
    elif task == 'T4':
        return T4_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown task: {task}")

def main():
    print("=" * 60)
    print("  Base Model Eval Data Builder")
    print("=" * 60)

    if not EVAL_FILE.exists():
        print(f"ERROR: {EVAL_FILE} not found. Build dataset first.")
        sys.exit(1)

    with open(EVAL_FILE, encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_data)} decontaminated examples from {EVAL_FILE}")

    # Group by task
    by_task = {}
    for ex in all_data:
        task = ex.get('task', 'unknown')
        by_task.setdefault(task, []).append(ex)

    # Sample stratified target: 250 samples in total (62, 63, 62, 63)
    random.seed(SEED)
    sampled = []
    targets = {'T1': 62, 'T2': 63, 'T3': 62, 'T4': 63}

    for task, target in targets.items():
        examples = by_task.get(task, [])
        if len(examples) < target:
            print(f"Warning: task {task} only has {len(examples)} examples (requested {target})")
            sampled.extend(examples)
        else:
            sampled.extend(random.sample(examples, target))

    # Shuffle to mix tasks
    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} total stratified examples.")

    # Process and format system prompt for the base model
    formatted = []
    for ex in sampled:
        task = ex['task']
        messages = ex['messages']
        
        # Modify the system message (first message in the role: system)
        new_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                new_messages.append({"role": "system", "content": format_system_prompt(task)})
            else:
                new_messages.append(msg)
        
        # Build the formatted entry
        formatted_entry = {
            "task": task,
            "primaryid": ex.get("primaryid", ""),
            "messages": new_messages
        }
        formatted.append(formatted_entry)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in formatted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Successfully created base eval dataset: {OUTPUT_FILE} ({len(formatted)} lines)")
    print("=" * 60)

if __name__ == "__main__":
    main()
