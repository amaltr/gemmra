"""
DAPO Training Script — Stage 2 (GROUND-ZERO REBUILD)
Run this on GPU (AMD MI300X) AFTER SFT training is complete.

Key change: CORRECTNESS REWARD added as PRIMARY learning signal.
Previous version had only format/style rewards — no way to tell right from wrong.
Now the model learns to actually get the answer correct.

5 reward signals:
1. CORRECTNESS (w=2.0) — compares answer to ground truth (PRIMARY signal)
2. Format compliance (w=0.5) — Gemma 4 thinking tokens + structure
3. Task structure (w=0.5) — task-specific fields present
4. Reasoning quality (w=0.8) — domain terminology + reasoning depth
5. Faithfulness (w=1.0) — case-data grounding (CRPO-inspired)
"""

import os
import sys
import re
import torch
import yaml
from pathlib import Path

# Add project root to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.meddra_soc import compute_t2_similarity

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# ============================================================
# Configuration — loaded from configs/grpo_config.yaml with defaults
# ============================================================

def _load_yaml_config(config_path: str = "configs/grpo_config.yaml") -> dict:
    """Load config from YAML file, returning empty dict if not found."""
    path = Path(config_path)
    if path.exists():
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            print(f"  ✅ Loaded config from {config_path}")
            return cfg
        except Exception as e:
            print(f"  ⚠️  Could not parse {config_path}: {e} — using defaults")
    else:
        print(f"  ℹ️  Config file {config_path} not found — using defaults")
    return {}

_cfg = _load_yaml_config()
_model_cfg = _cfg.get('model', {})
_train_cfg = _cfg.get('training', {})
_data_cfg = _cfg.get('data', {})
_out_cfg = _cfg.get('output', {})

SFT_CHECKPOINT = _model_cfg.get('checkpoint', "checkpoints/sft")
OUTPUT_DIR = _out_cfg.get('dir', "checkpoints/grpo_final")
TRAIN_FILE = _data_cfg.get('train_file', "data/processed/training_data.jsonl")

MAX_SEQ_LENGTH = _model_cfg.get('max_seq_length', 8192)
LOAD_IN_4BIT = _model_cfg.get('load_in_4bit', False)

LEARNING_RATE = _train_cfg.get('learning_rate', 5e-6)
# GRPO FIX: 4 generations → 8. With 4, SFT model produces near-identical
# outputs → zero reward variance → zero gradient → dead training.
# 8 gives more diversity for advantage estimation.
NUM_GENERATIONS = _train_cfg.get('num_generations', 8)
MAX_NEW_TOKENS = _train_cfg.get('max_new_tokens', 1024)
NUM_EPOCHS = _train_cfg.get('num_epochs', 1)
BATCH_SIZE = _train_cfg.get('per_device_batch_size', 1)
MAX_SAMPLES = _data_cfg.get('max_samples', 2000)

# YAML-DRIVEN REWARD WEIGHTS (fixes dead config bug)
# Previous: weights were hardcoded inline in each reward function.
# Changing grpo_config.yaml had NO effect. Now they are read from YAML.
_reward_cfg = _cfg.get('reward', {})
_reward_components = _reward_cfg.get('components', [])
# GRPO FIX: Removed format/structure/reasoning rewards — they had ZERO variance
# (SFT already formats perfectly, all generations identical → no gradient signal).
# Evidence: format_reward/std=0.002, frac_reward_zero_std=0.6
# Keep ONLY rewards that DIFFERENTIATE completions:
#   - correctness: different answers get different scores (the learning signal)
#   - faithfulness: different completions ground to different entities
REWARD_WEIGHTS = {
    'correctness': 2.0,
    'faithfulness': 1.0,
}
# Override defaults with YAML values
for comp in _reward_components:
    name = comp.get('name', '')
    if name in REWARD_WEIGHTS and 'weight' in comp:
        REWARD_WEIGHTS[name] = float(comp['weight'])
print(f"  Reward weights: {REWARD_WEIGHTS}")


# ============================================================
# Reward Functions
# ============================================================

def _extract_text(completion) -> str:
    """H3 FIX: Robustly extract text from whatever format TRL passes."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # List of message dicts: [{"role": "assistant", "content": "..."}]
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", str(completion[0]))
        # List of strings
        if completion and isinstance(completion[0], str):
            return completion[0]
    return str(completion)


def _extract_answer_text(completion) -> str:
    """C2 FIX: Extract ONLY the final answer portion, stripping the thinking trace.
    
    Without this, reward functions match keywords inside <|channel>thought....<channel|>
    draft reasoning. The model can game rewards by writing 'SERIOUS: YES' as a hypothesis
    in its thinking, even if the final answer says 'SERIOUS: NO'.
    
    GRPO FIX: Handle missing <channel|> — model sometimes uses blank line
    instead of <channel|> to separate thinking from answer.
    """
    full_text = _extract_text(completion)
    # Try standard pattern first: <|channel>thought...text...<channel|>
    answer = re.sub(r'<\|channel>thought.*?<channel\|>', '', full_text, flags=re.DOTALL)
    if answer.strip() != full_text.strip():
        return answer.strip()
    # Fallback: thinking separated by blank line (no <channel|>)
    if '<|channel>thought' in full_text:
        # Split on first blank line after thinking tag
        parts = re.split(r'\n\n', full_text, maxsplit=1)
        if len(parts) > 1:
            # Check if second part has structured fields
            candidate = parts[1].strip()
            if re.search(r'SERIOUS:|MedDRA PT:|LABELLED:|WHO-UMC', candidate, re.I):
                return candidate
    return full_text.strip()


def format_reward(completions: list, **kwargs) -> list[float]:
    """Reward: Does the output have proper Gemma 4 thinking tokens + reasonable length?
    
    S5 FIX: This reward ONLY checks format/length — NOT task-specific fields.
    Task-specific field checking moved to task_structure_reward.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        
        # Gemma 4 thinking mode format
        think_match = re.search(r'<\|channel>thought(.*?)<channel\|>', text, re.DOTALL)
        if think_match:
            think_len = len(think_match.group(1).split())
            score += min(0.5, think_len / 80.0)
        
        # Length bonus — substantive responses
        word_count = len(text.split())
        score += min(0.3, word_count / 400.0)
        
        # Has a clear answer section (after thinking)
        after_think = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL)
        if len(after_think.strip()) > 20:
            score += 0.2
        
        rewards.append(min(1.0, score) * REWARD_WEIGHTS['format_compliance'])
    return rewards


# C1 FIX: Task-specific field sets. Each task can now reach 1.0 with ONLY its own fields.
# Previous version summed ALL task fields — a correct T1 output could only score 0.45/1.0,
# incentivizing the model to hallucinate irrelevant headers from other tasks.
_TASK_FIELD_PATTERNS = {
    'T1': [  # Seriousness Assessment
        (r'SERIOUS:\s*(YES|NO)', 0.35),
        (r'Criteria met:', 0.25),
        (r'Rationale:', 0.25),
        (r'ICH E2A|seriousness criteria', 0.15),
    ],
    'T2': [  # MedDRA Coding
        (r'MedDRA PT:\s*\S+', 0.35),
        (r'Drug context:', 0.25),
        (r'Rationale:', 0.25),
        (r'coding|preferred term', 0.15),
    ],
    'T3': [  # Labelling Status
        (r'LABELLED:\s*(YES|NO)', 0.35),
        (r'Rationale:', 0.25),
        (r'Label section:|product label', 0.25),
        (r'mechanism|pharmacolog|class effect', 0.15),
    ],
    'T4': [  # Causality Assessment
        (r'WHO-UMC Causality:\s*\S+', 0.30),
        (r'Evidence:', 0.25),
        (r'Temporal|Dechallenge|Rechallenge', 0.25),
        (r'Rationale:|assessment', 0.20),
    ],
}

def task_structure_reward(completions: list, **kwargs) -> list[float]:
    """Reward: Does the output contain the correct task-specific structured fields?
    
    C1 FIX: Now task-aware — only checks fields for the CURRENT task.
    A correct T1 output can score 1.0, not the old 0.45 cap.
    
    C2 FIX: Searches only the answer section (thinking trace stripped).
    """
    task_labels = kwargs.get('task', [])
    
    rewards = []
    for i, completion in enumerate(completions):
        answer_text = _extract_answer_text(completion)  # C2 FIX
        task = task_labels[i] if i < len(task_labels) else ''
        
        # C1 FIX: Select only this task's fields (or fall back to generic)
        field_patterns = _TASK_FIELD_PATTERNS.get(task, [
            (r'Rationale:', 0.5),
            (r'\w+:', 0.5),
        ])
        
        score = 0.0
        for pattern, weight in field_patterns:
            if re.search(pattern, answer_text, re.IGNORECASE):
                score += weight
        
        rewards.append(min(1.0, score) * REWARD_WEIGHTS['task_structure'])
    return rewards


def reasoning_quality_reward(completions: list, **kwargs) -> list[float]:
    """Reward: Is the reasoning substantive and domain-relevant?
    
    S5 FIX: Only checks domain terminology depth, NOT structural fields.
    """
    DOMAIN_TERMS = [
        'ich e2a', 'meddra', 'who-umc', 'dechallenge', 'rechallenge',
        'temporal', 'causality', 'seriousness', 'hospitalization',
        'adverse event', 'pharmacovigilance', 'preferred term',
        'concomitant', 'indication', 'suspect', 'reaction',
    ]
    rewards = []
    for completion in completions:
        text = _extract_text(completion).lower()
        
        # Count domain terms (continuous)
        term_count = sum(1 for term in DOMAIN_TERMS if term in text)
        score = min(0.6, term_count * 0.05)
        
        # Reasoning depth: multiple reasoning steps/sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        score += min(0.2, len(sentences) * 0.02)
        
        # Explicit reasoning markers
        reasoning_markers = ['because', 'therefore', 'given that', 'this suggests',
                           'consistent with', 'based on', 'indicates', 'considering']
        marker_count = sum(1 for m in reasoning_markers if m in text)
        score += min(0.2, marker_count * 0.05)
        
        rewards.append(min(1.0, score) * REWARD_WEIGHTS['reasoning_quality'])
    return rewards


def _extract_prompt_entities(prompt_text: str) -> dict:
    """Extract specific entities from the input prompt for grounding checks.
    
    Returns a dict with entity types as keys and sets of extracted strings as values.
    These are the ACTUAL case details that the answer should reference.
    """
    entities = {'drugs': set(), 'aes': set(), 'demographics': set()}
    prompt_lower = prompt_text.lower()
    
    # Extract drug names — they appear after "on/prescribed/treated with/of"
    for m in re.finditer(r'(?:on|prescribed|treated with|drug[:\s]+|therapy with|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', prompt_text):
        drug = m.group(1).strip()
        if len(drug) > 2 and drug.lower() not in ('the', 'unknown', 'none', 'for'):
            entities['drugs'].add(drug.lower())
    
    # Extract AE terms — after "experienced/developed/reported/event"
    for m in re.finditer(r'(?:experienced|developed|reported|adverse event[:\s]+|complaint of)\s+(.+?)(?:\.|,|\n)', prompt_text, re.I):
        ae = m.group(1).strip().rstrip('.')
        if len(ae) > 2 and len(ae) < 80:
            entities['aes'].add(ae.lower())
    
    # Extract demographics — specific age/gender
    age_match = re.search(r'(\d+)(?:\s*-?\s*year|\s*-?\s*month|\s*-?\s*week)', prompt_text, re.I)
    if age_match:
        entities['demographics'].add(age_match.group(1))
    if re.search(r'\bmale\b', prompt_lower):
        entities['demographics'].add('male')
    if re.search(r'\bfemale\b', prompt_lower):
        entities['demographics'].add('female')
    
    return entities


def faithfulness_reward(completions: list, **kwargs) -> list[float]:
    """Reward: Does the ANSWER reference specific entities FROM THE INPUT PROMPT?
    
    M2 FIX: Now prompt-grounded — extracts actual drug names, AE terms, and
    demographics from the input prompt, then checks if the answer references
    THOSE specific entities. Generic boilerplate ("the patient recovered") no
    longer scores because 'patient' alone isn't a prompt-specific entity.
    
    C2 FIX: Searches only the answer section (thinking trace stripped).
    """
    # C2 FIX: TRL GRPOTrainer passes 'prompts' (plural), not 'prompt' (singular).
    # The singular 'prompt' column is often filtered out by TRL before calling rewards.
    # Try both to be safe across TRL versions.
    prompts = kwargs.get('prompts', kwargs.get('prompt', []))
    
    rewards = []
    for i, completion in enumerate(completions):
        answer = _extract_answer_text(completion)  # C2 FIX: answer only
        answer_lower = answer.lower()
        score = 0.0
        
        # Get the input prompt for this sample
        prompt = prompts[i] if i < len(prompts) else ''
        # Extract prompt text from whatever format TRL passes
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            prompt = prompt[0].get('content', str(prompt[0]))
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            prompt = prompt[0]
        prompt = str(prompt)
        if not prompt or len(prompt) < 10:
            # No usable prompt — fall back to minimal keyword check
            rewards.append(0.3)
            continue
        
        entities = _extract_prompt_entities(prompt)
        
        # Drug grounding: does the answer mention the ACTUAL drug from the prompt?
        if entities['drugs']:
            drugs_mentioned = sum(1 for d in entities['drugs'] if d in answer_lower)
            score += min(0.3, drugs_mentioned * 0.3 / max(len(entities['drugs']), 1))
        
        # AE grounding: does the answer mention the ACTUAL adverse event?
        if entities['aes']:
            aes_mentioned = sum(1 for ae in entities['aes'] if ae in answer_lower)
            score += min(0.25, aes_mentioned * 0.25 / max(len(entities['aes']), 1))
        
        # Demographic grounding: does the answer reference the specific age/gender?
        if entities['demographics']:
            demos_mentioned = sum(1 for d in entities['demographics'] if d in answer_lower)
            score += min(0.15, demos_mentioned * 0.15 / max(len(entities['demographics']), 1))
        
        # Clinical evidence grounding (still valuable, but only worth 0.3 total)
        # These are harder to extract from prompts, so we keep keyword matching
        # but with reduced weight so they can't dominate
        if re.search(r'dechallenge|rechallenge', answer, re.I):
            score += 0.15
        if re.search(r'resolved|recurred|persisted|improved', answer, re.I):
            score += 0.15
        
        rewards.append(min(1.0, score) * REWARD_WEIGHTS['faithfulness'])
    return rewards


def correctness_reward(completions: list, **kwargs) -> list[float]:
    """PRIMARY REWARD: Compare model output to ground truth answer labels.
    
    C1 FIX: Now accesses actual ground truth from the dataset's 'answer_label' column,
    passed by TRL's GRPOTrainer via kwargs. Previous version was circular — it only
    checked self-consistency ("did you mention hospitalization if you said YES?").
    
    Now it actually checks:
    - T1: Does SERIOUS: YES/NO match the ground truth label?
    - T2: Does MedDRA PT match the ground truth (fuzzy)?
    - T3: Does LABELLED: YES/NO match the ground truth label?
    - T4: Does WHO-UMC Causality match the ground truth level (with partial credit)?
    """
    # C1 FIX: Get ground truth labels from dataset columns passed by GRPOTrainer
    answer_labels = kwargs.get('answer_label', [])
    task_labels = kwargs.get('task', [])
    
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_answer_text(completion)  # C2 FIX: answer only, not thinking trace
        
        # Get ground truth for this sample (if available)
        gt_label = answer_labels[i] if i < len(answer_labels) else ''
        task = task_labels[i] if i < len(task_labels) else ''
        
        # If no ground truth available, fall back to neutral score
        if not gt_label:
            rewards.append(0.5)
            continue
        
        score = 0.0
        gt_upper = gt_label.strip().upper()
        
        if task == 'T1':
            # Binary match: SERIOUS YES/NO
            pred_match = re.search(r'SERIOUS:\s*(YES|NO)', text, re.IGNORECASE)
            if pred_match:
                pred = pred_match.group(1).upper()
                score = 1.0 if pred == gt_upper else 0.0
            else:
                score = 0.0  # No answer found
                
        elif task == 'T2':
            # MedDRA PT matching — multi-level via SOC classifier
            pt_match = re.search(r'MedDRA PT:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
            if pt_match:
                pred_pt = pt_match.group(1).strip()
                gt_pt = gt_label.strip()
                # compute_t2_similarity returns 1.0/0.9/0.7/0.5/0.0
                # for exact/synonym/fuzzy/SOC/miss
                score = compute_t2_similarity(pred_pt, gt_pt)
            else:
                score = 0.0
                
        elif task == 'T3':
            # Binary match: LABELLED YES/NO
            pred_match = re.search(r'LABELLED:\s*(YES|NO)', text, re.IGNORECASE)
            if pred_match:
                pred = pred_match.group(1).upper()
                score = 1.0 if pred == gt_upper else 0.0
            else:
                score = 0.0
                
        elif task == 'T4':
            # Causality level match with partial credit
            pred_match = re.search(r'WHO-UMC Causality:\s*(\w+)', text, re.IGNORECASE)
            if pred_match:
                # G3 FIX: Strip non-alpha chars (same as eval M1 fix)
                pred_level = re.sub(r'[^a-zA-Z]', '', pred_match.group(1)).capitalize()
                gt_level = re.sub(r'[^a-zA-Z]', '', gt_label).capitalize()
                
                if pred_level == gt_level:
                    score = 1.0  # Exact match
                else:
                    # Partial credit: ordinal distance
                    order = ['Certain', 'Probable', 'Possible', 'Conditional', 'Unlikely', 'Unassessable']
                    if pred_level in order and gt_level in order:
                        dist = abs(order.index(pred_level) - order.index(gt_level))
                        score = max(0.0, 1.0 - dist * 0.25)  # 1 step = 0.75, 2 = 0.50, etc.
                    else:
                        score = 0.1  # Invalid level but at least produced something
            else:
                score = 0.0
        else:
            score = 0.5  # Unknown task
        
        rewards.append(score * REWARD_WEIGHTS['correctness'])
    
    return rewards


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Gemmra — DAPO Training (Stage 2)")
    print("=" * 60)
    
    # M3 FIX: --local now loads from YAML (single source of truth), not hardcoded values.
    # Only model name + load_in_4bit are truly environment-specific;
    # all other hyperparameters should come from configs/local_grpo_config.yaml.
    global SFT_CHECKPOINT, OUTPUT_DIR, MAX_SAMPLES, NUM_GENERATIONS, BATCH_SIZE, LOAD_IN_4BIT, MAX_SEQ_LENGTH, MAX_NEW_TOKENS
    
    # --smoke: Quick pipeline validation (200 samples, 256 max tokens)
    # Purpose: verify reward functions fire, data loads, model generates, loss decreases.
    # Runtime: ~20-40 min on MI300X vs ~8-30 hrs for full run.
    # Keep num_generations=4 — GRPO needs ≥4 for meaningful advantage estimation.
    if '--smoke' in sys.argv:
        MAX_SAMPLES = _data_cfg.get('smoke_samples', 100)
        MAX_NEW_TOKENS = 256
        print(f"\n  🔥 SMOKE TEST MODE: {MAX_SAMPLES} samples, 256 max tokens")
        print(f"     Purpose: verify pipeline works before committing to full run")
        print(f"     num_generations kept at {NUM_GENERATIONS} (GRPO needs ≥4 for advantage)")
    
    if '--local' in sys.argv:
        local_cfg = _load_yaml_config("configs/local_grpo_config.yaml")
        _local_model = local_cfg.get('model', {})
        _local_train = local_cfg.get('training', {})
        _local_data = local_cfg.get('data', {})
        _local_out = local_cfg.get('output', {})
        
        SFT_CHECKPOINT = _local_model.get('checkpoint', 'checkpoints/local_sft')
        OUTPUT_DIR = _local_out.get('dir', 'checkpoints/local_grpo')
        MAX_SAMPLES = _local_data.get('max_samples', 100)
        NUM_GENERATIONS = _local_train.get('num_generations', 2)
        BATCH_SIZE = _local_train.get('per_device_batch_size', 1)
        LOAD_IN_4BIT = _local_model.get('load_in_4bit', True)
        MAX_SEQ_LENGTH = _local_model.get('max_seq_length', 2048)
        MAX_NEW_TOKENS = _local_train.get('max_new_tokens', 128)
        print(f"\n  🖥️  LOCAL MODE: Loaded from configs/local_grpo_config.yaml")
    
    # Verify SFT checkpoint exists
    if not Path(SFT_CHECKPOINT).exists():
        print(f"  ❌ SFT checkpoint not found: {SFT_CHECKPOINT}")
        print(f"     Run first: python src/training/01_sft_train.py")
        return 1
    
    # Load SFT checkpoint
    print(f"\n  Loading SFT checkpoint: {SFT_CHECKPOINT}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_CHECKPOINT,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"  ✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    # Load data (subset for GRPO)
    print(f"\n  Loading training data (max {MAX_SAMPLES} samples)...")
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    # GRPO needs a 'prompt' field — extract from messages using native chat template.
    # CRITICAL: Must use the same Gemma 4 chat format the model was SFT-trained on.
    #
    # G1 FIX: Gemma 4 processor's apply_chat_template expects content in multimodal
    # format: [{"type": "text", "text": "..."}]. Plain strings cause TypeError.
    def _to_multimodal_msg(msg):
        content = msg.get('content', '')
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"role": msg["role"], "content": content}
    
    def extract_prompt(example):
        messages = example.get('messages', [])
        task = example.get('task', '')  # G4 FIX: preserve task for rewards
        if messages:
            prompt_messages = [m for m in messages if m['role'] != 'assistant']
            try:
                # G1 FIX: wrap in multimodal format for Gemma 4 processor
                mm_messages = [_to_multimodal_msg(m) for m in prompt_messages]
                # GRPO FIX: enable_thinking=True to match inference template
                try:
                    prompt = tokenizer.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                # Fallback: manual Gemma 4 format
                parts = []
                for msg in prompt_messages:
                    role = msg['role']
                    content = msg['content'] if isinstance(msg['content'], str) else msg['content'][0].get('text', '')
                    if role == 'system':
                        parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
                    elif role == 'user':
                        parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                parts.append("<start_of_turn>model\n")
                prompt = "\n".join(parts)
            
            # Extract ground truth answer label from assistant response.
            # G2 FIX: Strip thinking trace FIRST — previous version searched full text
            # including <|channel>thought..., which could grab wrong match.
            answer_label = ''
            assistant_msg = [m for m in messages if m['role'] == 'assistant']
            if assistant_msg:
                asst_text = assistant_msg[0].get('content', '')
                # Strip thinking trace to get only the answer section
                answer_only = re.sub(r'<\|channel>thought.*?<channel\|>', '', asst_text, flags=re.DOTALL).strip()
                
                if task == 'T1':
                    m = re.search(r'SERIOUS:\s*(YES|NO)', answer_only, re.IGNORECASE)
                    answer_label = m.group(1).upper() if m else ''
                elif task == 'T2':
                    m = re.search(r'MedDRA PT:\s*(.+?)(?:\n|$)', answer_only, re.IGNORECASE)
                    answer_label = m.group(1).strip().rstrip('.,;:!?') if m else ''
                elif task == 'T3':
                    m = re.search(r'LABELLED:\s*(YES|NO)', answer_only, re.IGNORECASE)
                    answer_label = m.group(1).upper() if m else ''
                elif task == 'T4':
                    m = re.search(r'WHO-UMC Causality:\s*(\w+)', answer_only, re.IGNORECASE)
                    answer_label = m.group(1).strip().capitalize() if m else ''
            
            return {"prompt": prompt, "answer_label": answer_label, "task": task}
        else:
            return {"prompt": example.get('text', ''), "answer_label": '', "task": task}
    
    dataset = dataset.map(extract_prompt)
    print(f"  ✅ Using {len(dataset)} prompts for GRPO (native chat template format)")
    
    # Configure DAPO (GRPO successor — TRL v1.0, June 2026)
    # DAPO fixes: entropy collapse, long-CoT bias, wasted compute on trivial batches
    # See: frontier_research_sweep.md for full analysis
    #
    # TRL renamed/removed params across versions (e.g. max_new_tokens → max_completion_length).
    # We validate ALL params against the GRPOConfig signature to prevent crashes.
    import inspect
    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    
    # Build the full desired config — we'll filter it below
    desired_config = {
        "per_device_train_batch_size": BATCH_SIZE,
        "num_generations": NUM_GENERATIONS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_EPOCHS,
        "output_dir": OUTPUT_DIR,
        "logging_steps": 5,
        "bf16": True,
        "beta": 0.0,                                          # DAPO paper: 0.0 removes KL penalty entirely
        "temperature": 1.5,                                    # GRPO FIX: 1.2→1.5. SFT model too consistent at 1.2 → zero reward variance
        # DAPO-specific params
        "loss_type": "dapo",                                   # DAPO mode
        "epsilon_high": 0.28,                                  # Clip-higher: prevents entropy collapse
        "mask_truncated_completions": True,                    # Ignore incomplete generations
        # Reward weights applied INSIDE each reward function using REWARD_WEIGHTS dict
        # (loaded from YAML config). TRL's reward_weights param is unreliable across versions
        # so we scale outputs directly. Changing configs/grpo_config.yaml now takes effect.
        "multi_objective_aggregation": "normalize_then_sum",   # Prevent any single reward from dominating
    }
    
    # Handle max_new_tokens → max_completion_length rename (TRL 0.x → 1.x)
    if "max_completion_length" in grpo_config_params:
        desired_config["max_completion_length"] = MAX_NEW_TOKENS
    elif "max_new_tokens" in grpo_config_params:
        desired_config["max_new_tokens"] = MAX_NEW_TOKENS
    else:
        print(f"  ⚠️  Neither max_completion_length nor max_new_tokens found in GRPOConfig")
    
    # Filter: only pass params that this TRL version's GRPOConfig actually accepts
    final_config = {}
    skipped = []
    for param, value in desired_config.items():
        if param in grpo_config_params:
            final_config[param] = value
        else:
            skipped.append(param)
    
    if skipped:
        print(f"  ⚠️  Skipped unsupported GRPOConfig params: {', '.join(skipped)}")
    
    dapo_keys = ["loss_type", "epsilon_high", "mask_truncated_completions",
                 "reward_weights", "multi_objective_aggregation", "beta"]
    dapo_enabled = [k for k in dapo_keys if k in final_config]
    if dapo_enabled:
        print(f"  ✅ DAPO params enabled: {', '.join(dapo_enabled)}")
    
    grpo_config = GRPOConfig(**final_config)
    
    # Train with multiple reward functions (research-informed)
    # Runtime estimate
    est_steps = len(dataset) // max(BATCH_SIZE, 1)
    avg_gen_tokens = min(MAX_NEW_TOKENS, 200)  # Most completions are <200 tokens
    est_gen_time_per_step = (NUM_GENERATIONS * avg_gen_tokens) / 3.0  # ~3 tok/s aggregate on 31B
    est_total_seconds = est_steps * est_gen_time_per_step
    est_hours = est_total_seconds / 3600
    
    print(f"\n{'=' * 60}")
    print(f"  Starting DAPO Training (GRPO successor, TRL v1.0)")
    print(f"  Samples: {len(dataset)}")
    print(f"  Generations per prompt: {NUM_GENERATIONS}")
    print(f"  Max completion tokens: {MAX_NEW_TOKENS}")
    print(f"  Estimated steps: {est_steps}")
    print(f"  Estimated runtime: {est_hours:.1f} hours")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Mode: DAPO (clip-higher + token-level loss)")
    print(f"  Reward weights (from YAML config):")
    for name, weight in REWARD_WEIGHTS.items():
        print(f"    {name:25s} w={weight}")
    print(f"  Aggregation: normalize_then_sum")
    print(f"{'=' * 60}\n")
    
    # GRPO FIX: Only correctness + faithfulness rewards.
    # format/structure/reasoning had ZERO variance (SFT already handles them)
    # → 60% of batches had frac_reward_zero_std → dead gradients.
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[
            correctness_reward,        # PRIMARY: is the answer correct? (the ONLY learning signal)
            faithfulness_reward,       # SECONDARY: does answer ground to input entities?
        ],
    )
    
    result = trainer.train()
    
    # Save final model
    print(f"\n  💾 Saving final model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metadata
    import json
    meta = {
        "stage": "GRPO",
        "base_checkpoint": SFT_CHECKPOINT,
        "learning_rate": LEARNING_RATE,
        "num_generations": NUM_GENERATIONS,
        "max_samples": MAX_SAMPLES,
    }
    with open(Path(OUTPUT_DIR) / "training_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✅ GRPO training complete!")
    print(f"\n  Next steps:")
    print(f"     python src/eval/evaluate.py          # Run evaluation")
    print(f"     python src/training/03_showcase_70b.py  # AMD showcase (optional)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
