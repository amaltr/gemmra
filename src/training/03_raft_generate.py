"""
RAFT (Rejection Sampling Fine-Tuning) — Stage 2 Alternative

Instead of GRPO (which failed due to reward variance collapse), this script:
1. Loads the SFT model checkpoint
2. Generates N completions per training prompt (T2 focus, optional T3)
3. Scores each completion against ground truth
4. Keeps ONLY correct completions (acceptance threshold)
5. Saves as augmented training data (JSONL)

Then re-run 01_sft_train.py with the combined data.

WHY THIS WORKS WHEN GRPO DOESN'T:
- GRPO needs reward VARIANCE within a group → fails when model is already good (8/8 same answer)
- RAFT only needs SOME correct generations → even 1/8 correct is enough
- No group advantage computation → no dead gradients
- Model learns its OWN successful reasoning patterns → better generalization
- Standard SFT training dynamics → proven stable

Usage:
  python src/training/03_raft_generate.py --smoke     # 20 T2 samples, ~10 min
  python src/training/03_raft_generate.py              # All T2 + T3, ~2-3 hrs
  python src/training/03_raft_generate.py --tasks T2   # T2 only
  python src/training/03_raft_generate.py --tasks T2 T3 --n 16  # 16 gens per prompt

After RAFT generation:
  python src/training/01_sft_train.py                 # Retrain with augmented data
"""

import os
import sys
import json
import re
import time
import random
import argparse
import torch
from pathlib import Path
from collections import Counter

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from unsloth import FastLanguageModel

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.meddra_soc import compute_t2_similarity


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "checkpoints/sft"
TRAIN_FILE = "data/processed/training_data.jsonl"
EVAL_FILE = "data/processed/eval_data.jsonl"
OUTPUT_FILE = "data/processed/raft_augmented_data.jsonl"
COMBINED_FILE = "data/processed/training_data_raft.jsonl"  # Original + RAFT

# RAFT hyperparameters (evidence-based choices documented inline)
NUM_GENERATIONS = 8       # Same as GRPO. 8 gives good diversity-efficiency tradeoff.
TEMPERATURE = 0.8         # Lower than GRPO's 1.5: we want quality, not exploration.
                          # 0.8 produces varied-but-coherent reasoning paths.
TOP_P = 0.95              # Nucleus sampling for diversity within quality range.
MAX_NEW_TOKENS = 512      # Match eval/check_raw inference settings.
MAX_SEQ_LENGTH = 8192     # Match SFT training.

# Acceptance thresholds (per task)
# T2: fuzzy match (0.7) = substring/word-overlap match or better
# T3: exact binary match only (1.0)
ACCEPT_THRESHOLD = {
    'T2': 0.7,   # Accept fuzzy match or better (0.7/0.9/1.0)
    'T3': 1.0,   # Accept exact match only (YES/NO is binary, no partial credit)
}

# Which tasks to target (T1=0.995, T4=0.986 — already at ceiling, skip)
DEFAULT_TASKS = ['T2', 'T3']


# ============================================================
# Argument Parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RAFT: Rejection Sampling Fine-Tuning")
    parser.add_argument('--smoke', action='store_true',
                        help='Quick validation: 20 T2 samples only (~10 min)')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Tasks to target (default: T2 T3). Example: --tasks T2')
    parser.add_argument('--n', type=int, default=NUM_GENERATIONS,
                        help=f'Generations per prompt (default: {NUM_GENERATIONS})')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override acceptance threshold (e.g., 0.5 for more permissive)')
    parser.add_argument('--ckpt', type=str, default=CHECKPOINT,
                        help=f'Checkpoint path (default: {CHECKPOINT})')
    parser.add_argument('--skip-combine', action='store_true',
                        help='Only generate RAFT data, do not combine with original')
    return parser.parse_args()


# ============================================================
# Model Loading
# ============================================================

def load_model(ckpt_path):
    """Load SFT checkpoint for inference."""
    if not Path(ckpt_path).exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"     Run first: python src/training/01_sft_train.py")
        sys.exit(1)

    load_in_4bit = False
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 40:
            load_in_4bit = True

    print(f"  Loading: {ckpt_path} (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  ✅ Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


# ============================================================
# Data Loading
# ============================================================

def load_training_data(tasks, max_samples=None):
    """Load training samples for target tasks."""
    data_path = Path(TRAIN_FILE)
    if not data_path.exists():
        print(f"  ❌ Training data not found: {TRAIN_FILE}")
        sys.exit(1)

    with open(str(data_path), encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # Filter to target tasks
    filtered = [d for d in all_data if d.get('task', '') in tasks]

    if max_samples:
        random.seed(42)
        random.shuffle(filtered)
        filtered = filtered[:max_samples]

    by_task = Counter(d.get('task', '') for d in filtered)
    print(f"  Loaded {len(filtered)} samples for RAFT:")
    for t, c in sorted(by_task.items()):
        print(f"    {t}: {c}")

    return filtered, all_data


# ============================================================
# Prompt Formatting (must match check_raw.py / evaluate.py)
# ============================================================

def format_prompt(messages, tokenizer):
    """Format messages into prompt string using tokenizer's chat template."""
    prompt_msgs = []
    for m in messages:
        if m['role'] != 'assistant':
            content = m.get('content', '')
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            prompt_msgs.append({"role": m["role"], "content": content})

    try:
        return tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )


# ============================================================
# Ground Truth Extraction
# ============================================================

def extract_gt_label(messages, task):
    """Extract ground truth answer label from assistant message."""
    asst = [m for m in messages if m['role'] == 'assistant']
    if not asst:
        return ''

    content = asst[0].get('content', '')
    # Strip thinking trace to get answer-only
    answer = re.sub(r'<\|channel>thought.*?<channel\|>', '', content, flags=re.DOTALL).strip()

    if task == 'T1':
        m = re.search(r'SERIOUS:\s*(YES|NO)', answer, re.I)
        return m.group(1).upper() if m else ''
    elif task == 'T2':
        m = re.search(r'MedDRA PT:\s*(.+?)(?:\n|$)', answer, re.I)
        return m.group(1).strip().rstrip('.,;:!?') if m else ''
    elif task == 'T3':
        m = re.search(r'LABELLED:\s*(YES|NO)', answer, re.I)
        return m.group(1).upper() if m else ''
    elif task == 'T4':
        m = re.search(r'WHO-UMC Causality:\s*(\w+)', answer, re.I)
        return m.group(1).strip().capitalize() if m else ''
    return ''


def extract_pred_label(generated_text, task):
    """Extract prediction from generated text (strip thinking trace)."""
    answer = re.sub(r'<\|channel>thought.*?<channel\|>', '', generated_text, flags=re.DOTALL).strip()
    # Also handle missing <channel|> — split on double newline after thinking
    if '<|channel>thought' in generated_text and answer == generated_text.strip():
        parts = re.split(r'\n\n', generated_text, maxsplit=1)
        if len(parts) > 1:
            candidate = parts[1].strip()
            if re.search(r'SERIOUS:|MedDRA PT:|LABELLED:|WHO-UMC', candidate, re.I):
                answer = candidate

    if task == 'T2':
        m = re.search(r'MedDRA PT:\s*(.+?)(?:\n|$)', answer, re.I)
        return m.group(1).strip().rstrip('.,;:!?') if m else ''
    elif task == 'T3':
        m = re.search(r'LABELLED:\s*(YES|NO)', answer, re.I)
        return m.group(1).upper() if m else ''
    return ''


# ============================================================
# Scoring
# ============================================================

def score_completion(pred_label, gt_label, task):
    """Score a completion against ground truth.

    Returns:
        float: Score between 0.0 and 1.0
    """
    if not pred_label or not gt_label:
        return 0.0

    if task == 'T2':
        return compute_t2_similarity(pred_label, gt_label)
    elif task == 'T3':
        return 1.0 if pred_label.upper() == gt_label.upper() else 0.0
    return 0.0


# ============================================================
# RAFT Generation Core
# ============================================================

def generate_and_filter(model, tokenizer, samples, args):
    """Generate N completions per prompt, score, filter.

    Returns:
        list: Accepted (prompt, completion) pairs as training examples
        dict: Statistics for debugging
    """
    n_gen = args.n
    thresholds = dict(ACCEPT_THRESHOLD)
    if args.threshold is not None:
        for t in thresholds:
            thresholds[t] = args.threshold

    stats = {
        'total_prompts': 0,
        'total_generations': 0,
        'accepted': 0,
        'rejected': 0,
        'by_task': {},
        'score_distribution': [],
        'acceptance_rate_per_prompt': [],
        'samples': [],  # First 5 for debugging
    }

    accepted_examples = []
    start_time = time.time()

    for idx, sample in enumerate(samples):
        task = sample.get('task', '')
        messages = sample.get('messages', [])
        gt_label = extract_gt_label(messages, task)
        threshold = thresholds.get(task, 0.7)

        if not gt_label:
            continue

        stats['total_prompts'] += 1

        # Format prompt
        prompt_text = format_prompt(messages, tokenizer)

        # Tokenize — Gemma 4 processor requires text= kwarg with list
        inputs = tokenizer(
            text=[prompt_text], return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        # Generate N completions one at a time
        # (num_return_sequences>1 can OOM on large models; loop is safer)
        all_gen_texts = []
        for _gi in range(n_gen):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_tokens = output[0, prompt_len:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            # Clean up EOS/pad tokens but keep thinking markers
            eos_tok = tokenizer.eos_token or ''
            pad_tok = tokenizer.pad_token or ''
            if eos_tok:
                generated_text = generated_text.replace(eos_tok, '')
            if pad_tok and pad_tok != eos_tok:
                generated_text = generated_text.replace(pad_tok, '')
            generated_text = generated_text.strip()
            all_gen_texts.append(generated_text)

        prompt_accepted = 0
        prompt_scores = []

        for gen_idx, generated_text in enumerate(all_gen_texts):
            stats['total_generations'] += 1

            # Extract prediction and score
            pred_label = extract_pred_label(generated_text, task)
            score = score_completion(pred_label, gt_label, task)
            prompt_scores.append(score)

            if score >= threshold:
                stats['accepted'] += 1
                prompt_accepted += 1

                # Build training example in same format as original data
                # Use model's OWN generated text as the completion
                new_example = {
                    'task': task,
                    'primaryid': sample.get('primaryid', '') + f'_raft_{gen_idx}',
                    'messages': [
                        m for m in messages if m['role'] != 'assistant'
                    ] + [
                        {"role": "assistant", "content": generated_text}
                    ],
                    'raft_score': score,
                    'raft_gt': gt_label,
                    'raft_pred': pred_label,
                }
                accepted_examples.append(new_example)

                # Save debug sample (first 5)
                if len(stats['samples']) < 5:
                    stats['samples'].append({
                        'task': task,
                        'gt': gt_label,
                        'pred': pred_label,
                        'score': score,
                        'generated_len': len(generated_text.split()),
                        'generated_preview': generated_text[:300],
                    })
            else:
                stats['rejected'] += 1

        stats['score_distribution'].extend(prompt_scores)
        stats['acceptance_rate_per_prompt'].append(prompt_accepted / n_gen)

        # Track per-task stats
        if task not in stats['by_task']:
            stats['by_task'][task] = {'prompts': 0, 'accepted': 0, 'total_gen': 0}
        stats['by_task'][task]['prompts'] += 1
        stats['by_task'][task]['accepted'] += prompt_accepted
        stats['by_task'][task]['total_gen'] += n_gen

        # Progress logging
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(samples) - idx - 1) / rate if rate > 0 else 0

        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            print(f"  [{idx+1}/{len(samples)}] "
                  f"accept={stats['accepted']}/{stats['total_generations']} "
                  f"({100*stats['accepted']/max(1,stats['total_generations']):.1f}%) "
                  f"| scores={prompt_scores} "
                  f"| ETA: {remaining/60:.1f}m")

    stats['elapsed'] = time.time() - start_time
    return accepted_examples, stats


# ============================================================
# Statistics & Debugging Output
# ============================================================

def print_stats(stats):
    """Print comprehensive RAFT statistics for debugging."""
    print(f"\n{'=' * 60}")
    print(f"  RAFT GENERATION RESULTS")
    print(f"{'=' * 60}")

    total = stats['total_generations']
    accepted = stats['accepted']
    rate = 100 * accepted / max(1, total)

    print(f"  Total prompts:     {stats['total_prompts']}")
    print(f"  Total generations: {total}")
    print(f"  Accepted:          {accepted} ({rate:.1f}%)")
    print(f"  Rejected:          {stats['rejected']}")
    print(f"  Runtime:           {stats['elapsed']/60:.1f} min")

    print(f"\n  Per-Task Breakdown:")
    for task, ts in sorted(stats['by_task'].items()):
        task_rate = 100 * ts['accepted'] / max(1, ts['total_gen'])
        avg_accept = ts['accepted'] / max(1, ts['prompts'])
        print(f"    {task}: {ts['accepted']}/{ts['total_gen']} accepted "
              f"({task_rate:.1f}%, avg {avg_accept:.1f}/prompt)")

    # Score distribution
    scores = stats['score_distribution']
    if scores:
        buckets = Counter()
        for s in scores:
            if s == 0.0:
                buckets['0.0 (miss)'] += 1
            elif s < 0.5:
                buckets['0.1-0.4 (SOC only)'] += 1
            elif s < 0.7:
                buckets['0.5-0.6 (SOC match)'] += 1
            elif s < 0.9:
                buckets['0.7-0.8 (fuzzy)'] += 1
            elif s < 1.0:
                buckets['0.9 (synonym)'] += 1
            else:
                buckets['1.0 (exact)'] += 1

        print(f"\n  Score Distribution:")
        for bucket in ['0.0 (miss)', '0.1-0.4 (SOC only)', '0.5-0.6 (SOC match)',
                        '0.7-0.8 (fuzzy)', '0.9 (synonym)', '1.0 (exact)']:
            count = buckets.get(bucket, 0)
            pct = 100 * count / max(1, len(scores))
            bar = '█' * int(pct / 2)
            print(f"    {bucket:25s} {count:5d} ({pct:5.1f}%) {bar}")

    # Per-prompt acceptance rate distribution
    rates = stats['acceptance_rate_per_prompt']
    if rates:
        avg_rate = sum(rates) / len(rates)
        zero_prompts = sum(1 for r in rates if r == 0)
        full_prompts = sum(1 for r in rates if r == 1.0)
        print(f"\n  Per-Prompt Acceptance:")
        print(f"    Average: {avg_rate:.2f} ({avg_rate * 100:.1f}%)")
        print(f"    0/N (all rejected): {zero_prompts} prompts ({100*zero_prompts/max(1,len(rates)):.1f}%)")
        print(f"    N/N (all accepted): {full_prompts} prompts ({100*full_prompts/max(1,len(rates)):.1f}%)")

    # Debug samples
    if stats['samples']:
        print(f"\n  Sample Accepted Completions:")
        for i, s in enumerate(stats['samples'][:3]):
            print(f"\n  --- Sample {i+1} ({s['task']}) score={s['score']} ---")
            print(f"  GT:   {s['gt']}")
            print(f"  PRED: {s['pred']}")
            print(f"  Generated ({s['generated_len']} words):")
            print(f"    {s['generated_preview'][:200]}...")


# ============================================================
# Data Combination
# ============================================================

def combine_data(raft_examples, original_data_path, output_path):
    """Combine original training data with RAFT-augmented examples.

    Strategy:
    - Keep ALL original training data (preserves T1, T4 performance)
    - Add RAFT-accepted T2/T3 examples
    - Shuffle combined dataset
    - No overweighting — model's own correct generations are high quality
    """
    print(f"\n  Combining data...")

    # Load original
    with open(str(original_data_path), encoding='utf-8') as f:
        original = [json.loads(line) for line in f if line.strip()]

    # Strip RAFT metadata before combining (keep messages/task/primaryid only)
    clean_raft = []
    for ex in raft_examples:
        clean = {
            'task': ex['task'],
            'primaryid': ex['primaryid'],
            'messages': ex['messages'],
        }
        # Build text field (required by some pipelines)
        clean['text'] = json.dumps(ex['messages'])
        clean_raft.append(clean)

    combined = original + clean_raft
    random.seed(42)
    random.shuffle(combined)

    # Save
    with open(str(output_path), 'w', encoding='utf-8') as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    by_task = Counter(d.get('task', '') for d in combined)
    raft_by_task = Counter(d.get('task', '') for d in clean_raft)

    print(f"  ✅ Combined dataset saved: {output_path}")
    print(f"     Original: {len(original)}")
    print(f"     RAFT:     {len(clean_raft)}")
    print(f"     Total:    {len(combined)}")
    print(f"\n     By task (original + RAFT):")
    for t in sorted(by_task):
        orig_count = sum(1 for d in original if d.get('task', '') == t)
        raft_count = raft_by_task.get(t, 0)
        print(f"       {t}: {orig_count} + {raft_count} = {by_task[t]}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("  Gemmra — RAFT (Rejection Sampling Fine-Tuning)")
    print("=" * 60)

    # Determine tasks
    tasks = args.tasks or DEFAULT_TASKS
    tasks = [t.upper() for t in tasks]

    # Smoke mode
    max_samples = None
    if args.smoke:
        max_samples = 20
        tasks = ['T2']  # Only T2 for smoke (most interesting)
        print(f"\n  🔥 SMOKE MODE: {max_samples} T2 samples, {args.n} gens each")
        print(f"     Purpose: verify RAFT pipeline works before full run")
        print(f"     Expected runtime: ~10-15 min on MI300X")

    print(f"\n  Configuration:")
    print(f"    Checkpoint:   {args.ckpt}")
    print(f"    Tasks:        {tasks}")
    print(f"    Generations:  {args.n} per prompt")
    print(f"    Temperature:  {TEMPERATURE}")
    print(f"    Top-p:        {TOP_P}")
    print(f"    Max tokens:   {MAX_NEW_TOKENS}")
    print(f"    Thresholds:   {ACCEPT_THRESHOLD}")
    if args.threshold:
        print(f"    Override:     all tasks → {args.threshold}")

    # Load model
    model, tokenizer = load_model(args.ckpt)

    # Load data
    samples, all_data = load_training_data(tasks, max_samples=max_samples)

    if not samples:
        print("  ❌ No samples found for target tasks!")
        return 1

    # Generate and filter
    print(f"\n  Starting RAFT generation...")
    print(f"  Generating {args.n} completions × {len(samples)} prompts = "
          f"{args.n * len(samples)} total generations")
    est_time = len(samples) * args.n * 3 / 60  # ~3 sec per generation
    print(f"  Estimated runtime: ~{est_time:.0f} min\n")

    accepted, stats = generate_and_filter(model, tokenizer, samples, args)

    # Print stats
    print_stats(stats)

    if not accepted:
        print(f"\n  ⚠️  No completions accepted! Check thresholds or model quality.")
        print(f"     Try: --threshold 0.5 for more permissive acceptance")
        return 1

    # Save RAFT-only data
    raft_path = Path(OUTPUT_FILE)
    raft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(raft_path), 'w', encoding='utf-8') as f:
        for ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"\n  💾 RAFT data saved: {raft_path} ({len(accepted)} examples)")

    # Combine with original data
    if not args.skip_combine and not args.smoke:
        combine_data(accepted, TRAIN_FILE, COMBINED_FILE)
        print(f"\n  Next steps:")
        print(f"     1. Update configs/sft_config.yaml:")
        print(f"        train_file: 'data/processed/training_data_raft.jsonl'")
        print(f"     2. Retrain SFT:")
        print(f"        python src/training/01_sft_train.py")
        print(f"     3. Evaluate:")
        print(f"        python src/eval/evaluate.py --quick")
        print(f"     4. Debug check:")
        print(f"        python check_raw.py --task T2 --n 10")
    elif args.smoke:
        print(f"\n  🔥 Smoke test complete!")
        print(f"     Acceptance rate: {100*stats['accepted']/max(1,stats['total_generations']):.1f}%")
        if stats['accepted'] > 0:
            print(f"     ✅ Pipeline works. Run without --smoke for full RAFT.")
        else:
            print(f"     ⚠️  Zero accepted. Lower threshold with --threshold 0.5")

    return 0


if __name__ == "__main__":
    sys.exit(main())
