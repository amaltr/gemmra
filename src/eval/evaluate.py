"""
Evaluation Script — Measure model performance on all tasks.
Run this on GPU after training.

Uses decontaminated eval set (MeditronFO-adopted: zero overlap with training).
Supports Gemma 4 thinking tokens (<|channel>thought ... <channel|>).

Flags:
  --quick       Fast sanity check: 200 stratified samples (~2 min)
  --full        Full eval: all samples (default if no flag)
  --no-base     Skip base model ablation

Computes: F1 (T1), MedDRA accuracy (T2), Labelling F1 (T3),
          Causality ordinal match (T4), Format compliance.
"""

import os
import re
import sys
import json
import time
import random
import torch
from pathlib import Path
from collections import Counter

# Add project root to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.meddra_soc import compute_t2_similarity, classify_soc

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from unsloth import FastLanguageModel

# Eval batch size — MI300X can handle 16 easily with 192GB HBM
EVAL_BATCH_SIZE = 16
MAX_NEW_TOKENS = 768  # 512 truncated T2 BioDEX thinking traces before answer field


# ============================================================
# Metrics
# ============================================================

def compute_f1(y_true: list, y_pred: list, positive_label: str = "YES") -> dict:
    """Compute precision, recall, F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def compute_format_compliance(outputs: list[str]) -> float:
    """What % of outputs have Gemma 4 thinking tokens or valid structured output?"""
    compliant = sum(
        1 for o in outputs
        if (re.search(r'<\|channel>thought.*?<channel\|>', o, re.DOTALL) or
            re.search(r'SERIOUS:\s*(YES|NO)', o, re.IGNORECASE) or
            re.search(r'LABELLED:\s*(YES|NO)', o, re.IGNORECASE) or
            re.search(r'MedDRA PT:', o, re.IGNORECASE) or
            re.search(r'WHO-UMC Causality:', o, re.IGNORECASE))
    )
    return compliant / len(outputs) if outputs else 0


def extract_answer_field(output: str, field: str) -> str | None:
    """Extract a field value from model output.
    
    Handles both Gemma 4 thinking format and direct output.
    M1 FIX: Strips trailing punctuation (periods, commas, semicolons)
    so 'Probable.' matches 'Probable'.
    """
    # Try to extract from after thinking tokens first
    after_think = re.sub(r'<\|channel>thought.*?<channel\|>', '', output, flags=re.DOTALL)
    if not after_think.strip():
        after_think = output
    
    field_match = re.search(rf'{field}:\s*(.+?)(?:\n|$)', after_think)
    if field_match:
        return field_match.group(1).strip().strip('*_').rstrip('.,;:!?')
    
    # Fallback: search entire output
    field_match = re.search(rf'{field}:\s*(.+?)(?:\n|$)', output)
    if field_match:
        return field_match.group(1).strip().strip('*_').rstrip('.,;:!?')
    return None


# ============================================================
# Prompt Formatting
# ============================================================

def _to_multimodal_msg(msg):
    """E1 FIX: Wrap plain string content into Gemma 4 multimodal format."""
    content = msg.get('content', '')
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {"role": msg["role"], "content": content}


def format_prompt(example, tokenizer):
    """Format a single example into a prompt string using chat template."""
    messages = example.get('messages', [])
    if messages:
        prompt_messages = [_to_multimodal_msg(m) for m in messages if m['role'] != 'assistant']
        # enable_thinking=True: Gemma 4 adds <|think|> to system turn instead of
        # pre-closing <|channel>thought\n<channel|>. Model generates actual thinking.
        try:
            return tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True
            )
        except TypeError:
            # Fallback if tokenizer doesn't support enable_thinking
            return tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
    else:
        return f"### Instruction:\n{example.get('instruction','')}\n\n### Input:\n{example.get('input','')}\n\n### Response:\n"


# ============================================================
# Batched Evaluation
# ============================================================

def evaluate_model(model, tokenizer, test_data: list[dict], model_name: str,
                   batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Run batched evaluation on test data.
    
    Uses left-padding for correct batched generation with decoder-only models.
    Processes batch_size prompts simultaneously → 4x speedup on MI300X.
    """
    print(f"\n  Evaluating: {model_name}")
    print(f"  Test cases: {len(test_data)} | Batch size: {batch_size}")
    
    FastLanguageModel.for_inference(model)
    
    # CRITICAL: Left-padding for batched generation.
    # Decoder-only models generate from the END of the sequence.
    # Right-padding puts pads between prompt and generation position → garbage.
    # Left-padding ensures all sequences align at the right → clean generation.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {"outputs": [], "t1": {"true": [], "pred": []}, 
               "t2": {"true": [], "pred": []}, "t3": {"true": [], "pred": []},
               "t4": {"true": [], "pred": []}}
    
    # Pre-format all prompts
    print(f"    Formatting prompts...")
    prompts = []
    for example in test_data:
        try:
            prompts.append(format_prompt(example, tokenizer))
        except Exception:
            # Fallback for any formatting errors
            prompts.append("")
    
    # Batched generation
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_examples = test_data[batch_start:batch_end]
        
        # Progress with ETA
        elapsed = time.time() - start_time
        if batch_idx > 0:
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            print(f"    Batch {batch_idx+1}/{total_batches} "
                  f"({batch_end}/{len(prompts)}) — ETA: {eta:.0f}s", end='\r')
        else:
            print(f"    Batch 1/{total_batches}...", end='\r')
        
        # Skip empty prompts
        valid_indices = [i for i, p in enumerate(batch_prompts) if p]
        if not valid_indices:
            for ex in batch_examples:
                results["outputs"].append("")
            continue
        
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        
        # Tokenize batch with left-padding
        # Gemma 4 processor requires text= kwarg (positional goes to images param)
        inputs = tokenizer(
            text=valid_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192
        ).to("cuda")
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding for eval (deterministic)
            )
        
        # C1 FIX: With left-padding, all prompts are right-aligned to length M.
        # Generated tokens start at index M (padded input length), NOT at
        # attention_mask.sum() (active token count P). Using P would include
        # trailing prompt tokens when P < M, corrupting the decoded output.
        generated_texts = []
        input_seq_len = inputs['input_ids'].shape[1]  # M = padded input length
        for j in range(len(valid_prompts)):
            gen_ids = output_ids[j][input_seq_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            text = text.replace(tokenizer.eos_token, '')
            if tokenizer.pad_token:
                text = text.replace(tokenizer.pad_token, '')
            generated_texts.append(text.strip())
        
        # Map back to batch (fill in empty prompts)
        gen_idx = 0
        for i, ex in enumerate(batch_examples):
            if i in valid_indices:
                generated = generated_texts[gen_idx]
                gen_idx += 1
            else:
                generated = ""
            
            results["outputs"].append(generated)
            
            # Score this example
            messages = ex.get('messages', [])
            if messages:
                expected = messages[-1].get('content', '') if messages[-1]['role'] == 'assistant' else ''
            else:
                expected = ex.get('output', '')
            
            task = ex.get('task', '')
            system_content = messages[0].get('content', '') if messages else ex.get('instruction', '')
            
            if task == 'T1' or 'seriousness' in system_content.lower():
                true_val = "YES" if "SERIOUS: YES" in expected else "NO"
                pred_val = extract_answer_field(generated, "SERIOUS") or ""
                # BUG FIX: "NO" in "UNKNOWN" = True. Use exact match instead.
                pred_upper = pred_val.strip().upper()
                if pred_upper == "YES" or pred_upper.startswith("YES"):
                    pred_val = "YES"
                elif pred_upper == "NO" or pred_upper.startswith("NO"):
                    pred_val = "NO"
                else:
                    pred_val = "UNKNOWN"
                results["t1"]["true"].append(true_val)
                results["t1"]["pred"].append(pred_val)
                
            elif task == 'T2' or 'meddra' in system_content.lower():
                true_pt = extract_answer_field(expected, "MedDRA PT") or ""
                pred_pt = extract_answer_field(generated, "MedDRA PT") or ""
                results["t2"]["true"].append(true_pt.lower().strip())
                results["t2"]["pred"].append(pred_pt.lower().strip())
            
            elif task == 'T3' or 'label' in system_content.lower():
                true_val = "YES" if "LABELLED: YES" in expected else "NO"
                pred_val = extract_answer_field(generated, "LABELLED") or ""
                # BUG FIX: same as T1 — exact match, not substring
                pred_upper = pred_val.strip().upper()
                if pred_upper == "YES" or pred_upper.startswith("YES"):
                    pred_val = "YES"
                elif pred_upper == "NO" or pred_upper.startswith("NO"):
                    pred_val = "NO"
                else:
                    pred_val = "UNKNOWN"
                results["t3"]["true"].append(true_val)
                results["t3"]["pred"].append(pred_val)
                
            elif task == 'T4' or 'causality' in system_content.lower():
                true_caus = extract_answer_field(expected, "WHO-UMC Causality") or ""
                pred_caus = extract_answer_field(generated, "WHO-UMC Causality") or ""
                results["t4"]["true"].append(true_caus.strip())
                results["t4"]["pred"].append(pred_caus.strip())
        
        # Free GPU memory between batches
        del inputs, output_ids
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n    Done: {len(test_data)} examples in {total_time:.1f}s "
          f"({total_time/len(test_data):.2f}s/example)")
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Compute metrics
    metrics = {}
    
    # Format compliance
    metrics["format_compliance"] = compute_format_compliance(results["outputs"])
    
    # T1 F1
    if results["t1"]["true"]:
        metrics["t1_seriousness"] = compute_f1(results["t1"]["true"], results["t1"]["pred"], "YES")
    
    # T2 Accuracy — multi-level scoring using MedDRA SOC classifier
    if results["t2"]["true"]:
        exact = 0
        synonym = 0
        fuzzy = 0
        soc = 0
        total_score = 0.0
        for t, p in zip(results["t2"]["true"], results["t2"]["pred"]):
            t_lower, p_lower = t.lower().strip(), p.lower().strip()
            if not p_lower or not t_lower:
                continue
            score = compute_t2_similarity(p_lower, t_lower)
            total_score += score
            if score >= 1.0:
                exact += 1
                synonym += 1
                fuzzy += 1
                soc += 1
            elif score >= 0.9:
                synonym += 1
                fuzzy += 1
                soc += 1
            elif score >= 0.7:
                fuzzy += 1
                soc += 1
            elif score >= 0.5:
                soc += 1
        n = len(results["t2"]["true"])
        metrics["t2_meddra"] = {
            "exact_match": exact / n,
            "synonym_match": synonym / n,
            "fuzzy_match": fuzzy / n,
            "soc_match": soc / n,
            "weighted_score": total_score / n,
        }
    
    # T3 F1
    if results["t3"]["true"]:
        metrics["t3_labelling"] = compute_f1(results["t3"]["true"], results["t3"]["pred"], "YES")
    
    # T4 Accuracy — ordinal distance with partial credit
    if results["t4"]["true"]:
        causality_order = ['Certain', 'Probable', 'Possible', 'Conditional', 'Unlikely', 'Unassessable']
        exact_match = 0
        partial_total = 0.0
        for t, p in zip(results["t4"]["true"], results["t4"]["pred"]):
            # M1 FIX: Strip non-alpha chars so 'Probable.' matches 'Probable'
            t_cap = re.sub(r'[^a-zA-Z]', '', t).capitalize()
            p_cap = re.sub(r'[^a-zA-Z]', '', p).capitalize()
            if t_cap == p_cap:
                exact_match += 1
                partial_total += 1.0
            elif t_cap in causality_order and p_cap in causality_order:
                dist = abs(causality_order.index(t_cap) - causality_order.index(p_cap))
                partial_total += max(0.0, 1.0 - dist * 0.25)
        n = len(results["t4"]["true"])
        metrics["t4_causality"] = {
            "exact_match": exact_match / n,
            "weighted_match": partial_total / n,
        }
    
    # Per-task counts
    metrics["sample_counts"] = {
        "t1": len(results["t1"]["true"]),
        "t2": len(results["t2"]["true"]),
        "t3": len(results["t3"]["true"]),
        "t4": len(results["t4"]["true"]),
        "total": len(test_data),
    }
    
    return metrics


# ============================================================
# Stratified Sampling
# ============================================================

def stratified_sample(test_data: list[dict], n_per_task: int = 50) -> list[dict]:
    """Sample n_per_task examples from each task for quick evaluation."""
    random.seed(42)  # Fixed seed for reproducible eval — cross-run comparison requires same samples
    by_task = {}
    for ex in test_data:
        task = ex.get('task', 'unknown')
        by_task.setdefault(task, []).append(ex)
    
    sampled = []
    for task, examples in sorted(by_task.items()):
        n = min(n_per_task, len(examples))
        sampled.extend(random.sample(examples, n))
    
    random.shuffle(sampled)
    return sampled


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Gemmra — Model Evaluation")
    print("=" * 60)
    
    # Parse flags
    quick_mode = '--quick' in sys.argv
    skip_base = '--no-base' in sys.argv
    
    # --checkpoint flag: eval ONE checkpoint only (skip base + others)
    single_ckpt = None
    for i, arg in enumerate(sys.argv):
        if arg == '--checkpoint' and i + 1 < len(sys.argv):
            single_ckpt = sys.argv[i + 1]
            skip_base = True  # Don't waste time on base when targeting one checkpoint
            break
    
    if quick_mode:
        print("\n  ⚡ QUICK MODE: 200 stratified samples for sanity check")
    
    # Load decontaminated eval data (MeditronFO-adopted)
    eval_file = Path("data/processed/eval_data.jsonl")
    train_file = Path("data/processed/training_data.jsonl")
    
    if eval_file.exists():
        with open(str(eval_file)) as f:
            test_data = [json.loads(line) for line in f]
        print(f"\n  Loaded decontaminated eval set: {len(test_data)} examples")
    elif train_file.exists():
        with open(str(train_file)) as f:
            all_data = [json.loads(line) for line in f]
        test_data = all_data[-100:]
        print(f"\n  eval_data.jsonl not found, using last 100 from training data")
        print(f"     WARNING: This may have data leakage!")
    else:
        print(f"  No data found")
        return 1
    
    # Quick mode: stratified sample
    if quick_mode:
        test_data = stratified_sample(test_data, n_per_task=50)
        print(f"  Sampled: {len(test_data)} examples (50 per task)")
    
    # Task distribution
    task_counts = Counter(ex.get('task', '?') for ex in test_data)
    print(f"  Tasks: {dict(task_counts)}")
    
    # Checkpoints to evaluate (order = most important first)
    checkpoints = [
        ("checkpoints/sft", "SFT Only (Primary)", False),
        ("checkpoints/sft_raft", "SFT + RAFT", False),
        ("checkpoints/wise_ft_a90", "SFT + WiSE-FT α=0.9", False),
        ("checkpoints/wise_ft_a80", "SFT + WiSE-FT α=0.8", False),
        ("checkpoints/wise_ft_a70", "SFT + WiSE-FT α=0.7", False),
        ("checkpoints/grpo_final", "SFT + GRPO (Final)", False),
        ("checkpoints/local_grpo", "Local SFT + GRPO", True),
        ("checkpoints/local_sft", "Local SFT Only", True),
    ]
    
    # If --checkpoint specified, only eval that one
    if single_ckpt:
        checkpoints = [(single_ckpt, Path(single_ckpt).name, False)]
    
    all_metrics = {}
    total_start = time.time()
    
    # Base model ablation (optional, skip 12B — only 27B matters)
    if not skip_base:
        base_model = "google/gemma-4-31b-it"
        try:
            print(f"\n  Evaluating base model: {base_model}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model, max_seq_length=8192, load_in_4bit=False,
                dtype=torch.bfloat16,
            )
            metrics = evaluate_model(model, tokenizer, test_data, f"Base: {base_model}")
            all_metrics[f"Base: {base_model}"] = metrics
            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Skipping base model: {e}")
    
    # Evaluate checkpoints
    for ckpt_path, ckpt_name, is_local in checkpoints:
        if Path(ckpt_path).exists():
            load_4bit = is_local  # Local checkpoints trained with 4-bit
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt_path, max_seq_length=8192, load_in_4bit=load_4bit,
                dtype=torch.bfloat16,
            )
            metrics = evaluate_model(model, tokenizer, test_data, ckpt_name)
            all_metrics[ckpt_name] = metrics
            
            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            print(f"\n  Skipping {ckpt_name} — not found at {ckpt_path}")
    
    total_time = time.time() - total_start
    
    # Print comparison
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS  (total: {total_time:.0f}s)")
    print(f"{'=' * 60}")
    
    for name, metrics in all_metrics.items():
        print(f"\n  {name}:")
        print(f"     Format Compliance: {metrics.get('format_compliance', 0):.1%}")
        if 't1_seriousness' in metrics:
            t1 = metrics['t1_seriousness']
            print(f"     T1 Seriousness F1: {t1['f1']:.3f} (P={t1['precision']:.3f}, R={t1['recall']:.3f})")
        if 't2_meddra' in metrics:
            t2 = metrics['t2_meddra']
            print(f"     T2 MedDRA: exact={t2['exact_match']:.3f} synonym={t2['synonym_match']:.3f} fuzzy={t2['fuzzy_match']:.3f} soc={t2['soc_match']:.3f} weighted={t2['weighted_score']:.3f}")
        if 't3_labelling' in metrics:
            t3 = metrics['t3_labelling']
            print(f"     T3 Labelling F1: {t3['f1']:.3f} (P={t3['precision']:.3f}, R={t3['recall']:.3f})")
        if 't4_causality' in metrics:
            t4 = metrics['t4_causality']
            print(f"     T4 Causality: exact={t4['exact_match']:.3f} weighted={t4['weighted_match']:.3f}")
        if 'sample_counts' in metrics:
            sc = metrics['sample_counts']
            print(f"     Samples: T1={sc['t1']} T2={sc['t2']} T3={sc['t3']} T4={sc['t4']}")
    
    # Save results
    output_path = Path("experiments/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
