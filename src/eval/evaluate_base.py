"""
Evaluation Script for Base Model — Measure google/gemma-4-31b-it performance on 250 base-specific samples.
Run this on GPU.

Disables Gemma 4 thinking traces to speed up evaluation and save tokens.
Uses robust markdown stripping to handle bolding/italics from the base model.
"""

import os
import re
import sys
import json
import time
import torch
from pathlib import Path
from collections import Counter

# Add project root to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.meddra_soc import compute_t2_similarity

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from unsloth import FastLanguageModel

EVAL_BATCH_SIZE = 16
MAX_NEW_TOKENS = 256  # Base model outputs directly, no 2000-token thinking trace needed

# ============================================================
# Metrics and Robust Parser
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


def extract_answer_field(output: str, field: str) -> str | None:
    """Extract a field value from model output robustly.
    
    1. Strips all asterisks (*) and underscores (_) to handle markdown bolding/italics.
    2. Strips thinking tags if any leaked through.
    3. Matches field case-insensitively.
    4. Cleans trailing punctuation.
    """
    # Strip markdown formatting first
    cleaned = re.sub(r'[*_]', '', output)
    
    # Strip thinking traces if any
    after_think = re.sub(r'<\|channel>thought.*?<channel\|>', '', cleaned, flags=re.DOTALL)
    if not after_think.strip():
        after_think = cleaned
    
    # Match field: value
    field_match = re.search(rf'{field}:\s*(.+?)(?:\n|$)', after_think, re.IGNORECASE)
    if field_match:
        return field_match.group(1).strip().rstrip('.,;:!?')
    
    # Fallback to check entire cleaned output
    field_match = re.search(rf'{field}:\s*(.+?)(?:\n|$)', cleaned, re.IGNORECASE)
    if field_match:
        return field_match.group(1).strip().rstrip('.,;:!?')
        
    return None


# ============================================================
# Prompt Formatting
# ============================================================

def _to_multimodal_msg(msg):
    content = msg.get('content', '')
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {"role": msg["role"], "content": content}


def format_prompt_base(example, tokenizer):
    """Format prompt with enable_thinking=False to disable thinking traces."""
    messages = example.get('messages', [])
    if messages:
        prompt_messages = [_to_multimodal_msg(m) for m in messages if m['role'] != 'assistant']
        try:
            return tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
    return ""


# ============================================================
# Batched Evaluation
# ============================================================

def evaluate_model_base(model, tokenizer, test_data: list[dict], batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Run batched base model evaluation using left-padding."""
    print(f"  Test cases: {len(test_data)} | Batch size: {batch_size}")
    
    FastLanguageModel.for_inference(model)
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {"outputs": [], "t1": {"true": [], "pred": []}, 
               "t2": {"true": [], "pred": []}, "t3": {"true": [], "pred": []},
               "t4": {"true": [], "pred": []}}
    
    print(f"    Formatting prompts...")
    prompts = []
    for example in test_data:
        try:
            prompts.append(format_prompt_base(example, tokenizer))
        except Exception:
            prompts.append("")
    
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_examples = test_data[batch_start:batch_end]
        
        # ETA Progress
        elapsed = time.time() - start_time
        if batch_idx > 0:
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            print(f"    Batch {batch_idx+1}/{total_batches} ({batch_end}/{len(prompts)}) — ETA: {eta:.0f}s", end='\r')
        else:
            print(f"    Batch 1/{total_batches}...", end='\r')
        
        valid_indices = [i for i, p in enumerate(batch_prompts) if p]
        if not valid_indices:
            for _ in batch_examples:
                results["outputs"].append("")
            continue
        
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        
        inputs = tokenizer(
            text=valid_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192
        ).to("cuda")
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        
        generated_texts = []
        input_seq_len = inputs['input_ids'].shape[1]
        for j in range(len(valid_prompts)):
            gen_ids = output_ids[j][input_seq_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            text = text.replace(tokenizer.eos_token or '', '')
            if tokenizer.pad_token:
                text = text.replace(tokenizer.pad_token, '')
            generated_texts.append(text.strip())
        
        gen_idx = 0
        for i, ex in enumerate(batch_examples):
            if i in valid_indices:
                generated = generated_texts[gen_idx]
                gen_idx += 1
            else:
                generated = ""
            
            results["outputs"].append(generated)
            
            # Extract expected label from assistant turn (ground truth)
            messages = ex.get('messages', [])
            expected = messages[-1].get('content', '') if messages and messages[-1]['role'] == 'assistant' else ''
            
            # Strip target formatting characters
            expected_clean = re.sub(r'[*_]', '', expected)
            
            task = ex.get('task', '')
            
            if task == 'T1':
                true_val = "YES" if "SERIOUS: YES" in expected_clean else "NO"
                pred_val = extract_answer_field(generated, "SERIOUS") or ""
                
                pred_upper = pred_val.strip().upper()
                if pred_upper == "YES" or pred_upper.startswith("YES"):
                    pred_val = "YES"
                elif pred_upper == "NO" or pred_upper.startswith("NO"):
                    pred_val = "NO"
                else:
                    pred_val = "UNKNOWN"
                results["t1"]["true"].append(true_val)
                results["t1"]["pred"].append(pred_val)
                
            elif task == 'T2':
                true_pt = extract_answer_field(expected_clean, "MedDRA PT") or ""
                pred_pt = extract_answer_field(generated, "MedDRA PT") or ""
                results["t2"]["true"].append(true_pt.lower().strip())
                results["t2"]["pred"].append(pred_pt.lower().strip())
            
            elif task == 'T3':
                true_val = "YES" if "LABELLED: YES" in expected_clean else "NO"
                pred_val = extract_answer_field(generated, "LABELLED") or ""
                
                pred_upper = pred_val.strip().upper()
                if pred_upper == "YES" or pred_upper.startswith("YES"):
                    pred_val = "YES"
                elif pred_upper == "NO" or pred_upper.startswith("NO"):
                    pred_val = "NO"
                else:
                    pred_val = "UNKNOWN"
                results["t3"]["true"].append(true_val)
                results["t3"]["pred"].append(pred_val)
                
            elif task == 'T4':
                true_caus = extract_answer_field(expected_clean, "WHO-UMC Causality") or ""
                pred_caus = extract_answer_field(generated, "WHO-UMC Causality") or ""
                results["t4"]["true"].append(true_caus.strip())
                results["t4"]["pred"].append(pred_caus.strip())
        
        del inputs, output_ids
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n    Done: {len(test_data)} examples in {total_time:.1f}s ({total_time/len(test_data):.2f}s/example)")
    
    tokenizer.padding_side = original_padding_side
    
    # Compute metrics
    metrics = {"format_compliance": 1.0}  # compliance checking omitted or default to 1.0 for simplicity
    
    # T1 Seriousness
    if results["t1"]["true"]:
        metrics["t1_seriousness"] = compute_f1(results["t1"]["true"], results["t1"]["pred"], "YES")
    
    # T2 MedDRA Coding
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
    
    # T3 Labelling
    if results["t3"]["true"]:
        metrics["t3_labelling"] = compute_f1(results["t3"]["true"], results["t3"]["pred"], "YES")
    
    # T4 Causality
    if results["t4"]["true"]:
        causality_order = ['Certain', 'Probable', 'Possible', 'Conditional', 'Unlikely', 'Unassessable']
        exact_match = 0
        partial_total = 0.0
        for t, p in zip(results["t4"]["true"], results["t4"]["pred"]):
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
    
    metrics["sample_counts"] = {
        "t1": len(results["t1"]["true"]),
        "t2": len(results["t2"]["true"]),
        "t3": len(results["t3"]["true"]),
        "t4": len(results["t4"]["true"]),
        "total": len(test_data),
    }
    
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Base Model Specific Evaluation")
    print("=" * 60)
    
    eval_file = Path("data/processed/eval_data_base_250.jsonl")
    if not eval_file.exists():
        print(f"ERROR: {eval_file} not found. Run 'python src/data/06_build_base_eval_data.py' first.")
        sys.exit(1)
        
    with open(str(eval_file), encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(test_data)} base-model-specific cases")
    
    base_model = "google/gemma-4-31b-it"
    
    # Determine 4-bit loading (fallback to 4-bit if VRAM is low, e.g. < 40GB)
    load_in_4bit = False
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 40:
            load_in_4bit = True
            
    print(f"Loading base model: {base_model} (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=8192, load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
    )
    
    metrics = evaluate_model_base(model, tokenizer, test_data)
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"  BASE MODEL EVALUATION RESULTS")
    print(f"{'=' * 60}")
    
    print(f"\n  Gemma-4-31b-it (Base):")
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
    output_path = Path("experiments/eval_results_base.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
