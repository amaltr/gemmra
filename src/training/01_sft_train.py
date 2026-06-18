"""
SFT Training Script — Stage 1 of the training pipeline.
Run this on GPU (AMD MI300X — 192 GB VRAM).

Loads base model in bf16 (no quantization) → applies LoRA (r=64) → trains → saves.
See docs/decisions/ADR-003-training-strategy.md for rationale.

Why bf16 instead of 4-bit QLoRA:
- MI300X has 192 GB VRAM; bf16 LoRA uses ~95 GB → no need to quantize
- No quantization noise → better gradient quality → better fine-tuning
- Eliminates bitsandbytes dependency for model loading (failure point #1)
- MI300X bf16 TFLOPS are massive → may actually train faster
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Ensure AMD environment
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================
# Configuration — loaded from configs/sft_config.yaml with defaults
# ============================================================

def _load_yaml_config(config_path: str = None) -> dict:
    """Load config from YAML file, returning empty dict if not found."""
    if config_path is None:
        # Support --config path/to/config.yaml for RAFT and other custom configs
        if '--config' in sys.argv:
            idx = sys.argv.index('--config')
            if idx + 1 < len(sys.argv):
                config_path = sys.argv[idx + 1]
            else:
                config_path = "configs/sft_config.yaml"
        elif '--local' in sys.argv:
            config_path = "configs/local_sft_config.yaml"
        else:
            config_path = "configs/sft_config.yaml"
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
_lora_cfg = _cfg.get('lora', {})
_train_cfg = _cfg.get('training', {})
_data_cfg = _cfg.get('data', {})
_out_cfg = _cfg.get('output', {})

# Model — Gemma 4 31B primary (thinking mode + best reasoning in class)
PRIMARY_MODEL = _model_cfg.get('name', "google/gemma-4-31b-it")
FALLBACK_MODELS = [
    _model_cfg.get('fallback_1', "google/gemma-4-12b-it"),
]

MAX_SEQ_LENGTH = _model_cfg.get('max_seq_length', 8192)
LOAD_IN_4BIT = _model_cfg.get('load_in_4bit', False)
DTYPE = torch.bfloat16

# LoRA — r=64 for deep domain adaptation (192 GB VRAM supports this easily)
LORA_R = _lora_cfg.get('r', 64)
LORA_ALPHA = _lora_cfg.get('alpha', 128)
LORA_DROPOUT = _lora_cfg.get('dropout', 0)
TARGET_MODULES = _lora_cfg.get('target_modules', [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
])

# Training — larger batch leverages 192 GB VRAM headroom
TRAIN_FILE = _data_cfg.get('train_file', "data/processed/training_data.jsonl")
OUTPUT_DIR = _out_cfg.get('dir', "checkpoints/sft")
BATCH_SIZE = _train_cfg.get('per_device_batch_size', 8)
GRAD_ACCUM = _train_cfg.get('gradient_accumulation_steps', 4)
LEARNING_RATE = _train_cfg.get('learning_rate', 5e-5)
NUM_EPOCHS = _train_cfg.get('num_epochs', 1)
WARMUP_RATIO = _train_cfg.get('warmup_ratio', 0.05)
SAVE_STEPS = _train_cfg.get('save_steps', 200)
EVAL_STEPS = _train_cfg.get('eval_steps', 200)
EVAL_FILE = _data_cfg.get('eval_file', 'data/processed/eval_data.jsonl')
MAX_SAMPLES = _train_cfg.get('max_samples', None)


def load_model():
    """Load base model with fallback chain."""
    all_models = [PRIMARY_MODEL] + FALLBACK_MODELS
    for model_name in all_models:
        print(f"\n  Loading: {model_name} ...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
                dtype=DTYPE,
            )
            print(f"  ✅ Loaded: {model_name}")
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"     VRAM used: {vram:.1f} GB")
            return model, tokenizer, model_name
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    raise RuntimeError("All models failed to load!")


def apply_lora(model):
    """Apply LoRA adapter."""
    print("\n  Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✅ LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    
    return model


def load_data(tokenizer):
    """Load training dataset and format for SFT.
    
    Uses tokenizer.apply_chat_template on the 'messages' field to ensure:
    1. Correct special tokens (<bos>, turn markers) are prepended/appended
    2. Format matches the model's pre-training format exactly
    3. Consistent with inference-time tokenization
    
    The pre-built 'text' field is NOT used because it was manually concatenated
    without <bos> and without proper special token handling.
    """
    print(f"\n  Loading data: {TRAIN_FILE}")
    
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_FILE}\n"
            f"Run first: python src/data/03_build_training_data.py"
        )
    
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # Apply max_samples limit if set
    if MAX_SAMPLES is not None:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
        print(f"  ✅ Loaded subset of {len(dataset):,} training examples (max_samples limit)")
    else:
        print(f"  ✅ Loaded {len(dataset):,} training examples")
    
    # Verify data format
    sample = dataset[0]
    if 'messages' not in sample:
        raise ValueError(
            f"Dataset must have a 'messages' field for proper chat template formatting. "
            f"Found keys: {list(sample.keys())}"
        )
    
    print(f"  📝 Format: 'messages' (chat format)")
    
    # Unsloth completion_only_loss=True rules:
    #   - formatting_func + completion_only_loss = INCOMPATIBLE (raises ValueError)
    #   - messages column alone = "must specify formatting_func" 
    #   - prompt/completion as STRINGS = apply_chat_template fails on strings
    #   - prompt/completion as LIST-OF-DICTS = _tokenize_pc calls apply_chat_template
    #
    # Gemma 4 multimodal processor quirk:
    #   apply_chat_template iterates message["content"] looking for image/video dicts.
    #   Plain string content → iterates chars → TypeError: string indices must be integers.
    #   Fix: wrap content in multimodal text format: [{"type": "text", "text": "..."}]
    
    def _to_multimodal_msg(msg):
        """Wrap plain string content into Gemma 4 multimodal format."""
        content = msg.get('content', '')
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"role": msg["role"], "content": content}
    
    def _strip_thinking_tokens(content: str) -> str:
        """Strip <|channel>thought from assistant content.
        
        WHY THIS IS CORRECT:
        Gemma 4 template (thinking OFF) appends <|channel>thought\n<channel|>
        as a pre-closed empty block at add_generation_prompt=True.
        
        During training, completions go through apply_chat_template with
        add_generation_prompt=False, so NO <|channel>thought is added.
        
        If we keep <|channel>thought in our content, the model learns:
          <|turn>model\n<|channel>thought\n{think}\n<channel|>\n{answer}
        
        At inference, template gives:
          <|turn>model\n<|channel>thought\n<channel|>  (pre-closed)
        
        Model expects {think} after <|channel>thought but gets <channel|>
        instead -> falls back to base model format -> structured fields lost.
        
        By stripping, model learns:
          <|turn>model\n{think}\n<channel|>\n{answer}
        
        At inference after the pre-closed <channel|>, model generates the
        structured {answer} it learned -> format compliance 100%.
        
        Thinking trace is empty (cosmetic), but answers are correct.
        """
        content = content.replace("<|channel>thought\n", "", 1)
        content = content.replace("<|channel>thought", "", 1)
        return content
    
    def split_messages(example):
        messages = example['messages']
        prompt_msgs = [_to_multimodal_msg(m) for m in messages if m['role'] != 'assistant']
        
        completion_msgs = []
        for m in messages:
            if m['role'] == 'assistant':
                content = m['content']
                if isinstance(content, str):
                    content = _strip_thinking_tokens(content)
                completion_msgs.append(_to_multimodal_msg(
                    {"role": "assistant", "content": content}
                ))
        return {"prompt": prompt_msgs, "completion": completion_msgs}
    
    dataset = dataset.map(split_messages)
    
    # Remove all columns except prompt/completion
    cols_to_remove = [c for c in dataset.column_names if c not in ('prompt', 'completion')]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    
    print(f"     prompt: {len(dataset[0]['prompt'])} msgs (multimodal format)")
    print(f"     completion: {len(dataset[0]['completion'])} msgs (multimodal format)")
    
    return dataset


def load_eval_data(tokenizer):
    """Load evaluation dataset for validation during training."""
    if not Path(EVAL_FILE).exists():
        print(f"  \u26a0\ufe0f  Eval file not found: {EVAL_FILE} \u2014 training without validation")
        return None

    print(f"\n  Loading eval data: {EVAL_FILE}")
    dataset = load_dataset("json", data_files=EVAL_FILE, split="train")

    def _to_multimodal_msg(msg):
        content = msg.get('content', '')
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"role": msg["role"], "content": content}

    def split_messages(example):
        messages = example['messages']
        prompt_msgs = [_to_multimodal_msg(m) for m in messages if m['role'] != 'assistant']
        # Strip thinking tokens (same as train data) — see _strip_thinking_tokens docstring
        completion_msgs = []
        for m in messages:
            if m['role'] == 'assistant':
                content = m['content']
                if isinstance(content, str):
                    content = content.replace("<|channel>thought\n", "", 1)
                    content = content.replace("<|channel>thought", "", 1)
                completion_msgs.append(_to_multimodal_msg(
                    {"role": "assistant", "content": content}
                ))
        return {"prompt": prompt_msgs, "completion": completion_msgs}

    dataset = dataset.map(split_messages)
    cols_to_remove = [c for c in dataset.column_names if c not in ('prompt', 'completion')]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    print(f"  \u2705 Loaded {len(dataset):,} eval examples")
    return dataset


def train(model, tokenizer, dataset, model_name, eval_dataset=None):
    """Run SFT training with response-only masking.
    
    Uses completion_only_loss=True with prompt/completion as list-of-dicts.
    Unsloth's _tokenize_pc calls apply_chat_template on each internally.
    """
    print(f"\n{'=' * 60}")
    print(f"  Starting SFT Training")
    print(f"  Model: {model_name}")
    print(f"  Data: {len(dataset):,} examples")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Eval: every {EVAL_STEPS} steps")
    print(f"  Response masking: ON (completion-only training)")
    print(f"{'=' * 60}\n")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=True,
            logging_steps=10,
            output_dir=OUTPUT_DIR,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
            completion_only_loss=True,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=EVAL_STEPS if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        ),
    )
    
    # Train
    result = trainer.train()
    
    print(f"\n  ✅ Training complete!")
    print(f"     Total steps: {result.global_step}")
    print(f"     Final loss: {result.training_loss:.4f}")
    
    return result


def save_checkpoint(model, tokenizer, model_name):
    """Save the trained model."""
    print(f"\n  💾 Saving checkpoint to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metadata
    import json
    meta = {
        "base_model": model_name,
        "stage": "SFT",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
    }
    with open(Path(OUTPUT_DIR) / "training_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✅ Checkpoint saved!")
    print(f"\n  Next step:")
    print(f"     python src/training/02_grpo_train.py")


def main():
    print("=" * 60)
    print("  Gemmra — SFT Training (Stage 1)")
    print("=" * 60)
    
    # H4 FIX: --local only overrides model and quantization (must differ on 12GB GPU).
    # All other hyperparameters come from local_sft_config.yaml, making YAML authoritative.
    global PRIMARY_MODEL, LOAD_IN_4BIT, MAX_SAMPLES, NUM_EPOCHS, SAVE_STEPS, EVAL_STEPS, OUTPUT_DIR
    if '--local' in sys.argv:
        print("\n  🖥️  LOCAL MODE: Using local_sft_config.yaml settings")
        PRIMARY_MODEL = "google/gemma-4-12b-it"
        LOAD_IN_4BIT = True   # Local GPU (12 GB) needs 4-bit
    
    if '--smoke' in sys.argv:
        print("\n  🔥 SMOKE MODE: 500 samples, 0.1 epoch, quick validation")
        MAX_SAMPLES = 500
        NUM_EPOCHS = 0.1
        SAVE_STEPS = 50
        EVAL_STEPS = 50
        OUTPUT_DIR = "checkpoints/sft_smoke"
    
    # Load
    model, tokenizer, model_name = load_model()
    model = apply_lora(model)
    dataset = load_data(tokenizer)
    eval_dataset = load_eval_data(tokenizer)
    
    # Train
    result = train(model, tokenizer, dataset, model_name, eval_dataset=eval_dataset)
    
    # Save
    save_checkpoint(model, tokenizer, model_name)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
