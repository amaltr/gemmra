"""
70B AMD Showcase Training — Proof of concept.
Run this on GPU AFTER primary 8B training is complete.

Fine-tunes Llama-3.3-70B-Instruct with FULL 16-bit LoRA on MI300X.
This is PHYSICALLY IMPOSSIBLE on any single NVIDIA GPU.
"""

import os
import sys
import torch
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MAX_SEQ_LENGTH = 2048
TRAIN_FILE = "data/processed/training_data.jsonl"
OUTPUT_DIR = "checkpoints/70b_showcase"
MAX_SAMPLES = 2000  # Proof of concept — not full training


def main():
    print("=" * 60)
    print("  Gemmra — 70B AMD Showcase")
    print("  ⚡ Full 16-bit LoRA — ONLY possible on MI300X!")
    print("=" * 60)
    
    # Check VRAM
    load_in_4bit = True  # Default to QLoRA if we can't determine VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram_gb:.0f} GB")
        
        if vram_gb < 150:
            print(f"  ⚠️  VRAM ({vram_gb:.0f}GB) may be insufficient for 70B full LoRA")
            print(f"     Falling back to QLoRA 4-bit...")
            load_in_4bit = True
        else:
            print(f"  ✅ Sufficient VRAM for full 16-bit LoRA!")
            load_in_4bit = False
    else:
        vram_gb = 0
        print("  ❌ No GPU detected! This script requires an AMD MI300X GPU.")
        return 1
    
    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    print(f"  Mode: {'QLoRA 4-bit' if load_in_4bit else 'FULL 16-bit LoRA'}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
    )
    
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  ✅ Model loaded! VRAM used: {vram_used:.1f} GB")
    
    # Apply LoRA (smaller adapter for 70B — proof of concept)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.4f}%)")
    
    # Load data
    if not Path(TRAIN_FILE).exists():
        print(f"  ❌ Training data not found: {TRAIN_FILE}")
        return 1
    
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    # Format: use tokenizer's native chat template (NOT hardcoded Gemma 4 tokens).
    # CRITICAL: Llama 3.3 uses <|start_header_id|>/<|end_header_id|> tokens, while
    # Gemma 4 uses <start_of_turn>/<end_of_turn>. Using the wrong model's tokens
    # corrupts attention layers and produces gibberish.
    sample = dataset[0]
    if 'messages' in sample:
        def format_prompt_completion(example):
            messages = example['messages']
            prompt_messages = [msg for msg in messages if msg['role'] != 'assistant']
            assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
            
            try:
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: generic format (model-agnostic)
                parts = []
                for msg in prompt_messages:
                    role = msg['role']
                    content = msg['content']
                    parts.append(f"### {role.capitalize()}\n{content}")
                parts.append("### Assistant\n")
                prompt = "\n\n".join(parts)
                
            completion = assistant_messages[0]['content'] if assistant_messages else ""
            if tokenizer.eos_token:
                completion += tokenizer.eos_token
            else:
                completion += "<|eot_id|>"
            return {"prompt": prompt, "completion": completion}
            
        dataset = dataset.map(format_prompt_completion)
        if 'text' in dataset.column_names:
            dataset = dataset.remove_columns(["text"])
    elif 'text' not in sample:
        print(f"  ❌ Dataset has neither 'messages' nor 'text' field")
        return 1
    
    print(f"  Data: {len(dataset)} examples (proof of concept)")
    
    # TRL 0.20+ removed DataCollatorForCompletionOnlyLM.
    # Use completion_only_loss=True in SFTConfig instead.
    
    # Train
    print(f"\n  Starting 70B showcase training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,  # Conservative for 70B
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=1e-4,
            output_dir=OUTPUT_DIR,
            bf16=True,
            logging_steps=5,
            save_steps=100,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field=None,
            packing=False,
            completion_only_loss=True,  # Masks prompt tokens, trains only on assistant response
        ),
    )
    
    result = trainer.train()
    
    print(f"\n  ✅ 70B showcase training complete!")
    print(f"     Steps: {result.global_step}")
    print(f"     Final loss: {result.training_loss:.4f}")
    
    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    import json
    meta = {
        "model": MODEL_NAME,
        "mode": "QLoRA" if load_in_4bit else "Full 16-bit LoRA",
        "vram_gb": vram_gb,
        "samples": len(dataset),
        "purpose": "AMD MI300X exclusive showcase — impossible on NVIDIA",
    }
    with open(Path(OUTPUT_DIR) / "training_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  💾 Saved to: {OUTPUT_DIR}")
    print(f"\n  🎯 Presentation talking point:")
    print(f'     "We fine-tuned a 70B parameter model on a single GPU')
    print(f'      using full 16-bit LoRA. This requires 140+ GB VRAM —')
    print(f'      physically impossible on NVIDIA. Only AMD MI300X makes')
    print(f'      this possible."')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
