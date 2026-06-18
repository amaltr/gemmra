"""
Smoke Test — Verify AMD MI300X environment is correctly set up.
Run this BEFORE spending GPU time on actual training.
Expected runtime: < 3 minutes.
"""

import sys
import os

# Ensure critical env vars are set (in case user didn't source ~/.bashrc)
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

def check_env_vars():
    """Check required environment variables."""
    print("=" * 50)
    print("  Step 1: Environment Variables")
    print("=" * 50)
    
    hsa = os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')
    hf = os.environ.get('HF_HUB_DISABLE_XET', '')
    
    if hsa == '9.4.2':
        print(f"  ✅ HSA_OVERRIDE_GFX_VERSION = {hsa}")
    else:
        print(f"  ❌ HSA_OVERRIDE_GFX_VERSION = '{hsa}' (should be '9.4.2')")
        print("     Fix: export HSA_OVERRIDE_GFX_VERSION=9.4.2")
        return False
    
    if hf == '1':
        print(f"  ✅ HF_HUB_DISABLE_XET = {hf}")
    else:
        print(f"  ⚠️  HF_HUB_DISABLE_XET = '{hf}' (recommended: '1')")
    
    return True


def check_gpu():
    """Check GPU is available and is MI300X."""
    print("\n" + "=" * 50)
    print("  Step 2: GPU Detection")
    print("=" * 50)
    
    import torch
    
    if not torch.cuda.is_available():
        print("  ❌ No GPU detected! torch.cuda.is_available() = False")
        print("     Are you running on the AMD cloud notebook?")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"  ✅ GPU: {gpu_name}")
    print(f"  ✅ VRAM: {vram_gb:.0f} GB")
    
    if vram_gb < 100:
        print(f"  ⚠️  VRAM is {vram_gb:.0f}GB — expected 192GB for MI300X")
    
    return True


def check_bitsandbytes():
    """Check bitsandbytes is installed (optional for bf16 LoRA, needed for QLoRA)."""
    print("\n" + "=" * 50)
    print("  Step 3: bitsandbytes (optional — bf16 LoRA doesn't need it)")
    print("=" * 50)
    
    try:
        import bitsandbytes as bnb
        version = bnb.__version__
        print(f"  ✅ bitsandbytes version: {version}")
        
        if 'preview' in version or '1.33' in version or 'dev' in version:
            print("  ✅ Pre-release/dev version detected (correct for AMD ROCm)")
        else:
            print(f"  ⚠️  Version {version} — may not support AMD ROCm correctly")
            print("     Consider reinstalling the preview build")
        
        return True
    except ImportError:
        print("  ℹ️  bitsandbytes not installed (optional for bf16 LoRA)")
        print("     Only needed if you switch to QLoRA 4-bit training.")
        return True  # Not a blocker since we use bf16 LoRA


def check_model_load():
    """Try loading the base model with Unsloth."""
    print("\n" + "=" * 50)
    print("  Step 4: Model Loading (this may take 1-2 min)")
    print("=" * 50)
    
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("     Run: pip install 'unsloth[amd]'")
        return False, None
    
    # Gemma 4 family (same thinking tokens, seamless fallback)
    models_to_try = [
        ("google/gemma-4-31b-it", "Gemma 4 31B (PRIMARY — thinking mode + MMLU-Pro 85.2%)"),
        ("google/gemma-4-12b-it", "Gemma 4 12B (FALLBACK — same family, 2x faster)"),
    ]
    
    for model_name, description in models_to_try:
        print(f"\n  Trying: {description} ...")
        try:
            # Test bf16 loading (matches actual training config — no 4-bit quantization)
            # MI300X has 192 GB VRAM; bf16 31B model uses ~64 GB
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=False,  # bf16 LoRA — no quantization needed on MI300X
                dtype=torch.bfloat16,
            )
            
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  ✅ {description} loaded successfully!")
            print(f"     VRAM used: {vram_used:.1f} GB")
            
            # Quick LoRA test
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print(f"  ✅ LoRA adapter applied successfully!")
            
            # Quick inference test
            # NOTE: Gemma 4 is multimodal — its processor maps positional args
            # to 'images' not 'text', causing NoneType errors (Unsloth #4952).
            # We must use apply_chat_template (same path as actual training).
            FastLanguageModel.for_inference(model)
            messages = [
                {"role": "user", "content": "What is pharmacovigilance?"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"  ✅ Inference works! Response preview: {response[:80]}...")
            
            # Save the winning model name before cleanup
            winning_model = model_name
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            
            return True, winning_model
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            try:
                del model, tokenizer
                torch.cuda.empty_cache()
            except:
                pass
            continue
    
    print("\n  ❌ ALL models failed to load. Check your environment.")
    return False, None


def main():
    print("\n" + "🔬" * 25)
    print("  Gemmra — Smoke Test")
    print("🔬" * 25 + "\n")
    
    results = {}
    
    # Step 1: Environment
    results['env'] = check_env_vars()
    
    # Step 2: GPU
    results['gpu'] = check_gpu()
    
    # Step 3: bitsandbytes
    results['bnb'] = check_bitsandbytes()
    
    # Step 4: Model
    if results['gpu']:
        results['model'], winning_model = check_model_load()
    else:
        print("\n  ⏭️  Skipping model load (GPU check failed)")
        results['model'] = False
        winning_model = None
    
    # Summary
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    
    all_pass = all(results.values())
    
    for check, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check}")
    
    if winning_model:
        print(f"\n  🏆 Working model: {winning_model}")
        print(f"     → Update docs/decisions/ADR-002-base-model.md with this choice")
    
    if all_pass:
        print(f"\n  🎉 ALL CHECKS PASSED — Ready to train!")
        print(f"     Next: python src/training/01_sft_train.py")
        print(f"     (Training data is already in the repo — no data pipeline needed)")
    else:
        print(f"\n  ⚠️  Some checks failed — fix issues before training")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
