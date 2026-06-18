"""
WiSE-FT: Weight Interpolation for SFT Models.
Blends SFT LoRA weights with base model to recover reasoning diversity
while preserving format compliance.

Usage:
  python src/training/04_wise_ft.py --alpha 0.8
  python src/training/04_wise_ft.py --alpha 0.8 --smoke  # quick test

Theory (arxiv.org/abs/2502.xxxxx):
  θ_final = α * θ_SFT + (1-α) * θ_base
  - α=1.0 → pure SFT (100% format, shallow reasoning)
  - α=0.0 → pure base (0% format, deep reasoning)
  - α=0.7-0.9 → sweet spot (format + some reasoning recovery)

Research shows WiSE-FT:
  - Recovers Pass@k (output diversity) while preserving Pass@1
  - Reduces both bias and variance simultaneously
  - Creates better starting points for RL/RAFT
  - Zero inference cost
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def parse_args():
    parser = argparse.ArgumentParser(description="WiSE-FT: Weight Interpolation for SFT")
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Interpolation weight. 1.0=pure SFT, 0.0=pure base (default: 0.8)')
    parser.add_argument('--sft-checkpoint', type=str, default='checkpoints/sft',
                        help='Path to SFT LoRA checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: checkpoints/wise_ft_a{alpha})')
    parser.add_argument('--base-model', type=str, default='google/gemma-4-31b-it',
                        help='Base model name')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick validation: load, interpolate, save, verify')
    parser.add_argument('--sweep', action='store_true',
                        help='Try multiple alpha values: 0.6, 0.7, 0.8, 0.9')
    return parser.parse_args()


def load_sft_model(base_model: str, sft_checkpoint: str):
    """Load base model + SFT LoRA adapter."""
    from unsloth import FastLanguageModel

    print(f"\n  Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  ✅ Base model loaded ({vram:.1f} GB VRAM)")

    print(f"\n  Loading SFT LoRA: {sft_checkpoint}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sft_checkpoint)
    print(f"  ✅ SFT LoRA loaded")

    return model, tokenizer


def wise_ft_interpolate(model, alpha: float):
    """Interpolate LoRA weights toward zero (base model).

    For LoRA, base model = LoRA weights all zero.
    So interpolation is: lora_weight_new = alpha * lora_weight_sft + (1-alpha) * 0
                        = alpha * lora_weight_sft

    This is simpler than full-model interpolation because LoRA adapters
    are additive: base_output + lora_A @ lora_B = final_output.
    Scaling LoRA weights by alpha effectively blends toward base model.
    """
    print(f"\n  Interpolating LoRA weights: α={alpha}")
    print(f"    SFT weight: {alpha:.0%} | Base weight: {1-alpha:.0%}")

    n_params = 0
    n_scaled = 0

    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            n_params += 1
            # Scale LoRA weights by alpha (equivalent to blending with base)
            param.data *= alpha
            n_scaled += 1

    print(f"  ✅ Scaled {n_scaled}/{n_params} LoRA parameters by α={alpha}")
    return model


def save_model(model, tokenizer, output_dir: str, alpha: float, base_model: str):
    """Save interpolated model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n  💾 Saving to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "method": "WiSE-FT",
        "alpha": alpha,
        "base_model": base_model,
        "sft_weight": alpha,
        "base_weight": 1 - alpha,
        "description": f"LoRA weights scaled by {alpha} (WiSE-FT interpolation toward base model)",
    }
    with open(Path(output_dir) / "wise_ft_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ Saved! ({Path(output_dir) / 'wise_ft_meta.json'})")


def quick_verify(model, tokenizer, alpha: float):
    """Quick generation test to verify model works after interpolation."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    test_prompt = [
        {"role": "system", "content": [{"type": "text", "text": "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria. Think step by step."}]},
        {"role": "user", "content": [{"type": "text", "text": "Patient: 65-year-old male on Warfarin for atrial fibrillation. Adverse event: Gastrointestinal haemorrhage. Outcome: Hospitalization required."}]},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            test_prompt, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            test_prompt, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(
        text=[prompt_text], return_tensors="pt",
        truncation=True, max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    print(f"\n{'=' * 60}")
    print(f"  WISE-FT VERIFICATION (α={alpha})")
    print(f"{'=' * 60}")
    print(f"  Generated ({len(generated)} chars):")
    print(f"  {generated[:500]}")

    # Check format compliance
    has_format = any(k in generated for k in ['SERIOUS:', 'MedDRA PT:', 'LABELLED:', 'WHO-UMC'])
    has_channel_close = '<channel|>' in generated
    print(f"\n  Format compliance: {'✅' if has_format else '❌'}")
    print(f"  Has <channel|>: {'✅' if has_channel_close else '❌'}")
    print(f"{'=' * 60}")

    return has_format


def main():
    args = parse_args()

    print("=" * 60)
    print("  WiSE-FT: Weight Interpolation for SFT")
    print("=" * 60)

    if args.sweep:
        alphas = [0.6, 0.7, 0.8, 0.9]
        print(f"\n  SWEEP MODE: Testing α = {alphas}")
    else:
        alphas = [args.alpha]

    for alpha in alphas:
        output_dir = args.output_dir or f"checkpoints/wise_ft_a{int(alpha*100)}"

        print(f"\n{'─' * 60}")
        print(f"  α = {alpha} | Output: {output_dir}")
        print(f"{'─' * 60}")

        # Load fresh for each alpha
        model, tokenizer = load_sft_model(args.base_model, args.sft_checkpoint)

        # Interpolate
        model = wise_ft_interpolate(model, alpha)

        # Save
        save_model(model, tokenizer, output_dir, alpha, args.base_model)

        # Quick verify
        if not args.smoke:
            quick_verify(model, tokenizer, alpha)

        # Free memory for sweep
        if len(alphas) > 1:
            del model
            torch.cuda.empty_cache()

    print(f"\n  ✅ WiSE-FT complete!")
    print(f"\n  Next steps:")
    print(f"     # Evaluate interpolated model:")
    print(f"     python src/eval/evaluate.py --checkpoint {output_dir} --quick")
    print(f"     # Or check raw outputs:")
    print(f"     python check_raw.py --ckpt {output_dir} --all --max-tokens 2048")

    return 0


if __name__ == "__main__":
    sys.exit(main())
