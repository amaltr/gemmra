"""
HuggingFace Model Upload Script for Gemmra.

Run this on the AMD MI300X cloud machine where checkpoints exist.

Prerequisites:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login   # Use Amal's write token

Usage:
  python src/deploy/upload_to_hf.py                          # Upload SFT (primary model)
  python src/deploy/upload_to_hf.py --checkpoint checkpoints/wise_ft_a90 # Upload WiSE-FT checkpoint
  python src/deploy/upload_to_hf.py --repo-id team-gemmra/gemmra  # Custom repo
  python src/deploy/upload_to_hf.py --dry-run                # Preview without uploading
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Gemmra checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/sft",
        help="Path to checkpoint directory (default: checkpoints/sft)"
    )
    parser.add_argument(
        "--repo-id", type=str, default="team-gemmra/gemmra",
        help="HuggingFace repo ID (default: team-gemmra/gemmra)"
    )
    parser.add_argument(
        "--model-card", type=str, default="hackathon_final/huggingface_model_card.md",
        help="Path to model card README.md to include"
    )
    parser.add_argument(
        "--commit-message", type=str,
        default="Initial release: Gemmra SFT LoRA adapter (bf16, r=64) for pharmacovigilance",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--create-repo", action="store_true",
        help="Create the HuggingFace repo if it doesn't exist"
    )
    return parser.parse_args()


def validate_checkpoint(checkpoint_path: str) -> dict:
    """Validate that the checkpoint directory has the required files."""
    ckpt = Path(checkpoint_path)
    
    if not ckpt.exists():
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        print(f"     Available checkpoints:")
        for p in Path("checkpoints").glob("*/adapter_config.json"):
            print(f"       {p.parent}")
        sys.exit(1)
    
    # Required files
    required = ["adapter_config.json"]
    adapter_files = list(ckpt.glob("adapter_model*"))  # .safetensors or .bin
    
    missing = []
    for f in required:
        if not (ckpt / f).exists():
            missing.append(f)
    
    if not adapter_files:
        missing.append("adapter_model.safetensors (or .bin)")
    
    if missing:
        print(f"  ❌ Checkpoint is incomplete. Missing files:")
        for f in missing:
            print(f"       {f}")
        sys.exit(1)
    
    # Parse adapter config
    with open(ckpt / "adapter_config.json") as f:
        config = json.load(f)
    
    # Fix base_model_name_or_path if it points to Unsloth wrapper
    base_model = config.get("base_model_name_or_path", "")
    if "unsloth/" in base_model:
        canonical = base_model.replace("unsloth/", "google/").lower()
        print(f"  ⚠️  base_model_name_or_path uses Unsloth wrapper: {base_model}")
        print(f"     Updating to canonical: {canonical}")
        config["base_model_name_or_path"] = canonical
        with open(ckpt / "adapter_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  ✅ adapter_config.json updated!")
    
    # List all files
    files = list(ckpt.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    
    info = {
        "path": str(ckpt),
        "files": [str(f.relative_to(ckpt)) for f in files],
        "total_size_mb": total_size / 1e6,
        "base_model": config.get("base_model_name_or_path", "unknown"),
        "lora_r": config.get("r", "unknown"),
        "lora_alpha": config.get("lora_alpha", "unknown"),
        "target_modules": config.get("target_modules", []),
    }
    
    return info


def copy_model_card(checkpoint_path: str, model_card_path: str):
    """Copy model card README.md into checkpoint directory."""
    src = Path(model_card_path)
    dst = Path(checkpoint_path) / "README.md"
    
    if not src.exists():
        print(f"  ⚠️  Model card not found: {model_card_path}")
        print(f"     Upload will proceed without README.md")
        return False
    
    shutil.copy2(src, dst)
    print(f"  ✅ Model card copied: {src} → {dst}")
    return True


def upload(repo_id: str, checkpoint_path: str, commit_message: str, create_repo: bool):
    """Upload checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo as hf_create_repo
    
    api = HfApi()
    
    # Create repo if requested
    if create_repo:
        try:
            hf_create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"  ✅ Repo created/verified: {repo_id}")
        except Exception as e:
            print(f"  ⚠️  Repo creation: {e}")
    
    # Upload
    print(f"\n  📤 Uploading to: https://huggingface.co/{repo_id}")
    print(f"     From: {checkpoint_path}")
    print(f"     Commit: {commit_message}")
    print(f"     This may take a few minutes...\n")
    
    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    
    print(f"\n  ✅ Upload complete!")
    print(f"     View: https://huggingface.co/{repo_id}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Gemmra — HuggingFace Model Upload")
    print("=" * 60)
    
    # Validate checkpoint
    print(f"\n  📂 Validating checkpoint: {args.checkpoint}")
    info = validate_checkpoint(args.checkpoint)
    
    print(f"  ✅ Checkpoint valid!")
    print(f"     Base model: {info['base_model']}")
    print(f"     LoRA: r={info['lora_r']}, α={info['lora_alpha']}")
    print(f"     Target modules: {', '.join(info['target_modules'])}")
    print(f"     Total size: {info['total_size_mb']:.1f} MB")
    print(f"     Files ({len(info['files'])}):")
    for f in info['files']:
        print(f"       {f}")
    
    # Copy model card
    print(f"\n  📝 Model card: {args.model_card}")
    copy_model_card(args.checkpoint, args.model_card)
    
    # Dry run check
    if args.dry_run:
        print(f"\n  🏁 DRY RUN — would upload to: {args.repo_id}")
        print(f"     Commit: {args.commit_message}")
        print(f"     No files were uploaded.")
        return 0
    
    # Upload
    try:
        upload(args.repo_id, args.checkpoint, args.commit_message, args.create_repo)
    except Exception as e:
        print(f"\n  ❌ Upload failed: {e}")
        print(f"\n  Troubleshooting:")
        print(f"     1. Run: huggingface-cli login")
        print(f"     2. Ensure token has Write permission")
        print(f"     3. Ensure repo exists: huggingface-cli repo create {args.repo_id.split('/')[-1]} --type model")
        return 1
    
    # Post-upload instructions
    print(f"\n{'=' * 60}")
    print(f"  Next Steps:")
    print(f"  1. Verify: https://huggingface.co/{args.repo_id}")
    print(f"  2. Add collaborator: Repo Settings → Collaborators → Add 'bhaskarjha-dev'")
    print(f"  3. Test loading:")
    print(f"     from peft import PeftModel")
    print(f"     model = PeftModel.from_pretrained(base, '{args.repo_id}')")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
