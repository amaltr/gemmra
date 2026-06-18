# 🖥️ Local Training Guide — Fine-Tuning on Consumer Hardware

> **Can you fine-tune locally?** YES — with Gemma 4 12B (not 31B).
> This is useful for debugging, testing the pipeline, and iterating fast.
> The full model trains on AMD MI300X with Gemma 4 31B.

---

## What Fits vs What Doesn't

| Model | 4-bit Size | Training VRAM | Fits in 12GB? |
|-------|-----------|---------------|---------------|
| Gemma 4 31B | ~18-20 GB | ~24-28 GB | ❌ NO |
| **Gemma 4 12B** | **~7-8 GB** | **~9-11 GB** | **✅ YES (tight)** |
| Llama 3.1 8B | ~5 GB | ~7-9 GB | ✅ YES |

**Bottom line:** Gemma 4 12B in 4-bit QLoRA with batch_size=1 fits in 12GB VRAM.

---

## Why Train Locally?

| Benefit | Explanation |
|---------|-------------|
| **No GPU time limit** | Cloud GPUs have quotas — your laptop has unlimited time |
| **Faster iteration** | Test pipeline changes instantly, don't waste cloud hours |
| **Debug data issues** | Verify JSONL format, chat template, thinking tokens |
| **Verify eval works** | Test evaluation script before running on cloud |
| **Demo development** | Test Streamlit app with a real model |

---

## Setup (One-Time, ~15 minutes)

### Step 1: Install Python dependencies

```bash
cd gemmra

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows

# Install PyTorch with CUDA (for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training stack
pip install bitsandbytes
pip install "unsloth[colab-new]"
pip install git+https://github.com/huggingface/trl.git
pip install datasets transformers accelerate peft

# Install data pipeline dependencies
pip install duckdb pandas numpy requests lxml

# Install demo dependencies
pip install streamlit plotly
```

### Step 2: Verify CUDA works

```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"
```

### Step 3: Verify Unsloth

```python
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

---

## Running the Data Pipeline (CPU Only)

```bash
# Step 1: Download FAERS data (1-2 hrs, internet dependent)
python src/data/01_download_faers.py

# Step 2: Preprocess with DuckDB (~15 min)
python src/data/02_preprocess.py

# Step 3: Download OnSIDES database for T3 ground truth (~5 min)
python src/data/04_download_onsides.py

# Step 4: Build training JSONL (~10 min)
python src/data/03_build_training_data.py
```

The output files in `data/processed/` are the same format used by both local
and cloud training scripts. **Build once, use everywhere.**

---

## Running SFT Training (Local — Gemma 4 12B)

### Option A: Use the local flag

```bash
python src/training/01_sft_train.py --local
```

The script detects `--local` flag and uses reduced settings automatically
(12B model, 4-bit quantization, r=16, batch_size=1).

### Option B: Use the local config

```bash
python src/training/01_sft_train.py --config configs/local_sft_config.yaml
```

### What to expect locally

| Metric | Cloud (MI300X + 31B) | Local (12GB VRAM + 12B) |
|--------|---------------------|----------------------|
| VRAM usage | ~95 GB (bf16) | ~9-11 GB (4-bit) |
| Training speed | ~800 samples/hr | ~200 samples/hr |
| Time for 5000 samples | ~6 hrs | ~25 hrs |
| Time for 500 samples (test) | ~40 min | ~2.5 hrs |

> 💡 **Recommendation:** Train on ~500 samples locally just to verify the pipeline
> works end-to-end. Use the full dataset on AMD MI300X for the production model.

---

## Compatibility Notes

### The scripts are already compatible

1. **Fallback chain:** `01_sft_train.py` tries Gemma 4 31B first, then
   falls back to 12B. On a consumer GPU, it will automatically load 12B.

2. **Same data format:** Both models use the same JSONL chat format.
   `data/processed/training_data.jsonl` works with both.

3. **Same eval script:** `src/eval/evaluate.py` works with any checkpoint.

4. **HSA env var is harmless:** `os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')`
   only matters on AMD GPUs. On NVIDIA, it's ignored completely.

### What IS different

| Aspect | AMD Cloud (MI300X) | Local (NVIDIA) |
|--------|-------|-------|
| `HSA_OVERRIDE_GFX_VERSION` | Required | Ignored (harmless) |
| bitsandbytes version | ROCm pre-release | Standard pip version |
| AITER env vars | Set for inference | Ignored (NVIDIA) |
| Precision | bf16 (no quantization) | 4-bit QLoRA |

---

## Local Config File

A local-optimized config is at `configs/local_sft_config.yaml`:

```yaml
model:
  name: "google/gemma-4-12b-it"       # Fits in 12GB VRAM
  max_seq_length: 2048                 # Shorter for memory savings
  load_in_4bit: true
  dtype: "bfloat16"

lora:
  r: 16                                # Reduced from 64 for memory
  alpha: 32
  dropout: 0
  bias: "none"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  gradient_checkpointing: "unsloth"

training:
  per_device_batch_size: 1             # Minimum for 12GB VRAM
  gradient_accumulation_steps: 8       # Effective batch = 8
  learning_rate: 2.0e-4
  num_epochs: 1                        # Quick iteration
  warmup_ratio: 0.05
  max_samples: 2000                    # Subset for speed
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `MAX_SEQ_LENGTH` to 1024, or `BATCH_SIZE` to 1 |
| `bitsandbytes error` | `pip install --force-reinstall bitsandbytes` |
| `Unsloth import error` | Try `pip install unsloth` without extras |
| `Model download slow` | Set `HF_HUB_DISABLE_XET=1` env var |
| `torch.cuda not available` | Verify CUDA toolkit matches PyTorch version |
| `Training very slow` | Normal — 12B on 12GB VRAM is memory-bound. Use fewer samples |
