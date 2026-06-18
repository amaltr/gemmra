# 🚀 Execution Guide — From Setup to Inference

> Step-by-step instructions for reproducing the complete Gemmra pipeline on AMD MI300X.

---

## ⚡ Quick Overview

```
Phase 1: SETUP              (30 min)  — Environment, GPU verification
Phase 2: DATA PIPELINE      (1-2 hrs) — Download FAERS, OnSIDES, BioDEX → training pairs
Phase 3: SFT TRAINING       (2 hrs)   — LoRA fine-tuning on MI300X (bf16, r=64)
Phase 4: WiSE-FT            (10 min)  — Reasoning recovery via weight interpolation
Phase 5: EVALUATION          (30-90 min) — Gemmra-Bench v1.0 (3,645 samples)
Phase 6: INFERENCE           (on-demand) — Interactive console or demo mode
```

**Optional:** RAFT (rejection sampling) and GRPO (reinforcement learning) — documented but not required for reproducing final results.

---

## 📋 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 95 GB (bf16 LoRA) | 192 GB (AMD MI300X) |
| System RAM | 64 GB | 128 GB |
| Storage | 50 GB (code + data) | 100 GB (with checkpoints) |

> ⚠️ Training requires **95 GB VRAM** at bf16 — impossible on NVIDIA H100 (80 GB). AMD MI300X with 192 GB HBM3 is required for full-precision training without quantization.

### Software Requirements

- Python 3.10+
- PyTorch 2.x with ROCm support
- CUDA/ROCm drivers

---

## 🔧 Phase 1: Setup (30 min)

### Step 1.1: Clone and install

```bash
git clone https://github.com/bhaskarjha-dev/gemmra.git
cd gemmra
chmod +x src/setup/install.sh
bash src/setup/install.sh
```

**Expected output:**
```
==========================================
  Gemmra — Environment Setup
==========================================
[1/7] Setting environment variables...
[4/7] Verifying GPU...
  ✅ GPU: AMD Instinct MI300X
  ✅ VRAM: 206 GB
  ✅ PyTorch 2.x.x+rocmX.X
==========================================
  Setup complete! ✅
==========================================
```

### Step 1.2: Run smoke test

```bash
python src/setup/smoke_test.py
```

**Expected:** All checks pass with ✅ (env, GPU, dependencies, model loading).

---

## 📊 Phase 2: Data Pipeline (1-2 hrs)

### Step 2.1: Download raw data

```bash
# Download FAERS quarterly data (2019Q1–2026Q1, 29 quarters)
python src/data/01_download_faers.py

# Preprocess and join FAERS tables
python src/data/02_preprocess.py

# Download OnSIDES (drug label side effects for T3)
python src/data/04_download_onsides.py

# Download external datasets (BioDEX for T2 MedDRA coding)
python src/data/05_download_external_datasets.py
```

### Step 2.2: Build training data

```bash
# Combinatorial Diversity Engine — generates 32,355 training pairs
python src/data/03_build_training_data.py

# Build evaluation data (3,645 decontaminated samples)
python src/data/06_build_base_eval_data.py
```

**Output files:**
- `data/processed/training_data.jsonl` — 32,355 training pairs
- `data/processed/eval_data.jsonl` — 3,645 eval samples (content-hash decontaminated)

---

## 🧠 Phase 3: SFT Training (~1.9 hrs on MI300X)

### Step 3.1: Run SFT

```bash
python src/training/01_sft_train.py
```

**What this does:**
1. Loads Gemma 4 31B with bf16 LoRA (r=64, α=128, ALL linear layers)
2. Trains on 32,355 examples for 1 epoch
3. Evaluates on held-out examples every 200 steps
4. Auto-selects best checkpoint by eval_loss
5. Saves checkpoint to `checkpoints/sft/`

**Expected metrics:**
- Training loss: ~0.041
- Eval loss: ~0.075
- VRAM usage: ~95 / 192 GB
- Time: ~1.9 hours

### Step 3.2: Verify checkpoint

```bash
ls -la checkpoints/sft/
```

---

## 🔄 Phase 4: WiSE-FT Reasoning Recovery (10 min)

> WiSE-FT (Weight-space Interpolation for Semantic Fine-Tuning) recovers deep clinical reasoning that SFT compressed from 400 words to 45 words.

```bash
python src/training/04_wise_ft.py
```

**What this does:**
- Interpolates: `final_weights = 0.9 × SFT_weights + 0.1 × base_weights`
- Recovers the base model's deep reasoning while retaining SFT's format compliance
- Saves to `checkpoints/sft_wiseft/`

---

## 📈 Phase 5: Evaluation (30-90 min)

### Step 5.1: Quick evaluation (50 samples/task)

```bash
python src/eval/evaluate.py --quick
```

### Step 5.2: Full evaluation (3,645 samples)

```bash
python src/eval/evaluate.py
```

**Expected results:**

| Task | Metric | Score |
|------|--------|:---:|
| T1 Seriousness | F1 | **0.995** |
| T2 MedDRA Coding | Weighted | **0.667** |
| T3 Drug Labelling | F1 | **0.801** |
| T4 Causality | Weighted | **0.986** |
| Format Compliance | — | **100%** |
| **Composite** | — | **0.862** |

### Step 5.3: Base model comparison (optional)

```bash
python src/eval/evaluate_base.py
```

Evaluates the untuned Gemma 4 31B to measure improvement (base composite: 0.729).

---

## 🖥️ Phase 6: Inference

### Interactive console

```bash
python src/inference/run_inference.py
```

Features:
- Interactive menu for all 4 tasks
- Pre-loaded curated cases
- Manual case input
- Performance metrics (latency, throughput, TTFT)

### Demo mode (auto-runs 3 curated cases)

```bash
python src/inference/run_inference.py --demo
```

### Streamlit web demo

```bash
streamlit run src/demo/app.py
```

### Quick Python inference

```python
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.4.2'
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "checkpoints/sft/",
    max_seq_length=4096,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a pharmacovigilance expert. Assess case seriousness per ICH E2A criteria."},
    {"role": "user", "content": "Patient: 68 year-old female on Warfarin. Adverse event: gastrointestinal haemorrhage. Reported outcomes: HO."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False))
```

---

## 🔬 Optional: Advanced Training Methods

### GRPO/DAPO Reinforcement Learning

```bash
python src/training/02_grpo_train.py
```

> **Result:** +0.003 composite improvement, then reward collapse at step ~40. This validates that SFT+WiSE-FT is already at the learnable ceiling for this data. See [GRPO analysis](analysis/grpo_first_principles_analysis.md).

### RAFT Rejection Sampling

```bash
python src/training/03_raft_generate.py        # Generate candidates
python src/training/01_sft_train.py --config configs/raft_sft_config.yaml  # Retrain
```

> **Result:** RAFT validated the SFT ceiling — no significant improvement. See [SFT analysis](analysis/sft_eval_analysis.md).

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `per_device_batch_size` in config |
| `Model not found` | Fallback chain tries Gemma 4 12B automatically |
| `Loss is NaN` | Reduce learning rate or check data format |
| ROCm/GPU issues | Check `docs/research/amd_platform.md` for fixes |
| bitsandbytes crashes | Script auto-detects and works around this |
