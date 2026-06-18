---
library_name: peft
license: apache-2.0
base_model: google/gemma-4-31b-it
tags:
  - pharmacovigilance
  - drug-safety
  - medical
  - peft
  - lora
  - text-generation
  - gemma
  - gemma-4
  - amd
  - mi300x
  - rocm
  - tcs-amd-hackathon
datasets:
  - custom
language:
  - en
pipeline_tag: text-generation
model-index:
  - name: gemmra
    results:
      - task:
          type: text-generation
          name: Pharmacovigilance Assessment
        metrics:
          - type: accuracy
            value: 0.862
            name: Composite Score (Weighted)
          - type: accuracy
            value: 0.995
            name: T1 Seriousness (F1 Score)
          - type: accuracy
            value: 0.667
            name: T2 MedDRA Coding (Weighted)
          - type: accuracy
            value: 0.801
            name: T3 Labelling (F1 Score)
          - type: accuracy
            value: 0.986
            name: T4 Causality (Weighted)
---

# Gemmra — Pharmacovigilance LoRA Adapter for Gemma 4 31B

**Gemmra** is a LoRA adapter that transforms Google's Gemma 4 31B-IT into a specialized pharmacovigilance assessment system. It automates four critical drug safety tasks that typically take 30 minutes per case manually — completing them in under 10 seconds with auditable reasoning traces.

Built for the **TCS & AMD AI Hackathon 2026** on AMD Instinct MI300X (192 GB HBM3).

> ⚠️ **Research Use Only.** This model is for research and educational purposes. It does not provide professional medical or regulatory advice. Do not use for clinical decision-making without expert oversight.

## Key Results

Evaluated on **[Gemmra-Bench v1.0](https://github.com/bhaskarjha-dev/gemmra/blob/main/docs/evaluation/gemmra-bench.md)** — a 3,645-sample decontaminated pharmacovigilance benchmark covering 4 PV tasks with hierarchical scoring.

| Task | Metric | Score | Eval Samples |
|------|--------|:-----:|:------------:|
| T1: Seriousness Classification | F1 Score | **99.5%** | 1,013 |
| T2: MedDRA PT Coding | Weighted (Exact→Synonym→Fuzzy→SOC) | **66.7%** | 845 |
| T3: Drug Labelling Status | F1 Score | **80.1%** | 995 |
| T4: WHO-UMC Causality | Weighted (Exact + Partial) | **98.6%** | 792 |
| **Composite** | **Average (T1+T2+T3+T4)** | **86.2%** | **3,645** |
| Format Compliance | Structured Output Parsing | **100%** | 3,645 |

### Base Model Comparison

Evaluated on the same eval samples (base model used hand-crafted format prompts for fair comparison).

| Metric | Base Gemma 4 31B | Gemmra (SFT) | Δ |
|--------|:---:|:---:|:---:|
| T1 Seriousness (F1) | 97.7% | **99.5%** | +1.8pp |
| T2 MedDRA (Weighted) | 31.1% | **66.7%** | +35.6pp |
| T3 Labelling (F1) | 78.2% | **80.1%** | +1.9pp |
| T4 Causality (Weighted) | 84.5% | **98.6%** | +14.1pp |
| Composite | 72.9% | **86.2%** | +13.3pp |

## Model Details

- **Base Model:** [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it)
- **Method:** LoRA SFT (bf16, r=64) (WiSE-FT weight interpolation explored for reasoning recovery)
- **Training Hardware:** AMD Instinct MI300X (192 GB HBM3)
- **Precision:** bf16 (zero quantization — MI300X VRAM enables full precision)
- **Training Time:** ~1.9 hours
- **VRAM Usage:** 95 GB (training) / 61 GB (inference)

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 64 |
| Alpha (lora_alpha) | 128 |
| Dropout | 0.0 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Task Type | CAUSAL_LM |
| Trainable Parameters | ~0.5% of 31B |

### WiSE-FT (Weight Interpolation Exploration)

While pure SFT (α=1.0) is the primary model deployed due to its superior accuracy across 3 out of 4 tasks and 100% format compliance, we also explored **WiSE-FT** as a research variant to recover reasoning depth. Scaling the LoRA adapter weights by α=0.9 blends SFT format compliance with base model reasoning depth. This recovers the base model's native clinical reasoning (providing 400+ words of structured thinking) at a small cost of ~4% composite accuracy.

```
θ_final = α × θ_SFT + (1 - α) × θ_base (via LoRA adapter weight scaling)
```

## Training Data

| Source | Purpose | Volume |
|--------|---------|--------|
| [FDA FAERS](https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers) | Adverse event case reports (29 quarters, 2019Q1–2026Q1) | 12M+ cases |
| [BioDEX](https://github.com/KarelDO/BioDEX) | Biomedical literature → MedDRA PT mapping | T2 pairs |
| [OnSIDES](https://github.com/tatonetti-lab/onsides) | Drug label side effects → labelling ground truth | T3 pairs |

- **Training pairs:** 32,355 instruction-completion pairs
- **Eval samples:** 3,645 (content-hash decontaminated, MeditronFO-inspired splitting)
- **Diversity:** 93–99% unique completions via Combinatorial Diversity Engine

### Data Challenges Solved
1. **MedDRA is proprietary** — engineered PT training from BioDEX open literature
2. **FDA redacts doctor narratives** — built structured prompts from remaining FAERS fields
3. **BioDEX truncation** — abstracts cut at 500 chars hid ground truth from 92% of T2 data; fixing this single line gave 2.1× improvement
4. **Train/eval leakage** — content-hash splitting ensures zero contamination

## Usage

### Loading the Adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (requires ~62 GB VRAM in bf16)
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31b-it")

# Load Gemmra LoRA adapter
model = PeftModel.from_pretrained(base_model, "team-gemmra/gemmra")
```

### Running Inference

```python
messages = [
    {"role": "system", "content": "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria (Death, Life-threatening, Hospitalization, Disability, Congenital anomaly). Think step by step, then provide your structured assessment."},
    {"role": "user", "content": """Patient: 69-year-old female
Drug: ACTEMRA (tocilizumab)
Adverse events: Cardiac arrest, Pulmonary embolism, Acute kidney injury, Haemodialysis, Platelet count decreased
Outcome: Patient did not survive"""}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

**Expected Output:**
```
SERIOUS: YES
Criteria met: DE (Death), LT (Life-threatening), HO (Hospitalization), DS (Disability)
Rationale: The clinical outcome meets multiple seriousness categories, confirming serious classification.
```

### Using with Unsloth (Faster)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-31b-it",
    max_seq_length=8192,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, "team-gemmra/gemmra")
FastLanguageModel.for_inference(model)
```

## Four Pharmacovigilance Tasks

| Task | Input | Output | Regulatory Framework |
|------|-------|--------|---------------------|
| T1: Seriousness | Patient demographics, AEs, outcomes | SERIOUS: YES/NO + criteria (DE/LT/HO/DS/CA) | ICH E2A |
| T2: MedDRA Coding | Adverse event narrative | MedDRA Preferred Term | MedDRA hierarchy |
| T3: Labelling | Drug name + adverse event | LABELLED: YES/NO + evidence | Drug product labels |
| T4: Causality | Full case context | WHO-UMC category + 6-dim evidence | WHO-UMC criteria |

## Training Pipeline

```
FAERS + BioDEX + OnSIDES
    ↓
Combinatorial Diversity Engine → 32,355 pairs
    ↓
SFT (bf16 LoRA r=64 on MI300X, ~1.9 hrs) → Primary Adapter ✅
    ↓
WiSE-FT exploration (α=0.9) → Explored reasoning variant
    ↓
GRPO validation → +0.003 composite improvement → validated SFT ceiling
    ↓
Evaluation (3,645 decontaminated samples)
    ↓
This Adapter ✅
```

## Hardware Requirements

| Setup | VRAM Required | Notes |
|-------|:---:|-------|
| bf16 inference | ~62 GB | AMD MI300X (192 GB) ✅, 2× A100 80 GB ✅ |
| 4-bit inference | ~18 GB | Single A100/RTX 4090 |
| bf16 training (LoRA r=64) | ~95 GB | AMD MI300X only — impossible on single NVIDIA GPU |

## AMD MI300X Advantage

Training this model at bf16 precision with LoRA r=64 across all 7 linear layer types requires 95 GB VRAM. This is physically impossible on any single NVIDIA GPU (A100/H100 max at 80 GB). AMD MI300X's 192 GB HBM3 is the enabling technology — zero quantization means higher quality gradients and a better final model.

## Limitations

- **MedDRA vocabulary:** Trained on BioDEX-derived PTs (~5,000 terms), not the full proprietary MedDRA dictionary (80,000+ PTs). T2 accuracy will improve with dictionary augmentation.
- **Data source:** FDA FAERS data has known limitations — doctor narratives are redacted, outcome codes can be inconsistent.
- **Not a medical device:** Outputs require expert review before regulatory submission.
- **English only:** Trained exclusively on English-language adverse event reports.

## Citation

```bibtex
@misc{gemmra2026,
  title={Gemmra: Multi-Task Pharmacovigilance Assessment with Fine-Tuned Gemma 4 on AMD MI300X},
  author={Amal T R and Bhaskar Jha},
  year={2026},
  howpublished={TCS \& AMD AI Hackathon 2026},
  url={https://github.com/bhaskarjha-dev/gemmra}
}
```

## Contributors

- **[Amal T R](https://huggingface.co/Amaltrkmr)** — Model training, evaluation, data pipeline, WiSE-FT research
- **[Bhaskar Jha](https://huggingface.co/bhaskarjha-dev)** — Architecture, data engineering, website, presentation, system design

## Links

- 🌐 **Website:** [gemmra.bhaskarjha.dev](https://gemmra.bhaskarjha.dev)
- 💻 **GitHub:** [bhaskarjha-dev/gemmra](https://github.com/bhaskarjha-dev/gemmra) (upstream: [amaltr/gemmra](https://github.com/amaltr/gemmra))
- 🏆 **Hackathon:** TCS & AMD AI Hackathon 2026 — Track: Fine-Tuning (FINETUNING_005)
