<p align="center">
  <img src="gemmra_logo.png" alt="Gemmra" width="180"/>
</p>

<h1 align="center">Gemmra</h1>

<p align="center">
  <em>Multi-task pharmacovigilance AI with auditable clinical reasoning</em>
</p>

<p align="center">
  <a href="https://gemmra.bhaskarjha.dev"><strong>Website</strong></a> ·
  <a href="https://huggingface.co/amaltrkmr/gemmra"><strong>Model</strong></a> ·
  <a href="https://huggingface.co/amaltrkmr/gemmra-GGUF"><strong>GGUF</strong></a> ·
  <a href="docs/getting-started/execution-guide.md"><strong>Docs</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Gemma_4_31B-4285F4?logo=google&logoColor=white" alt="Gemma 4"/>
  <img src="https://img.shields.io/badge/GPU-AMD_MI300X-ED1C24?logo=amd&logoColor=white" alt="AMD"/>
  <img src="https://img.shields.io/badge/Composite-86.2%25-brightgreen" alt="Score"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/Format_Compliance-100%25-brightgreen" alt="Format"/>
</p>

---

Gemmra is a fine-tuned **Gemma 4 31B** that automates four critical drug safety review tasks — seriousness classification, MedDRA coding, drug labelling verification, and causality assessment — with auditable chain-of-thought reasoning. Trained on AMD Instinct MI300X using bf16 LoRA, it achieves **86.2% composite** across 3,645 decontaminated evaluation samples with **100% format compliance**.

> *In Ayurveda, **Nidāna** (निदान) is the systematic investigation of disease causation — exactly what Gemmra does for drug safety.*

## Results

Evaluated on [**Gemmra-Bench v1.0**](docs/evaluation/gemmra-bench.md) — 3,645 decontaminated pharmacovigilance samples with hierarchical scoring.

| Task | What It Does | Score | Δ vs Base |
|------|-------------|:---:|:---:|
| **T1** Seriousness | Is this adverse event serious per ICH E2A? | **0.995** F1 | +1.8% |
| **T2** MedDRA Coding | Map adverse event to standard medical terminology | **0.667** weighted | **+114%** |
| **T3** Labelling | Is this side effect on the drug's label? | **0.801** F1 | +2.4% |
| **T4** Causality | Did the drug cause this event? (WHO-UMC scale) | **0.986** weighted | +16.7% |
| **Composite** | | **0.862** | **+18.2%** |

Format compliance: **100%** across all 3,645 samples. Every response includes visible thinking traces.

## Quick Start

```bash
# Clone
git clone https://github.com/bhaskarjha-dev/gemmra.git && cd gemmra

# Setup environment (AMD MI300X)
bash src/setup/install.sh

# Build training data from public sources
python src/data/01_download_faers.py          # Download FDA FAERS
python src/data/02_preprocess.py              # Clean & join with DuckDB
python src/data/04_download_onsides.py        # Drug label side effects
python src/data/03_build_training_data.py     # Generate 32K training pairs

# Train (~1.9 hrs on MI300X)
python src/training/01_sft_train.py

# Evaluate
python src/eval/evaluate.py --quick           # Sanity check (200 samples)
python src/eval/evaluate.py                   # Full benchmark (3,645 samples)

# Run inference
python src/inference/run_inference.py --demo   # Demo with curated cases
```

<details>
<summary><strong>Run locally with Ollama</strong></summary>

```bash
ollama run gemmra
```

Or use the quantized GGUF directly:

```bash
# Download from HuggingFace
huggingface-cli download team-gemmra/gemmra-GGUF --local-dir ./models

# Run with llama.cpp
./llama-server -m models/gemmra-Q4_K_M.gguf -c 4096
```

</details>

<details>
<summary><strong>Run with Python (Unsloth)</strong></summary>

```python
from unsloth import FastLanguageModel
import torch

# Downloads the LoRA adapter automatically from HuggingFace on first run
model, tokenizer = FastLanguageModel.from_pretrained(
    "amaltrkmr/gemmra", max_seq_length=4096,
    load_in_4bit=False, dtype=torch.bfloat16,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a pharmacovigilance expert. Assess case seriousness per ICH E2A."},
    {"role": "user", "content": "Patient: 68F on Warfarin. AE: GI haemorrhage. Outcome: HO."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False))
```

</details>

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE                                                  │
│  FAERS (FDA) + OnSIDES (drug labels) + BioDEX (literature)      │
│       ↓                                                         │
│  Combinatorial Diversity Engine → 32,355 training pairs          │
│  (93-99% unique completions per task)                            │
├─────────────────────────────────────────────────────────────────┤
│  TRAINING (AMD MI300X — 192 GB HBM3)                             │
│                                                                  │
│  Gemma 4 31B  →  bf16 LoRA SFT (r=64, ~1.9 hrs)                 │
│                      ↓                                           │
│                  WiSE-FT (reasoning recovery)                    │
│                      ↓                                           │
│                  GRPO validation → confirmed SFT ceiling          │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUT                                                          │
│                                                                  │
│  ⟨think⟩ step-by-step clinical reasoning ⟨/think⟩               │
│  SERIOUS: YES                                                    │
│  Criteria: HO (Hospitalization)                                  │
│  Rationale: Patient admitted for GI haemorrhage on Warfarin...   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovations

| # | Innovation | Impact |
|---|-----------|--------|
| 1 | **Data-first debugging** | Fixing one truncation line → 2.1× T2 improvement |
| 2 | **Auditable thinking traces** | Gemma 4 native `⟨channel⟩thought` for regulatory compliance |
| 3 | **WiSE-FT reasoning recovery** | Weight interpolation recovers 400-word reasoning SFT compressed to 45 |
| 4 | **Gemmra-Bench v1.0** | 3,645-sample decontaminated PV evaluation benchmark |
| 5 | **Hierarchical MedDRA scoring** | 4-level evaluation: exact → synonym → fuzzy → SOC |
| 6 | **Published negative results** | GRPO/DAPO/RAFT failures proving SFT+WiSE-FT ceiling |
| 7 | **AMD MI300X advantage** | bf16 LoRA r=64 needs 95 GB — impossible on 80 GB H100 |

## Model Availability

| Format | Link | Size | Use Case |
|--------|------|:----:|----------|
| LoRA Adapter | [amaltrkmr/gemmra](https://huggingface.co/amaltrkmr/gemmra) | ~500 MB | Apply to base Gemma 4 31B |
| GGUF Q4_K_M | [team-gemmra/gemmra-GGUF](https://huggingface.co/team-gemmra/gemmra-GGUF) | ~18 GB | Local inference (Ollama / llama.cpp) |
| GGUF Q8_0 | [team-gemmra/gemmra-GGUF](https://huggingface.co/team-gemmra/gemmra-GGUF) | ~33 GB | High-quality local inference |

## Repository Structure

```
gemmra/
│
├── src/                              # All source code
│   ├── data/                         #   FAERS download, preprocessing, training data
│   ├── training/                     #   SFT, GRPO, RAFT, WiSE-FT scripts
│   ├── eval/                         #   Gemmra-Bench evaluation framework
│   ├── inference/                    #   Interactive & demo inference
│   ├── demo/                         #   Streamlit web demo
│   ├── deploy/                       #   HuggingFace upload
│   ├── setup/                        #   Environment setup (install.sh, smoke_test.py)
│   └── utils/                        #   MedDRA SOC mapping utilities
│
├── configs/                          # Training hyperparameters (YAML)
│   ├── sft_config.yaml               #   Primary SFT configuration
│   └── local_sft_config.yaml         #   Consumer hardware (12GB VRAM)
│
├── docs/                             # Documentation (organized by topic)
│   ├── getting-started/              #   Setup & reproduction guides
│   ├── architecture/                 #   Problem definition, design decisions (ADRs)
│   ├── evaluation/                   #   Benchmarks, score history, error analysis
│   ├── research/                     #   39 verified discoveries, model comparisons
│   ├── domain/                       #   Pharmacovigilance background
│   ├── business/                     #   ROI, AMD hardware advantages
│   └── diagrams/                     #   Architecture diagrams (Mermaid + SVG)
│
├── data/                             # Data directory (populated by scripts)
├── MODEL_CARD.md                     # HuggingFace model card
├── LICENSE                           # Apache 2.0
└── requirements.txt                  # Python dependencies
```

## Documentation

| Section | Contents |
|---------|----------|
| [**Getting Started**](docs/getting-started/) | [Execution Guide](docs/getting-started/execution-guide.md) · [Local Training](docs/getting-started/local-training.md) |
| [**Architecture**](docs/architecture/) | [Problem Statement](docs/architecture/problem-statement.md) · [Research Pipeline](docs/architecture/research-pipeline.md) · [Innovations](docs/architecture/innovations.md) · [ADRs](docs/architecture/decisions/) |
| [**Evaluation**](docs/evaluation/) | [Gemmra-Bench](docs/evaluation/gemmra-bench.md) · [Score Progression](docs/evaluation/score-progression.md) · [Error Analysis](docs/evaluation/error-analysis.md) · [Deep Analyses](docs/evaluation/analysis/) |
| [**Research**](docs/research/) | [Discoveries Log](docs/research/discoveries_log.md) · [Model Comparison](docs/research/model_comparison.md) · [MeditronFO Analysis](docs/research/meditron_fo_reference.md) |
| [**Domain**](docs/domain/) | [Pharmacovigilance 101](docs/domain/pharmacovigilance_101.md) · [FAERS Schema](docs/domain/faers_schema.md) · [Worked Examples](docs/domain/worked_examples.md) |

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Base Model | Gemma 4 31B | MMLU-Pro 85.2%, native thinking mode, Apache 2.0 |
| Training | bf16 LoRA (r=64, α=128) | Full precision — zero quantization on 192 GB HBM |
| Reasoning | WiSE-FT (α=0.9) | Recovers deep clinical reasoning SFT compressed |
| Framework | Unsloth + TRL | Fast LoRA training with native AMD ROCm support |
| Data | FAERS + OnSIDES + BioDEX | 32,355 pairs from public pharmacovigilance sources |
| GPU | AMD MI300X | 192 GB HBM3, 5.3 TB/s — enables bf16 training impossible on H100 |

## Team

<table>
  <tr>
    <td align="center"><strong>Amal T R</strong><br/>Training · Inference · Domain Research · Deployment<br/><a href="https://github.com/amaltrkmr">GitHub</a> · <a href="https://huggingface.co/amaltrkmr">HuggingFace</a></td>
    <td align="center"><strong>Bhaskar Jha</strong><br/>Architecture · Data · Evaluation · Website<br/><a href="https://github.com/bhaskarjha-dev">GitHub</a> · <a href="https://huggingface.co/bhaskarjha-dev">HuggingFace</a></td>  
  </tr>
</table>

Built for the **TCS & AMD AI Hackathon 2026** — Track: Fine-Tuning (FINETUNING_005)

## Citation

```bibtex
@misc{gemmra2026,
  title   = {Gemmra: Multi-Task Pharmacovigilance Assessment with Fine-Tuned Gemma 4 on AMD MI300X},
  author  = {Amal T R and Bhaskar Jha},
  year    = {2026},
  url     = {https://github.com/bhaskarjha-dev/gemmra}
}
```

## License

[Apache 2.0](LICENSE) — see [MODEL_CARD.md](MODEL_CARD.md) for model-specific details.

---

<p align="center">
  <a href="https://gemmra.bhaskarjha.dev">Website</a> ·
  <a href="https://huggingface.co/amaltrkmr/gemmra">HuggingFace</a> ·
  <a href="docs/getting-started/execution-guide.md">Get Started</a>
</p>
