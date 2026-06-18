# Source Code — Execution Guide

> Scripts are **numbered in execution order.** Run 01 before 02 before 03.
> SFT + WiSE-FT shipped. GRPO/RAFT explored & documented as negative results.

---

## Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SETUP (run once)                          │
│                                                             │
│  1. src/setup/install.sh        ← Install all dependencies  │
│  2. src/setup/smoke_test.py     ← Verify everything works   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                DATA PIPELINE (CPU — no GPU needed)           │
│                                                             │
│  3. src/data/01_download_faers.py  ← Download FAERS ZIPs    │
│  4. src/data/02_preprocess.py      ← DuckDB: clean & join   │
│  5. src/data/04_download_onsides.py ← OnSIDES for T3         │
│  6. src/data/05_download_external_datasets.py               │
│     └── PHEE, BioDEX, ADE Corpus + drug class map            │
│  7. src/data/03_build_training_data.py ← Create JSONL        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                TRAINING (GPU — AMD MI300X)                    │
│                                                             │
│  8. src/training/01_sft_train.py  ← ✅ SHIPPED (0.862)      │
│     └── bf16 LoRA r=64, α=128, 1 epoch, ~2 hrs              │
│                                                             │
│  [EXPLORED — NOT SHIPPED]                                    │
│  ✗ src/training/02_grpo_train.py  ← GRPO/DAPO (FAILED)      │
│  ✗ src/training/03_raft_generate.py ← RAFT (FAILED)         │
│  ○ src/training/05_showcase_70b.py ← 70B AMD demo           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                EVALUATION (GPU)                              │
│                                                             │
│  9. src/eval/evaluate.py         ← Gemmra-Bench v1.0 (3,645 samples) │
│ 10. src/eval/evaluate_base.py    ← Base model comparison              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                DEMO (GPU)                                    │
│                                                             │
│ 11. src/demo/app.py             ← Streamlit interactive demo │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Folder Guide

### `setup/` — Run First
| Script | GPU? | Time | What It Does |
|--------|------|------|-------------|
| `install.sh` | ❌ | 5 min | Installs Unsloth, TRL v1.0; sets AITER env vars |
| `smoke_test.py` | ✅ | 3 min | Verifies GPU, model loading with fallback |

### `data/` — Data Pipeline (CPU)
| Script | GPU? | Time | What It Does |
|--------|------|------|-------------|
| `01_download_faers.py` | ❌ | 1-2 hrs | Downloads & extracts FAERS quarters from FDA |
| `02_preprocess.py` | ❌ | 15 min | DuckDB: load, dedup, quality filter, master join |
| `04_download_onsides.py` | ❌ | 5 min | Downloads OnSIDES drug label ADE data for T3 |
| `05_download_external_datasets.py` | ❌ | 10 min | Downloads PHEE, BioDEX, ADE Corpus |
| `03_build_training_data.py` | ❌ | 10 min | Creates JSONL with all 4 tasks |

### `training/` — Model Training (GPU)
| Script | GPU? | Time | Status | What It Does |
|--------|------|------|:---:|-------------|
| `01_sft_train.py` | ✅ | ~2 hrs | ✅ SHIPPED | bf16 LoRA SFT on Gemma 4 31B (r=64, α=128) |
| `02_grpo_train.py` | ✅ | 6 hrs | ❌ FAILED | GRPO/DAPO — reward variance collapsed |
| `03_raft_generate.py` | ✅ | ~1 hr | ❌ FAILED | RAFT — binary scores, zero diversity |
| `05_showcase_70b.py` | ✅ | 1.5 hrs | ○ DEMO | 70B full LoRA — AMD exclusive showcase |

### `eval/` — Evaluation (GPU)
| Script | GPU? | Time | What It Does |
|--------|------|------|-------------|
| `evaluate.py` | ✅ | ~87 min | Full eval: F1, MedDRA hierarchical, format compliance |
| `evaluate_base.py` | ✅ | ~60 min | Base model (untuned) evaluation for comparison |

### `demo/` — Interactive Demo
| Script | GPU? | What It Does |
|--------|------|-------------|
| `app.py` | ✅ | Streamlit app with all 4 tasks |

## Key Design Decisions in Code

- **Model:** Gemma 4 31B (`google/gemma-4-31b-it`) — see [ADR-002](../docs/architecture/decisions/ADR-002-base-model.md)
- **LoRA:** r=64, α=128, all linear layers, bf16 (no quantization)
- **Training:** SFT + WiSE-FT shipped. GRPO/DAPO/RAFT all explored, validated ceiling (see [analysis](../docs/evaluation/analysis/))
- **Tokens:** Gemma 4 native `<|channel>thought ... <channel|>` for thinking traces
- **Data quality:** MeditronFO-adopted decontamination, 50:50 T3 balance
- **AMD optimization:** bf16 without quantization (192 GB VRAM)
- All scripts set `HSA_OVERRIDE_GFX_VERSION` and `HF_HUB_DISABLE_XET` automatically

## Final Results — Gemmra-Bench v1.0 (SFT v6, 3,645 samples)

| Task | Score |
|------|:---:|
| T1 Seriousness F1 | **0.995** |
| T2 MedDRA Weighted | **0.667** |
| T3 Labelling F1 | **0.801** |
| T4 Causality Weighted | **0.986** |
| Format Compliance | **100%** |
| **Composite** | **0.862** |
