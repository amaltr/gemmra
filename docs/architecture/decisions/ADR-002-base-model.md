# ADR-002: Base Model Selection

**Status:** ✅ DECIDED — June 9, 2026 (v3 — CORRECTED from false premise)  
**Date:** June 9, 2026  
**Decision Makers:** Team  
**Supersedes:** v2 (Qwen3-32B — based on false claim Gemma 4 lacks thinking mode)

---

## Critical Correction

v2 selected Qwen3-32B based on "native thinking mode" being unique to Qwen.
**This was FALSE.** Gemma 4 has built-in thinking mode (`enable_thinking=True`)
across all variants. With thinking mode being a TIE, Gemma 4 31B's superior
benchmarks (MMLU-Pro 85.2% vs 79%) make it the clear winner.

## Decision

### Primary: Gemma 4 31B (`google/gemma-4-31b-it`)

A dense 31B model with:
- **MMLU-Pro 85.2%** (highest in its weight class)
- **Built-in thinking mode** (`enable_thinking=True` / `<|channel>thought`)
- **Multimodal** (text, image, video, audio)
- Released April 2, 2026 — 2+ months of validation
- Apache 2.0 license, Unsloth + AMD MI300X confirmed
- MeditronFO-inspired pipeline validates medical SFT on Gemma architecture

### Why Gemma 4 31B?

1. **Best raw reasoning** — MMLU-Pro 85.2%, GPQA Diamond 84.3%, AIME 89.2%
2. **Has thinking mode** — `enable_thinking=True` for T2+T4 reasoning
3. **MeditronFO validation** — Gemma-3-27B-MeditronFO beat MedGemma; Gemma 4 31B is stronger
4. **Innovation story** — "GemmraFO" on latest Gemma 4 architecture
5. **Same-family fallback** — 12B uses identical format, zero code changes
6. **Dense architecture** — all 31B params active during training

### Fallback Chain

| Priority | Model | Why |
|----------|-------|-----|
| **PRIMARY** | `google/gemma-4-31b-it` | Best reasoning + thinking mode |
| Fallback | `google/gemma-4-12b-it` | Same family, ~7GB, 2x faster |
| Showcase | `meta-llama/Llama-3.3-70B-Instruct` | AMD exclusive (140GB VRAM) |

## Alternatives Evaluated

| Model | Why Rejected |
|-------|-------------|
| **Qwen3-32B** | ❌ MMLU-Pro 79% vs 85.2%; thinking mode is NOT unique to Qwen; 14 months old |
| **Gemma 4 26B-A4B (MoE)** | ❌ Only 3.8B active params during training; router excluded from LoRA |
| **Gemma 4 12B** | ✅ Viable fallback but 77.2% MMLU-Pro < 85.2%; same thinking mode |
| **Qwen3.6-27B** | ❌ Different family; DeltaNet architecture; less proven |
| **MedGemma 27B** | ❌ Beat by MeditronFO Gemma-3-27B; no thinking mode; Gemma 3 arch |

## VRAM Budget (MI300X — 192 GB)

| Component | bf16 LoRA | Notes |
|-----------|-----------|-------|
| Gemma 4 31B (bf16) | ~64 GB | Full precision, no quantization noise |
| LoRA adapters (r=64) | ~400 MB | All linear layers |
| Optimizer (AdamW) | ~800 MB | Only for LoRA params |
| Activations (grad ckpt) | ~15-25 GB | Gradient checkpointing enabled |
| **Total** | **~85-95 GB** | **~100 GB headroom ✅** |

## Storage Budget (Corrected — June 10, 2026)

| Storage | Size | What Goes Here |
|---------|------|----------------|
| `/` (ephemeral, 3.1 TB) | Model cache, raw data, packages | Re-downloadable |
| `/workspace/shared/` (persistent NFS, 28 GB) | LoRA adapters (~500 MB), final code snapshot | Must survive session restarts |

> **Key insight:** Only LoRA adapter weights (~500 MB) need to persist.
> The base model re-downloads from HuggingFace each session.

## Risk Mitigation

- bf16 LoRA uses ~95 GB of 192 GB VRAM → 100 GB headroom for safety
- If bf16 OOMs on a specific batch → reduce batch_size from 8→4 as runtime fix
- Save SFT checkpoint early — if GRPO fails, SFT alone is a valid submission
- Smoke test on first GPU session verifies bf16 compatibility
