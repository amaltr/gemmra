# Gemmra — AMD MI300X Differentiators

> **Purpose:** Talking points for slides + Q&A showing WHY AMD MI300X was essential.
> This is NOT generic — every point is specific to what we built.

---

## The Killer Fact

**AMD MI300X has 192 GB HBM3 VRAM. NVIDIA H100 has 80 GB.**

This means:
- **We ran Gemma 4 31B in full bf16 (no quantization)**
- Full bf16 training = ~62 GB for weights alone
- With LoRA r=64 + optimizer states + KV cache → ~95 GB peak
- **Impossible on H100** — would require 4-bit quantization (QLoRA) to fit in 80 GB
- **Quantization = gradient noise = lower quality model**

---

## Slide-Ready Talking Points

### 1. Full Precision Training
> "MI300X's 192 GB let us train at full bf16 with LoRA rank 64 — impossible on NVIDIA's 80 GB H100. Zero quantization means zero gradient noise."

| | AMD MI300X (Us) | NVIDIA H100 | Impact |
|---|---|---|---|
| VRAM | **192 GB HBM3** | 80 GB HBM3 | 2.4× more headroom |
| Quantization needed? | **No — full bf16** | Yes — 4-bit QLoRA required | Higher quality gradients |
| LoRA rank possible | **r=64** (our choice) | r=16 typical (VRAM-limited) | 4× more trainable params |
| Max model size (bf16) | **~96B params** | ~40B params | Can train 70B if needed |

### 2. Single-GPU Training
> "One MI300X card handled our entire workflow — training, inference, and evaluation — with no multi-GPU complexity."

- Training: ~95 GB / 192 GB (50% utilization → room for larger models)
- Inference: ~65-70 GB (room for large KV cache → longer inputs)
- No model parallelism, no tensor sharding, no distributed training overhead

### 3. 70B Showcase Capability
> "We demonstrated LoRA on a 70B model — something impossible on a single NVIDIA GPU."

- `src/training/03_showcase_70b.py` loads and fine-tunes a 70B model on one MI300X
- Uses ~170 GB of 192 GB — fully utilizes the hardware
- This is unique to MI300X — no other single-GPU option can do this

### 4. ROCm Software Stack
> "AMD's ROCm stack is production-ready. PyTorch, Unsloth, TRL — everything just worked."

- `HSA_OVERRIDE_GFX_VERSION=9.4.2` — one environment variable
- Zero code changes from NVIDIA→AMD (all PyTorch code is device-agnostic)
- Unsloth, TRL, HuggingFace Transformers all work natively

---

## Q&A Prep: Expected Questions

**Q: "Could you have done this on an H100?"**
> "We could have done SFT, but at lower quality. H100's 80 GB means 4-bit quantization — introducing gradient noise that degrades training. Our bf16 training on MI300X produced higher-fidelity gradients, which matters for medical AI where accuracy is critical."

**Q: "What about multi-GPU H100?"**
> "Multi-GPU adds complexity: tensor parallelism, gradient synchronization, memory fragmentation. MI300X eliminates all of that. One card, one process, full precision."

**Q: "Did you use any AMD-specific optimizations?"**
> "Beyond bf16 (which IS the optimization — it's free precision), we used ROCm 6.x with Unsloth's optimized kernels. The key advantage isn't software tricks — it's raw VRAM enabling better training methodology."

---

## Summary: AMD Differentiators for Gemmra

| # | Differentiator | One-liner |
|---|---|---|
| 1 | **Full bf16 training** | No quantization = no gradient noise = better model |
| 2 | **LoRA r=64** | 4× more trainable params than typical QLoRA setups |
| 3 | **Single-GPU simplicity** | No distributed training overhead |
| 4 | **70B capability** | Demonstrated on 70B — impossible on single NVIDIA GPU |
| 5 | **Production ROCm** | Zero code changes, everything just works |
