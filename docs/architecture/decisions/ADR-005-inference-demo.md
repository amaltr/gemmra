# ADR-005: Inference & Demo Strategy

**Status:** ✅ DECIDED  
**Date:** June 9, 2026

---

## Decision

Use **vLLM** for inference serving and **Streamlit** for the demo UI.

## vLLM on AMD ROCm

- vLLM treats ROCm as **first-class** (confirmed June 2026)
- Pre-built image: `vllm/vllm-openai-rocm:latest`
- Provides OpenAI-compatible API → easy Streamlit integration
- 3-5× faster than `model.generate()` for interactive demos

### Startup Command
```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./grpo_final \
  --tokenizer ./grpo_final \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048
```

### Fallback
If vLLM has issues on AMD, use direct `model.generate()` with Unsloth's `FastLanguageModel.for_inference(model)`.

## Demo UI Design Principles

1. **Pre-loaded cases** — never type live during demo
2. **Streaming output** — show Gemma 4 thinking trace (`<|channel>thought`) streaming in real-time
3. **Dual-panel** — input on left, structured output on right
4. **Before/After toggle** — switch between base model and fine-tuned
5. **Professional color palette** — medical blues, clean whites, green for success
6. **Tab layout** — one tab per task (T1-T4)

## Implementation

See [`src/demo/app.py`](../../src/demo/app.py)
