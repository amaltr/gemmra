# Research Findings

> Research conducted during development, covering model selection, data sources,
> training techniques, and AMD platform configuration. Every claim verified.

## Documents

| File | What It Contains |
|------|-----------------|
| [discoveries_log.md](discoveries_log.md) | ⭐ **Start here** — 39 discoveries from data pipeline audits & model research |
| [model_comparison.md](model_comparison.md) | Model evaluation matrix (Gemma 4, Qwen3, Llama 3.3, MedGemma) |
| [fine_tuning_techniques.md](fine_tuning_techniques.md) | SFT vs LoRA vs GRPO vs DPO vs DAPO comparison |
| [data_sources.md](data_sources.md) | Every data source with URLs, formats, limitations |
| [amd_platform.md](amd_platform.md) | MI300X specs, known bugs, ROCm fixes |
| [scoring_strategy.md](scoring_strategy.md) | Evaluation criteria and target metrics |
| [meditron_fo_reference.md](meditron_fo_reference.md) | MeditronFO paper analysis — pipeline validation |

## Key Principle

Research docs capture **FACTS** (what we learned).  
Decision docs (`../decisions/`) capture **CHOICES** (what we decided based on those facts).

This separation means facts remain valid even if we change our decisions.
