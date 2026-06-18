# Architecture Decision Records (ADRs)

## What Are ADRs?

An ADR captures **a decision, its context, and the reasoning behind it.**
When you wonder "why did we pick X?" — the answer is here.

## How to Read an ADR

Each ADR follows this structure:
- **Status:** DECIDED ✅ or SUPERSEDED ⚠️
- **Context:** What problem needed a decision
- **Decision:** What we chose
- **Rationale:** Why we chose it (with alternatives considered)
- **Consequences:** What this decision means for the project

## ADR Index

| ADR | Topic | Status | One-Line Summary |
|-----|-------|--------|------------------|
| [ADR-001](ADR-001-problem-selection.md) | Problem Selection | ✅ Decided | Chose FINETUNING_005 — only problem with 4 measurable tasks + free data |
| [ADR-002](ADR-002-base-model.md) | Base Model | ✅ Decided | Gemma 4 31B (thinking mode, MMLU-Pro 85.2%) — Qwen3/Meditron3 rejected |
| [ADR-003](ADR-003-training-strategy.md) | Training Strategy | ✅ Final | SFT + WiSE-FT shipped (0.862). GRPO/DAPO/RAFT explored, validated ceiling. |
| [ADR-004](ADR-004-data-pipeline.md) | Data Pipeline | ✅ Decided | FAERS + BioDEX + OnSIDES; decontamination + adversarial negatives |
| [ADR-005](ADR-005-inference-demo.md) | Inference & Demo | ✅ Decided | Unsloth native + Ollama GGUF for local inference |
| [ADR-006](ADR-006-thinking-and-onsides.md) | Thinking + OnSIDES | ✅ Decided | Enable thinking for all tasks; OnSIDES for T3 labelling |
| [ADR-007](ADR-007-website-strategy.md) | Website | ✅ Decided | Astro-based project showcase at gemmra.bhaskarjha.dev |
