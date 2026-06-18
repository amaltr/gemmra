# 📖 Documentation

Organized by topic — start with **Getting Started** to reproduce the full pipeline.

## Sections

### [Getting Started](getting-started/)
Setup, environment configuration, and step-by-step reproduction guides.
- [**Execution Guide**](getting-started/execution-guide.md) — Full pipeline from setup to inference
- [**Local Training**](getting-started/local-training.md) — Fine-tuning on consumer hardware (12GB VRAM)

### [Architecture](architecture/)
System design, problem definition, and key technical decisions.
- [**Problem Statement**](architecture/problem-statement.md) — The 4 pharmacovigilance tasks
- [**Research Pipeline**](architecture/research-pipeline.md) — All 10 decision points
- [**Innovations**](architecture/innovations.md) — What makes this solution different
- [**Decisions (ADRs)**](architecture/decisions/) — Architecture Decision Records (ADR-001 through ADR-007)

### [Evaluation](evaluation/)
Benchmarks, results, error analysis, and training experiments.
- [**Gemmra-Bench**](evaluation/gemmra-bench.md) — The evaluation benchmark (3,645 samples)
- [**Score Progression**](evaluation/score-progression.md) — How scores evolved across versions
- [**Error Analysis**](evaluation/error-analysis.md) — Remaining failure modes
- [**Deep Analyses**](evaluation/analysis/) — GRPO failure post-mortem, SFT analyses, MeditronFO comparison

### [Research](research/)
Verified findings from model selection, data sources, and AMD platform research.
- [**Discoveries Log**](research/discoveries_log.md) — 39 verified discoveries
- [**Model Comparison**](research/model_comparison.md) — Gemma 4 vs Qwen3 vs Llama 3.3
- [**Fine-Tuning Techniques**](research/fine_tuning_techniques.md) — SFT vs GRPO vs DPO landscape
- [**Data Sources**](research/data_sources.md) — FAERS, OnSIDES, BioDEX
- [**AMD Platform**](research/amd_platform.md) — MI300X configuration and ROCm
- [**MeditronFO Reference**](research/meditron_fo_reference.md) — EPFL paper analysis

### [Domain Knowledge](domain/)
Pharmacovigilance background for contributors new to the field.
- [**Pharmacovigilance 101**](domain/pharmacovigilance_101.md) — What is PV and why it matters
- [**FAERS Schema**](domain/faers_schema.md) — FDA Adverse Event Reporting System tables
- [**Worked Examples**](domain/worked_examples.md) — Real PV cases with expert reasoning

### [Business & Impact](business/)
ROI, market analysis, and AMD hardware advantages.
- [**AMD Differentiators**](business/amd-differentiators.md) — Why MI300X enables this project
- [**ROI Analysis**](business/roi-analysis.md) — Market opportunity and cost savings

### [Diagrams](diagrams/)
Mermaid source (`.mmd`) and rendered SVG (`.svg`) architecture diagrams.
