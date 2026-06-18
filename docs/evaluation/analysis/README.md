# Deep Analysis Archive

> Detailed analysis documents from the development journey. These capture the
> reasoning, discoveries, and failures that shaped the final Gemmra model.
> Status banners indicate how current each document is relative to the shipped model.

## Documents (Ordered by Relevance)

### Key Analyses
| Document | Status | Summary |
|----------|:---:|---------|
| [grpo_first_principles_analysis.md](grpo_first_principles_analysis.md) | ✅ Current | GRPO/DAPO failure post-mortem. 7 failure mechanisms. Proves SFT ceiling. |
| [sft_v6_analysis.md](sft_v6_analysis.md) | ✅ Current | Analysis of the **shipped model**. Score evolution, training dynamics, architecture decisions. |
| [sft_v5_analysis.md](sft_v5_analysis.md) | ✅ Key | Found the **BioDEX truncation bug** → T2 improved 2.1×. |
| [data_quality_audit.md](data_quality_audit.md) | ✅ Resolved | Data issues that led to BioDEX fix and T3 rebalancing. |

### Research & Architecture
| Document | Status | Summary |
|----------|:---:|---------|
| [meditronfo_forensic_analysis.md](meditronfo_forensic_analysis.md) | ✅ Reference | MeditronFO paper deep dive. Model selection validated. |
| [task_design_deep_dive.md](task_design_deep_dive.md) | ✅ Reference | Template design for all 4 tasks. Key recommendations adopted. |
| [sft_eval_analysis.md](sft_eval_analysis.md) | ⚠️ Early | Eval of SFT v3 (before BioDEX fix). Captures debugging methodology. |
| [dapo_framework_analysis.md](dapo_framework_analysis.md) | ⚠️ Historical | Validated TRL's DAPO implementation — but DAPO didn't help. |
| [sft_first_principles_analysis.md](sft_first_principles_analysis.md) | ⚠️ Historical | SFT + GRPO loss function math. SFT sections valid, GRPO sections stale. |
