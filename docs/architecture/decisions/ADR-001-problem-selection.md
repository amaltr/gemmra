# ADR-001: Problem Statement Selection

**Status:** ✅ DECIDED  
**Date:** June 2026  
**Decision Makers:** Team (You + Amal)

## Context

The TCS-AMD AI Hackathon offers multiple problem statements across 3 tracks:
- Track 1: AI Agents (No-Code/Low-Code)
- Track 2: RAG (Intermediate)
- Track 3: Fine-Tuning (Advanced)

## Decision

**Chosen: FINETUNING_005 — AI-led Medical Review Assistant**

## Rationale

### Why Fine-Tuning Track (Track 3)?
1. **Highest differentiation** — fewer teams attempt fine-tuning (harder)
2. **AMD MI300X advantage is most visible** — 192GB needed for 70B model training
3. **Technical Implementation score is 40%** — fine-tuning demonstrates the deepest technical competence
4. **Cannot be faked** — judges can tell the difference between prompting and actual fine-tuning

### Why FINETUNING_005 Specifically?

| Factor | Score | Reasoning |
|--------|-------|-----------|
| **Multi-task opportunity** | ⭐⭐⭐⭐⭐ | Only problem with 4 distinct, measurable tasks |
| **Data availability** | ⭐⭐⭐⭐⭐ | FAERS = 20M+ free public cases from FDA |
| **Business relevance** | ⭐⭐⭐⭐⭐ | $15B+ pharmacovigilance market, TCS Life Sciences |
| **AMD showcase** | ⭐⭐⭐⭐⭐ | 70B model on single GPU = AMD exclusive |
| **Fine-tuning necessity** | ⭐⭐⭐⭐⭐ | MedDRA has 80K terms — can't prompt-engineer this |
| **Measurability** | ⭐⭐⭐⭐⭐ | F1, Kappa, Precision — all quantifiable metrics |

### Alternatives Considered

All other FINETUNING problems have fewer tasks, less available data, or weaker AMD showcase potential.

## Consequences

- Must deliver 4 working task outputs (not just 1)
- Requires domain knowledge in pharmacovigilance
- Requires multi-source data pipeline (FAERS + CADEC + synthetic)
- High payoff if executed well — no other team will match this scope
