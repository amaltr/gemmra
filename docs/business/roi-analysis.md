# Gemmra — ROI & Business Impact Analysis

> Cost, speed, and regulatory value of automated pharmacovigilance assessment.

---

## The Business Case

### Current Manual Process

| Metric | Value | Source |
|--------|:-----:|---|
| FAERS reports per year | **2,000,000+** | FDA.gov |
| Avg review time per case | **30 minutes** | Industry benchmark |
| PV analyst hourly rate (US) | **~$40/hr** | Bureau of Labor Statistics |
| Cost per manual review | **~$20** | 30 min × $40/hr |
| **Total annual cost (US only)** | **$40M+** | 2M × $20 |
| Global PV market size (2024) | **$8.3B** | Grand View Research, Mordor Intelligence |
| Global PV market size (2030) | **$15–18B** | CAGR 12–14%, multiple sources |

### Gemmra AI Process

| Metric | Value | Calculation |
|--------|:-----:|---|
| Inference time per case | **10–20 sec** | Measured on AMD MI300X (varies by case complexity) |
| Throughput | **6.7 tok/s** | Single MI300X, bf16 precision |
| Cases per hour | **~200–400** | Depending on case complexity |
| GPU cost (MI300X cloud) | **~$3/hr** | AMD developer cloud pricing |
| **Cost per AI review** | **~$0.01** | ~$3/hr ÷ ~300 cases/hr |
| **Cost reduction** | **~2,000×** | $20 ÷ $0.01 |

### Scale Story

| Scenario | GPU Hours | GPU Cost | Manual Cost | Savings |
|----------|:---:|:---:|:---:|:---:|
| 100 cases | ~0.3 hrs | ~$1 | $2,000 | ~2,000× |
| 10,000 cases | ~33 hrs | ~$100 | $200,000 | ~2,000× |
| 2M cases (full FAERS) | ~6,700 hrs | ~$20K | $40,000,000 | ~2,000× |

---

## Regulatory Value

| Capability | Business Value |
|------------|---|
| **15-day compliance** | AI triages ALL incoming cases in hours, not weeks. Serious unlabelled events are flagged immediately per ICH E2D reporting requirements. |
| **Audit trail** | Thinking traces provide visible reasoning for every decision — satisfying regulatory requirement for documented rationale. |
| **Consistency** | 100% format compliance means zero manual reformatting. Every output is machine-parseable. |
| **Human-in-the-Loop** | Designed to augment reviewers, not replace them. Thinking traces give pre-assessed cases with visible reasoning to approve. |
| **Scale** | Process the entire FAERS backlog (20M+ reports) for systematic signal detection — impossible manually. |
| **E2B(R3) readiness** | FDA E2B(R3) deadline October 2026 — structured outputs align with the new ICSR format. |

---

## Headline Numbers

> **~2,000× cost reduction** — $20/case manual → ~$0.01/case AI

> **~100–180× speed improvement** — 30 minutes → 10–20 seconds per case (varies by complexity)

> **100% format compliance** — zero manual reformatting needed

> **Auditable reasoning** — every decision has a thinking trace for regulatory review

---

## Future Work: Revenue Opportunities

1. **PV-as-a-Service API** — SaaS for pharma companies (per-case pricing)
2. **Signal Detection Platform** — Aggregate case-level outputs → population-level safety signals
3. **Multi-regulatory** — Extend beyond FDA/FAERS to EMA (EudraVigilance), PMDA (Japan)
4. **Real-time FAERS Monitoring** — Stream incoming reports through Gemmra in real-time
5. **MedDRA Dictionary Integration** — Licensed MedDRA access for T2 score improvement (0.667 → 0.9+)
6. **Compliance Dashboard** — Automated regulatory reporting with audit trails
7. **TCS Life Sciences Pilot** — Production deployment within TCS PV operations
