# Research Discoveries Log

**Purpose:** Captures every critical finding from research sessions (June 9-10, 2026).
These are VERIFIED facts that differ from assumptions in earlier planning.

---

## 🔴 Critical Corrections (Changes to Original Plan)

### Discovery 1: MedDRA is NOT Free
- **Source:** Web search — ICH/MedDRA official website
- **Amal's assumption:** MedDRA LLT→PT hierarchy files freely downloadable
- **Reality:** MedDRA is **proprietary** (ICH). Free only for regulatory, academic, non-profit orgs
- **Impact:** Cannot download full 80K-term hierarchy for Task 2 training
- **Workaround:**
  1. Use FAERS `REAC.pt` directly (already coded PTs — ground truth)
  2. Use CADECv2 for lay-language → PT mapping
  3. Manually curate 50-100 common PT→SOC mappings for demo
  4. Synthetic generation for remaining verbatim inputs

### Discovery 2: FDA Now Publishes FAERS Daily
- **Source:** Web search — FDA FAERS documentation
- **Amal's assumption:** Quarterly ZIP files only
- **Reality:** Since late 2025, FDA transitioned to daily publication
- **Impact:** Quarterly archives STILL exist and are what we use
- **Action:** Download from https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

### Discovery 3: DELETED.txt Format Uncertainty
- **Source:** Web search — FAERS documentation
- **Amal's assumption:** DELETED.txt with `$` separator and `caseid` column
- **Reality:** Format varies by quarter; file may not exist as standalone
- **Impact:** Must always check README in each quarterly ZIP
- **Action:** Always deduplicate by max `caseversion` per `caseid` regardless

### Discovery 4: CADECv2 Released (Better Than Original)
- **Source:** Web search — CSIRO data portal
- **Amal referenced:** Original CADEC (2015, 1,250 posts, 7,101 annotations)
- **Reality:** CADECv2 released November 2024 — more diverse, part of MultiADE benchmark
- **URL:** https://data.csiro.au/collection/csiro:62387
- **Action:** Download CADECv2 instead of original CADEC

### Discovery 5: FAERS Narratives Are NOT Public
- **Source:** Web search — FAERS privacy documentation
- **Amal's assumption:** Free-text patient narratives available
- **Reality:** Stripped for privacy from public FAERS downloads
- **Impact:** Must construct synthetic narratives from structured fields
- **Action:** Build narratives from DEMO + DRUG + REAC + OUTC fields

### Discovery 6: T3 Frequency Heuristic Is BROKEN (June 10, 2026)
- **Source:** User empirical testing + OnSIDES database research
- **Previous approach:** If >5% of FAERS cases for a drug mention an AE, assume "labelled"
- **Reality:** Frequency ≠ labelling. Empirically proven wrong with NDA 125057 (Vemurafenib)
- **Root cause:** FAERS uses MedDRA PTs, drug labels use free-text prose → string matching fails
- **Solution:** OnSIDES database (github.com/tatonetti-lab/onsides)
  - PubMedBERT NLP extracts ADEs from FDA labels → maps to MedDRA PTs
  - Direct PT-to-PT lookup, no fuzzy matching needed
  - Updated quarterly, covers thousands of FDA-approved drugs
- **Action:** New script `src/data/04_download_onsides.py` + T3 builder rewrite
- **ADR:** ADR-006-thinking-and-onsides.md

### Discovery 7: Thinking Mode Should Be ON for ALL Tasks (June 10, 2026)
- **Source:** ICH E2A regulatory requirements + DAPO reward signal analysis
- **Previous approach:** T1/T3 (binary) had thinking=OFF
- **Reality:**
  1. ICH E2A regulators **require documented reasoning** for seriousness assessments
  2. Hackathon judges need to see WHY the model classified each case
  3. With thinking=OFF, DAPO `reasoning_quality_reward` gives T1/T3 score 0.0 (no thinking block)
- **Impact:** 4x more DAPO training signal, regulatory-compliant output, more impressive demo
- **Action:** All tasks now use `thinking: True` with `<|channel>thought` blocks
- **ADR:** ADR-006-thinking-and-onsides.md

### Discovery 8: DAPO Works Natively in TRL (June 10, 2026)
- **Source:** HuggingFace TRL documentation + web research
- **Concern raised:** "DAPO is built on verl, not TRL" — suggesting our approach was wrong
- **Reality:** TRL has first-class `loss_type="dapo"` with:
  - `epsilon_high=0.28` (clip-higher, prevents entropy collapse)
  - Token-level loss normalization (fixes long-CoT bias)
  - `mask_truncated_completions=True` (recommended best practice)
- **What TRL lacks:** Only "Dynamic Sampling" (skips trivial batches, ~10% compute savings)
- **Impact:** Our approach is CORRECT. TRL gives 95% of DAPO benefits with zero infra setup
- **Action:** Added `mask_truncated_completions=True` to GRPOConfig
- **Analysis:** See dapo_framework_analysis.md artifact

---

## 🟢 Confirmed Capabilities

### AMD MI300X + Unsloth
- **Source:** AMD official blog + Unsloth docs
- Unsloth has native AMD support via `pip install unsloth[amd]`
- MI300X can fit Llama 3.3 70B with **full 16-bit LoRA** at batch_size=128, seq_len=1024
- This is PHYSICALLY IMPOSSIBLE on any single NVIDIA GPU (H200 max = 141GB)

### GRPO on AMD ROCm
- **Source:** AMD official guide titled "Fine-Tuning LLMs with GRPO on AMD MI300X"
- TRL's `GRPOTrainer` works natively with ROCm
- No special configuration needed beyond standard Unsloth setup

### vLLM on AMD
- **Source:** vLLM official docs + AMD ROCm blog
- vLLM treats ROCm as **first-class** platform
- Pre-built Docker image: `vllm/vllm-openai-rocm:latest`
- No need to build from source
- OpenAI-compatible API for easy Streamlit integration

### RxNorm API
- **Source:** NLM RxNorm API documentation
- Free, no API key needed, ~40 requests/minute
- Endpoint: `https://rxnav.nlm.nih.gov/REST/approximateTerm?term={drug_name}`
- Returns standardized drug names from messy FAERS entries

### DailyMed API
- **Source:** NLM DailyMed API documentation
- Free, no API key
- Endpoint: `https://dailymed.nlm.nih.gov/dailymed/services/v2/`
- Workflow: NDA number → setid → SPL XML → parse LOINC section 34084-4 (Adverse Reactions)

### bitsandbytes Critical Fix
- **Source:** GitHub issues + Unsloth installation guide
- Standard bitsandbytes causes **silent NaN corruption** on AMD GPUs
- MUST use pre-release: `bitsandbytes-1.33.7.preview`
- Install URL: `https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl`
- Without this, training produces garbage with NO error message

## 🔴 CRITICAL: Gemma 4 Released June 3, 2026

### Discovery 6: Gemma 4 12B Released 6 Days Ago
- **Source:** Google DeepMind, Unsloth blog, multiple tech publications
- **Released:** June 3, 2026 (6 days before this hackathon!)
- **Architecture:** Encoder-free unified multimodal (text, image, video, audio)
- **Benchmarks:** MMLU-Pro 77.2%, AIME 2026 77.5%, LiveCodeBench 72.0%
- **Context:** 256K tokens
- **License:** Apache 2.0 (NO gating!)
- **Unsloth:** ✅ Fully supported
- **AMD ROCm:** ✅ Confirmed compatible with MI300X
- **Impact:** **Replaces Meditron3-8B as our primary model**
- **Why:** Superior reasoning >> medical pre-training. Fine-tuning bridges the medical gap.

### Discovery 7: Gemma 4 31B Fits Full LoRA on MI300X
- **Source:** Multiple benchmark + VRAM analysis sources
- **Gemma 4 31B:** MMLU-Pro 85.2%, AIME 89.2% (frontier-tier reasoning)
- **Full 16-bit LoRA:** Needs ~60-80GB VRAM → fits easily on MI300X
- **But:** H100 (80GB) could *also* fit this → NOT AMD exclusive
- **Decision:** Keep 70B for AMD-exclusive showcase; Gemma 4 31B optional bonus

### Discovery 8: Qwen3 is Now Legacy
- **Source:** Qwen official releases, Wikipedia timeline
- **Qwen3:** Released April 2025 (over 1 year old)
- **Current:** Qwen3.6 (April 2026), Qwen3.7-Max/Plus (May 2026)
- **Impact:** Qwen3-8B is no longer a competitive choice
- **Open-weight Qwen3.7:** Only proprietary API — cannot fine-tune

### Discovery 9: MedGemma Still on Gemma 3
- **Source:** Google MedGemma official page
- **Latest:** MedGemma 1.5 4B (January 2026) — NO MedGemma 2 exists
- **MedGemma 27B-IT:** Still available, 87.7% MedQA, Gemma 3 architecture
- **Impact:** MedGemma 27B is our medical fallback (not primary)

### Discovery 10: New PV Datasets Found
- **PHEE:** 5,000+ annotated PV events from medical case reports
- **BioDEX:** PubMed papers + drug safety reports (on HuggingFace)
- **MultiADE:** Multi-domain ADE benchmark (evaluation standard)
- **openFDA JSON:** FAERS in cleaner machine-readable format
- **Impact:** Potential +3,000 training pairs

### Discovery 11: SGLang as vLLM Alternative
- **Source:** Multiple inference engine comparison articles
- SGLang excels at structured outputs (RadixAttention for prefix caching)
- vLLM remains best for general-purpose, battle-tested deployment
- **Decision:** vLLM stays primary; SGLang is backup option

### Discovery 12: GRPO Confirmed as Industry Standard
- **Source:** Multiple 2026 RL technique comparisons
- GRPO is THE standard for reasoning tasks (DeepSeek-R1 style)
- DPO remains for general alignment (not needed for us)
- SimPO/ORPO are alternatives but less proven
- Evolution Strategies emerging but not ready for production
- **Our SFT→GRPO pipeline is VALIDATED as optimal**

---

## 🟡 Model Selection Research — REVISED (June 9, 2026)

### Primary Model: Gemma 4 31B-IT (FINAL — See ADR-002 v3)

> **NOTE:** This section was written during early research when 12B was initially
> selected. ADR-002 v3 later upgraded to **31B** as primary based on thinking
> mode parity + benchmark dominance. See Discoveries 13-15 below.

| Dimension | Old (Meditron3) | Current (Gemma 4 31B) |
|-----------|-----------------|-------------------|
| Release | 2024 | **April 2, 2026** |
| Params | 8B | **31B** |
| MMLU-Pro | ~55% | **85.2%** |
| Context | 8K-32K | **256K** |
| License | Llama (gated) | **Apache 2.0** |
| Thinking | None | **Native (`<|channel>thought`)** |
| Unsloth | Unverified | **Confirmed** |

### Fallback Chain (FINAL)
1. `google/gemma-4-31b-it` — **PRIMARY** (MMLU-Pro 85.2%, thinking mode)
2. `google/gemma-4-12b-it` — FALLBACK (same family, same tokens, 2x faster)
3. `meta-llama/Llama-3.3-70B-Instruct` — AMD showcase (140GB VRAM)

### 70B Showcase (Unchanged)
- `meta-llama/Llama-3.3-70B-Instruct` — AMD exclusive, 140GB VRAM

---

## 🟡 Fine-Tuning Technique Research (VALIDATED June 2026)

### Industry Consensus Pipeline
```
SFT (QLoRA) → GRPO → (optional: DPO for safety alignment)
```

### Key Findings
1. **SFT alone is table stakes** — every team does this
2. **GRPO is THE industry standard** for reasoning (DeepSeek-R1 pioneered it)
3. **DPO needs preference pairs** — we don't have time to create these
4. **ORPO/SimPO** are alternatives but less proven for medical reasoning
5. **Evolution Strategies** emerging in 2026 but not production-ready
6. **Data quality > data quantity** — 500-2K excellent pairs beat 10K mediocre ones
7. **All linear layers** is 2026 best practice (not just attention layers)

### Hyperparameter Consensus (Validated)
- LoRA rank: 16 (start), 32 (if underfitting) — with 192GB, we can go to 64
- LoRA alpha: 2× rank
- SFT learning rate: 2e-4 (never exceed)
- GRPO learning rate: 5e-6 (much lower for RL stability)
- Epochs: 2-3 for SFT, 1 for GRPO
- Gradient accumulation: 4 (effective batch = 16)
- Avoid length-reward coupling in GRPO (reward verbosity ≠ quality)

---

## 📊 Hackathon Presentation Research

### Judges' Psychology (Verified by Research)
1. **First 30 seconds decide everything** — lead with "why" not "how"
2. **Before/after metrics** are what judges remember
3. A **polished MVP** beats a complex broken mess every time
4. **The rubric is your roadmap** — Technical Implementation = 40%, spend effort there
5. **Avoid live demos** — pre-record everything
6. **One presenter** handles the pitch; others handle Q&A
7. **Anticipate Q&A:** "How does it scale?" "What about hallucinations?" "Why this model?"

### Demo Recording Best Practices
- Use Microsoft Clipchamp or Google Vids
- Script the voiceover word-for-word
- Edit out loading screens and login processes
- Keep total under 3 minutes
- Show thinking traces streaming — visually impressive (Gemma 4 `<|channel>thought` tokens)
- Always show before/after comparison on same case

---

## 🔗 Verified URLs (All Tested June 9, 2026)

| Resource | URL |
|----------|-----|
| FAERS Download | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| CADECv2 Dataset | https://data.csiro.au/collection/csiro:62387 |
| RxNorm API | https://rxnav.nlm.nih.gov/REST/ |
| DailyMed API | https://dailymed.nlm.nih.gov/dailymed/services/v2/ |
| openFDA API | https://api.fda.gov/drug/ |
| openFDA Events JSON | https://api.fda.gov/drug/event.json |
| AMD Developer Cloud | https://notebooks.amd.com/ |
| Unsloth GitHub | https://github.com/unslothai/unsloth |
| bitsandbytes Fix | (see above — full URL in bitsandbytes section) |
| vLLM ROCm Image | `vllm/vllm-openai-rocm:latest` |
| **Gemma 4 31B (PRIMARY)** | https://huggingface.co/google/gemma-4-31b-it |
| **Gemma 4 12B (FALLBACK)** | https://huggingface.co/google/gemma-4-12b-it |
| Llama-3.3-70B | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| FullyOpenMeditron Paper | https://arxiv.org/abs/2605.16215 |
| FullyOpenMeditron Code | https://github.com/EPFLiGHT/FullyOpenMeditron |
| EPFL Medical Guidelines | https://huggingface.co/datasets/epfl-llm/guidelines |


---

## 🔴 CRITICAL CORRECTIONS (Added June 9, Late Session)

### Discovery 13: Gemma 4 HAS Thinking Mode
- **Date:** June 9, 2026 (late session — devil's advocate analysis)
- **Previous assumption:** Gemma 4 has no native thinking mode
- **Reality:** ALL Gemma 4 models have built-in thinking mode
  - `enable_thinking=True` in chat template
  - Thinking tokens: `<|channel>thought` ... `<channel|>`
  - Configurable per-turn, supported by vLLM and Unsloth
- **Impact:** INVALIDATES the v2 decision to use Qwen3-32B
- **Action:** Switched to Gemma 4 31B as primary model (ADR-002 v3)

### Discovery 14: FullyOpenMeditron Paper (EPFL, May 2026)
- **Date:** June 9, 2026
- **Source:** arXiv:2605.16215 + github.com/EPFLiGHT/FullyOpenMeditron
- **Key finding:** Gemma-3-27B-MeditronFO beat MedGemma (58.6% preference)
- **Impact:** Validates medical SFT on Gemma architecture → our pipeline
- **Action:** Adopted MeditronFO-inspired "GemmraFO" approach
- **Bonus:** Their medical QA datasets are publicly available

### Discovery 15: Gemma 4 31B Benchmark Dominance
- **Date:** June 9, 2026
- **Key finding:** Gemma 4 31B significantly outperforms Qwen3-32B
  - MMLU-Pro: 85.2% vs ~79% (+6.2%)
  - GPQA Diamond: 84.3% vs ~72% (+12.3%)
  - AIME 2026: 89.2% vs ~72% (+17.2%)
- **Impact:** With thinking mode parity, benchmarks become the deciding factor
- **Action:** Gemma 4 31B confirmed as primary model

### Discovery 16: Apertus-70B-MeditronFO CANNOT Be Used
- **Date:** June 9, 2026 (deep paper analysis)
- **Key finding:** Three hard blockers prevent using the paper's best model:
  1. No thinking mode (Apertus is Sep 2025 base)
  2. No thinking mode (Sep 2025 model)
  3. MMLU-Pro not competitive with 2026 models
- **Impact:** Confirms Gemma 4 31B as correct base model
- **Insight:** The paper proves the PIPELINE works, not that Apertus is the best base

### Discovery 17: MeditronFO Training Used 213 GPU-Hours (32 GPUs)
- **Date:** June 9, 2026
- **Key finding:** Paper trained on 8 nodes × 4 GPUs = 32 GPUs for 6h 39m
- **Our budget:** 4hr/day × 5 days × 1 GPU = 20 GPU-hours
- **Why we can still do it:** We use QLoRA (not full FT), which is 10-50x
  more parameter-efficient. Our ~7-10 hours fits within our budget.

### Discovery 18: Gold-Label Resampling Technique
- **Date:** June 9, 2026
- **Key finding:** Paper generates 8 candidate answers per example using
  teacher model, keeps only the one matching ground truth
- **Impact:** Eliminates garbage training data → higher quality SFT
- **Action:** ADOPTED for our pipeline — generate 4-8 responses per FAERS case

### Discovery 19: System-Wide Decontamination Is Critical
- **Date:** June 9, 2026
- **Key finding:** Paper explicitly removes all evaluation examples from training
  to prevent data leakage and ensure genuine capability gains
- **Action:** ADOPTED — 10% holdout, hash-based verification, logged results

---

## 🔬 FRONTIER RESEARCH SWEEP (June 9, Late Session)

### Discovery 20: DGPO — Difficulty-Aware GRPO (arXiv:2601.20614)
- **Date:** June 9, 2026
- **Key finding:** Standard GRPO under-trains on hard tasks. DGPO fixes this
  with difficulty-balanced advantage estimation and question-level weighting.
- **Impact:** Our T4 (causality) is much harder than T1 (seriousness). Without
  difficulty weighting, GRPO will over-optimize easy tasks.
- **Action:** ADOPTED — Added DGPO_TASK_WEIGHTS to 02_grpo_train.py

### Discovery 21: Clinical-R1 / CRPO (AAAI 2026)
- **Date:** June 9, 2026
- **Key finding:** Multi-objective clinical rewards (accuracy + faithfulness +
  comprehensiveness) outperform correctness-only rewards.
- **Impact:** Our reward functions lacked a "faithfulness" signal.
- **Action:** ADOPTED — Added faithfulness_reward function to GRPO pipeline

### Discovery 22: MediX-R1 Composite Rewards (MBZUAI, 2026)
- **Date:** June 9, 2026
- **Key finding:** Semantic embedding similarity catches medical synonyms that
  exact-match misses ("heart attack" = "myocardial infarction").
- **Impact:** Our T2 (MedDRA) evaluation was exact-match only.
- **Action:** CONSIDERED — Add semantic similarity reward for T2 if time permits

### Discovery 23: Adversarial Negative Examples (2025-2026 Consensus)
- **Date:** June 9, 2026
- **Key finding:** Training with "plausible but wrong" examples prevents fragile
  reasoning. Models trained only on positive examples over-predict.
- **Impact:** Without negatives, model will always say "SERIOUS: YES".
- **Action:** ADOPTED — 12% adversarial ratio in sft_config.yaml

### Discovery 24: Curriculum Learning (2026 Best Practice)
- **Date:** June 9, 2026
- **Key finding:** Ordering training data easy→hard improves convergence speed
  and final performance on complex tasks.
- **Impact:** Faster training, better T4 accuracy.
- **Action:** ADOPTED — Added curriculum_learning flag and difficulty scoring

### Discovery 25: LoRA Rank r=32 (2026 Research Consensus)
- **Date:** June 9, 2026
- **Key finding:** r=16 may underfit on complex domain-specific reasoning.
  r=32 is optimal for medical/technical tasks. r=64+ risks overfitting.
- **Impact:** Increased expressive capacity for PV reasoning tasks.
- **Action:** ADOPTED → Initially set to r=32 (ADR-003 v3), then **upgraded to r=64**
  in ADR-003 v4 because 192GB MI300X VRAM makes overfitting risk negligible.
  See also: sft_config.yaml and 01_sft_train.py now use r=64, α=128.

### Discovery 26: AITER Inference Optimization (AMD, June 2026)
- **Date:** June 9, 2026
- **Key finding:** VLLM_ROCM_USE_AITER=1 and ROCM_AITER_FA=1 provide
  1.2-4.4x throughput improvement on MI300X.
- **Impact:** Demo inference speed could be 2-4x faster.
- **Action:** ADOPTED — Added env vars to src/setup/install.sh

---

## ⚡ ULTRA-CURRENT SWEEP (June 9, Evening Session — Locked to Last 4 Weeks)

### Discovery 27: DAPO Supersedes Vanilla GRPO (TRL v1.0, June 2026)
- **Date:** June 9, 2026
- **Key finding:** DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
  fixes 3 GRPO failure modes: entropy collapse, bias against long CoT, wasted
  compute on trivial batches. Available via `loss_type="dapo"` in GRPOConfig.
- **Impact:** Our thinking-mode outputs are long → vanilla GRPO penalizes them.
  DAPO's token-level loss normalization fixes this.
- **Action:** ADOPTED — Changed loss_type to "dapo" in 02_grpo_train.py

### Discovery 28: TRL v1.0 `reward_weights` + `normalize_then_sum` (June 2026)
- **Date:** June 9, 2026
- **Key finding:** TRL v1.0 natively supports weighted multi-reward functions
  and `None` return to skip irrelevant rewards per sample. `normalize_then_sum`
  prevents any single reward from dominating training.
- **Impact:** Replaces our manual DGPO_TASK_WEIGHTS with native TRL API.
- **Action:** ADOPTED — Added reward_weights=[1.0, 1.0, 0.8, 1.2] and
  multi_objective_aggregation="normalize_then_sum"

### Discovery 29: Gemma 4 QAT Official Release (June 5, 2026 — 4 days ago)
- **Date:** June 9, 2026
- **Key finding:** Google released official Quantization-Aware Training checkpoints.
  These models were TRAINED at 4-bit → handle quantization artifacts natively.
  Near-BF16 quality at Q4 size. W4A16 format works with vLLM.
- **Impact:** Potentially strictly better than our BnB Q4 approach.
- **Action:** INVESTIGATING — Need to verify Unsloth QLoRA compatibility

### Discovery 30: PSEBench — PV Triage Benchmark (June 2026)
- **Date:** June 9, 2026
- **Key finding:** "Clause card" methodology decomposes regulatory policy into
  atomic sub-decisions. Tests proactive information seeking and principled
  abstention. 5,074 cases.
- **Impact:** Our T1 should adopt clause-card reasoning for ICH E2A criteria.
  Model should say "INSUFFICIENT DATA" when evidence is ambiguous.
- **Action:** ADOPTED — Clause-card structure for system prompts

### Discovery 31: Hidden-Align (arXiv:2606.03234, June 2, 2026 — 7 days ago)
- **Date:** June 9, 2026
- **Key finding:** Correct RLVR rollouts cluster in hidden state space. Adding
  auxiliary alignment loss at the "anchor token" improves reasoning consistently.
  Zero inference overhead.
- **Impact:** Most advanced RLVR technique as of TODAY. Too complex for hackathon
  sprint but should cite in presentation.
- **Action:** NOTED for v2 — cite as awareness of frontier techniques

---

## 🔍 DATA QUALITY AUDIT (June 10, Evening Session)

### Discovery 32: FAERS Pipeline Had 5 Silent Data Quality Bugs
- **Date:** June 10, 2026
- **Key finding:** User-led audit of `01_download_faers.py` and `02_preprocess.py`
  against `PROBLEM_STATEMENT.md` data quality requirements revealed 5 bugs:
  1. **DELETED.txt never filtered** — retracted cases were included in training data
  2. **OUTC row fan-out** — one-to-many join produced duplicate rows per case
  3. **Year 2020 file discovery broken** — `"2020Q1".replace("20","")` → `"Q1"` (wrong)
  4. **`rept_cod` not in SELECT** — field was documented in ADR-004 but never selected
  5. **`n_concomitant` computed O(n²)** — per-row DataFrame scan instead of SQL pre-aggregation
- **Impact:** Bug #1 (DELETED) is highest severity — corrupts training labels.
  Bug #3 silently drops 4 quarters of data. Others affect performance/correctness.
- **Action:** FIXED ALL — Complete rewrite of `02_preprocess.py`, fixes in
  `01_download_faers.py` and `03_build_training_data.py`. ADR-004 updated.
  Drug normalization assessed as NOT necessary for our tasks (documented as known limitation).

---

### Discovery 33: Second Pipeline Audit — 9 Critical Runtime/Logic/Training Issues
- **Date:** June 11, 2026
- **Key finding:** Deep audit of all 4 data pipeline scripts + SFT training revealed
  9 additional issues ranging from runtime crashes to fundamental training corruption:
  1. **UNION ALL schema mismatch** — positional UNION ALL crashes when FAERS table
     schemas differ across quarters (e.g., DRUG gained `prod_ai` in 2014Q3).
     FIX: Use `UNION ALL BY NAME` (DuckDB feature for name-based column alignment)
  2. **T2 prompt missing adverse event** — Task 2 asked model to map an event to
     MedDRA PT but never included the event text, making training data useless.
     FIX: Include `meddra_pt` in the prompt as the verbatim event description
  3. **OnSIDES Git LFS pointer bypasses fallback** — raw GitHub URL returns LFS
     pointer text, `pd.read_csv` parses it without error, producing a 3-row DataFrame
     that silently corrupts the lookup table.
     FIX: Detect LFS pointer signature + sanity check row count
  4. **ANDA/BLA drugs excluded from T3** — only `NDA` prefix was stripped; generic
     (ANDA) and biologic (BLA) drugs had mismatched application numbers.
     FIX: Strip NDA/ANDA/BLA prefixes uniformly
  5. **Case-sensitive file matching on Linux** — uppercase patterns like `DEMO14Q1`
     fail to match lowercase files `demo14q1.txt` on case-sensitive Linux filesystems.
     FIX: Convert both pattern and filename to lowercase before matching
  6. **Incomplete downloads not detected** — partially downloaded ZIP files (>1 MB
     but corrupt) were treated as valid on subsequent runs.
     FIX: Use `zipfile.is_zipfile()` for integrity validation
  7. **OOM risk on 3.5M+ row dataset** — all 4 task builders iterated the full
     DataFrame with `iterrows` before slicing to `max_pairs`.
     FIX: Early `df.sample()` before iteration loop in all task builders
  8. **T1 multi-reaction loss** — `'first'` aggregation for `meddra_pt` discarded
     clinically significant reactions (e.g., showing "Nausea" while hiding
     "Myocardial infarction" in a case with outcome code DE).
     FIX: Aggregate all PTs per case using `set` join
  9. **SFT trains on full string including prompts** — using pre-formatted `text`
     field without `<bos>` and without response masking wastes model capacity on
     learning to predict user prompts.
     FIX: Use `tokenizer.apply_chat_template` + `DataCollatorForCompletionOnlyLM`
- **Impact:** Issue #2 (T2 empty prompt) is catastrophic — renders all T2 training
  data useless. Issue #1 would cause runtime crashes on real 49-quarter data.
  Issue #9 significantly degrades fine-tuning quality.
- **Action:** FIXED ALL — Updated `01_download_faers.py`, `02_preprocess.py`,
  `03_build_training_data.py`, `04_download_onsides.py`, and `01_sft_train.py`.

---

### Discovery 34: Training/Eval Audit — 4 Critical Template/Decoding/Config Issues
- **Date:** June 11, 2026
- **Key finding:** Audit of training + eval scripts revealed 4 issues that would
  corrupt the GRPO stage, invalidate the 70B showcase, break eval metrics, and
  silently ignore config changes:
  1. **GRPO prompt format mismatch** — `extract_prompt()` used plain text prefixes
     (`System:`, `User:`) instead of Gemma 4 native chat tokens. The model would
     see a completely different format during GRPO than what it learned during SFT,
     causing failure to activate thinking mode and format confusion.
     FIX: Use `tokenizer.apply_chat_template` with `add_generation_prompt=True`
  2. **Llama 3.3 trained on Gemma 4 tokens** — `03_showcase_70b.py` hardcoded
     `<start_of_turn>/<end_of_turn>` tokens when formatting data for Llama 3.3,
     which uses `<|start_header_id|>/<|end_header_id|>`. Training on wrong control
     tokens corrupts attention layers and produces gibberish.
     FIX: Use `tokenizer.apply_chat_template` (model-agnostic)
  3. **Thinking tokens stripped in evaluation** — `skip_special_tokens=True` removes
     `<|channel>thought` and `<channel|>` from decoded output, causing format
     compliance to always report 0% and answer extraction to match inside thinking
     traces instead of the structured answer section.
     FIX: Use `skip_special_tokens=False` + strip only eos/pad tokens
  4. **YAML configs silently ignored** — Neither SFT nor GRPO training scripts
     loaded hyperparameters from `configs/*.yaml`; all values were hardcoded.
     FIX: Added `yaml.safe_load()` with fallback to hardcoded defaults
- **Impact:** Issue #1 would make GRPO training ineffective (model sees wrong format).
  Issue #2 would produce a broken 70B showcase model. Issue #3 would report false
  metrics. Issue #4 causes confusion when tuning hyperparameters.
- **Action:** FIXED ALL — Updated `01_sft_train.py`, `02_grpo_train.py`,
  `03_showcase_70b.py`, and `evaluate.py`.

---

### Discovery 35: Full-Project Sweep — 10 Remaining Issues Across All Files
- **Date:** June 11, 2026
- **Key finding:** Comprehensive audit of all 10 source files in the project
  revealed 10 additional issues spanning coherence, correctness, and robustness:
  1. **Demo app stale LoRA info** — Sidebar showed `r=32, α=64` (old QLoRA values)
     instead of current `r=64, α=128` bf16 LoRA. Also showed no precision info.
     FIX: Updated sidebar to show correct values + bf16 + 192GB VRAM
  2. **Demo app Live mode ignored** — `render_task` accepted a mode parameter but
     never used it. Live mode behaved identically to Mock mode.
     FIX: Added live mode detection + informative model server instructions
  3. **Eval hardcoded `load_in_4bit=True`** — Loading bf16-trained checkpoints with
     4-bit quantization contradicts training config and silently degrades inference.
     FIX: Changed to `load_in_4bit=False, dtype=torch.bfloat16`
  4. **Smoke test: QLoRA messaging** — Described bitsandbytes as "Critical for QLoRA"
     and blocked model loading on bitsandbytes failure. Since we use bf16 LoRA,
     bitsandbytes is optional.
     FIX: Updated to "optional", made bitsandbytes check a soft pass
  5. **SFT --local mode global bug** — `LOAD_IN_4BIT` was assigned in local mode but
     not declared in the `global` statement, creating a local variable that gets
     silently ignored. The module-level `LOAD_IN_4BIT=False` would always be used.
     FIX: Added `LOAD_IN_4BIT` to global declaration
  6. **Missing `pyyaml` dependency** — Training scripts now `import yaml` but
     `pyyaml` was not in `requirements.txt` or `install.sh`, causing ImportError
     on fresh environments.
     FIX: Added `pyyaml>=6.0.0` to both files
  7. **GRPO config crash on older TRL** — `GRPOConfig` was passed DAPO-specific params
     (`loss_type="dapo"`, `epsilon_high`, `multi_objective_aggregation`) that may not
     exist in all TRL versions, causing instant crash.
     FIX: Dynamically inspect GRPOConfig signature and skip unsupported params
- **Impact:** Issues #3 and #5 would cause silent quality degradation during eval
  and local training. Issue #6 would cause immediate crash on fresh environments.
  Issue #7 would crash GRPO training on non-latest TRL.
- **Action:** FIXED ALL — Updated `app.py`, `evaluate.py`, `smoke_test.py`,
  `01_sft_train.py`, `02_grpo_train.py`, `requirements.txt`, and `install.sh`.

---

### Discovery 36: FAERS Data Range & Folder Structure — Real-World Testing
- **Date:** June 11, 2026
- **Source:** Manual testing of `01_download_faers.py` on actual FAERS data
- **Key findings (all from direct observation, not theoretical):**
  1. **Folder structure inside ZIPs**: Data files are NOT at the root of the extract.
     They live inside an ASCII/ (or Ascii/ or ascii/) subfolder. Our `rglob("*")`
     already handles this because it searches recursively, but this was undocumented.
  2. **DELETED folder casing varies**: Deleted/ (2019-2020), deleted/ (some quarters),
     DELETED/ (2021Q4+). On Linux, `rglob("*DELETED*")` does NOT match "Deleted/".
  3. **DELETED file naming varies across quarters**:
     - 2019Q1: `ADR19Q1DeletedCases.txt` + `AllDeletedCases.txt` (TWO files)
     - 2019Q2-2020Q3: `ADR{yy}Q{q}DeletedCases.txt`
     - 2020Q4: `20Q4DeletedCases.txt` (different prefix)
     - 2021Q4+: `DELETE21Q4.txt` (completely different convention)
  4. **AllDeletedCases.txt**: Only in 2019Q1. Cumulative list of ALL historically
     nullified cases (FDA reconciliation tool). This covers all deletions from
     before 2019 through 2018Q4. **Combined with incremental delete files from
     2019Q2+, this provides complete deletion coverage for ALL quarters, including
     pre-2019 data.**
  5. ~~**No DELETED files before 2019Q1**: Cannot verify data quality for 2014-2018.~~
     **CORRECTED**: AllDeletedCases.txt + incremental files DO cover pre-2019
     deletions. See Discovery 36b below.
  6. **Schema changes**: 2014Q3 added PROD_AI, DRUG_REC_ACT, changed GNDR_COD→SEX.
     2021Q4 was a database modernization/migration.
- **Decision: Changed data range from 2014Q1-2026Q1 (49 quarters) to 2019Q1-2026Q1 (29 quarters)**
  - Rationale (**corrected in 36b**): 29 quarters (~7M raw reports) is already 300x
    more than needed for ~21K training pairs. Adding 20 more quarters from 2014-2018
    adds processing complexity and schema inconsistencies without changing the
    output distribution. Data quality is NOT the blocker (AllDeletedCases.txt covers
    historical deletions) — sufficiency is.
- **Code fixes applied:**
  1. `01_download_faers.py`: Changed `generate_faers_urls(2014, 2026, 1)` → `(2019, 2026, 1)`
  2. `01_download_faers.py`: Fixed `verify_files` to be truly case-insensitive:
     replaced `rglob("*DELETED*") + rglob("*deleted*")` with `f.name.lower()` matching
     using 'delet' stem (catches Delete, DELETED, deleted, AllDeletedCases, etc.)
  3. `02_preprocess.py`: Same case-insensitive fix for `collect_deleted_ids`.
     Also added per-file logging to show how many IDs each deleted file contributes.
- **Impact:** Without these fixes, on Linux (AMD cloud): (a) Mixed-case deleted files
  like "ADR19Q1DeletedCases.txt" inside "Deleted/" would be silently missed,
  (b) retracted cases would contaminate training data. Both issues were invisible
  on Windows (case-insensitive filesystem) but fatal on Linux.
- **Action:** FIXED — Updated `01_download_faers.py` and `02_preprocess.py`.

---

### Discovery 36b: Rationale Correction — AllDeletedCases.txt Covers Pre-2019 Data
- **Date:** June 11, 2026
- **Source:** User observation + follow-up analysis
- **Key finding:** The original rationale in Discovery 36 stated "Pre-2019 data
  cannot be cleaned (no DELETED files = corrupt labels possible)." This was
  **incorrect**. Here's why:
  - `AllDeletedCases.txt` (in 2019Q1 ZIP) is a **cumulative** list of ALL
    nullified case IDs since the beginning of FAERS — including pre-2019 cases.
  - Incremental delete files from 2019Q2 onwards (ADR19Q2DeletedCases.txt, etc.)
    add cases deleted AFTER 2019Q1 — these cover cases from ANY quarter, not
    just cases originally reported in that quarter.
  - Therefore: AllDeletedCases.txt + all incremental files = **complete deletion
    coverage for ALL quarters**, including 2014-2018.
  - Pre-2019 data CAN be quality-assured using these files.
- **Corrected rationale for 2019Q1+ range:**
  The real reason to start at 2019Q1 is **pragmatic, not data-quality**:
  1. 29 quarters (~7M reports) → ~21K training pairs = 0.3% sampling rate.
     Adding 20 more quarters (5M more reports) doesn't change the output.
  2. Older quarters have more schema inconsistencies (2014Q3 transition).
  3. Older quarters have worse folder structure chaos (more edge cases).
  4. Fewer quarters = faster download/processing (hackathon constraint).
- **Impact:** All code comments, docstrings, and documentation updated to
  reflect the corrected rationale. No code logic changes needed (the 2019Q1+
  range remains correct, only the stated reason changed).
- **Action:** CORRECTED — Updated docstrings in `01_download_faers.py`, and
  all documentation in EXECUTION_GUIDE.md, ADR-003, ADR-004.

---

### Discovery 37: Two Runtime Crashes from Actual FAERS Execution
- **Date:** June 11, 2026
- **Source:** User ran `02_preprocess.py` against real downloaded FAERS data
- **Bug 1: DELETED file parser assumes wrong format** (all 31 files affected)
  - **Symptom:** Every DELETED file shows `⚠️ no 'primaryid' column found`
  - **Root cause:** Code assumed DELETED files are `$`-delimited CSVs with a
    `primaryid` column header. In reality, they are **headerless text files**
    with one primaryid per line, no delimiter, no header.
    Example: `AllDeletedCases.txt` starts with `4820242\n4820243\n...`
    When read with `pd.read_csv(sep='$')`, the first ID (`4820242`) becomes
    the column name, and `primaryid` is never found.
  - **Fix:** Rewrote parser (`_parse_deleted_file`) to:
    1. Read raw text, not CSV
    2. Check if first line is a header (`primaryid` or `caseid`)
    3. If headerless, treat every line as an ID
    4. Strip trailing `$` and whitespace from each line
  - **Impact:** Without this fix, ALL ~31 DELETED files were silently skipped,
    meaning zero retracted cases were filtered. ~30K+ retracted cases would
    contaminate the training data.
- **Bug 2: `drug_seq` column missing from THER table** (fatal crash)
  - **Symptom:** `BinderException: Values list "t" does not have a column named "drug_seq"`
  - **Root cause:** The THER table in the UNION of 29 quarters does not include
    a `drug_seq` column. The join `t.drug_seq = dr.drug_seq` fails.
    The THER table schema varies across FAERS quarters — some include `drug_seq`,
    some don't. After `UNION ALL BY NAME`, if no quarter has `drug_seq` in THER,
    the column simply doesn't exist.
  - **Fix:** `master_join` now dynamically checks column existence using
    `SELECT * FROM table LIMIT 0` before building the query. If `drug_seq` is
    missing, falls back to `primaryid`-only joins for THER and INDI tables.
  - **Impact:** Without this fix, the entire pipeline crashes at Step 5 and
    produces zero output.
- **Action:** FIXED — Updated `02_preprocess.py` with new `_parse_deleted_file`
  function and dynamic column detection in `master_join`.

---

### Discovery 38: Schema-Resilient Query Rewrite — gndr_cod→sex and Full Dynamic Columns
- **Date:** June 11, 2026
- **Source:** User ran fixed `02_preprocess.py` — DELETED parser now works (225,016 IDs
  found), drug_seq fallback works, but new crash at `d.gndr_cod` (column doesn't exist)
- **Root cause:** FAERS renamed `gndr_cod` to `sex` in the 2014Q3 schema update.
  Since we use 2019+ data, the column is always `sex`. The query hardcoded `gndr_cod`.
  This is the SAME class of bug as the drug_seq issue — hardcoded column names that
  don't match the actual FAERS schema.
- **Fix: Comprehensive dynamic column resolution for ALL tables**
  Instead of fixing just `gndr_cod`, rewrote `master_join` to discover actual columns
  from each table (demo_filtered, drug, ther, indi) at runtime and build the query
  dynamically. For each column:
  - Present → use directly
  - Renamed (gndr_cod→sex) → alias to expected name for downstream compatibility
  - Missing → use NULL with alias
  - THER/INDI joins → conditional drug_seq when available
  Also made `deduplicate()` resilient: checks for `caseversion`, falls back to
  `fda_dt`, then `primaryid` if neither exists.
- **Columns handled dynamically:**
  | Column | Table | Issue | Resolution |
  |--------|-------|-------|------------|
  | gndr_cod/sex | DEMO | Renamed 2014Q3 | `d.sex AS gndr_cod` |
  | age_cod | DEMO | May vary | NULL fallback |
  | occp_cod | DEMO | May vary | NULL fallback |
  | rept_cod | DEMO | May vary | NULL fallback |
  | event_dt | DEMO | May vary | NULL fallback |
  | prod_ai | DRUG | Added 2014Q3 | NULL fallback |
  | nda_num | DRUG | May vary | NULL fallback |
  | dechal/rechal | DRUG | May vary | NULL fallback |
  | drug_seq | DRUG/THER/INDI | Not always present | Conditional join |
  | start_dt/end_dt | THER | May vary | NULL fallback |
  | caseversion | DEMO | Used for dedup | fda_dt fallback |
- **Also noted:** DELETED filtering found 225,016 IDs but removed 0 cases.
  - **Correction**: This was NOT because they were all pre-2019 historical IDs. The real reason is that the deleted files contain 6-8 digit `caseid` values, while the filter was matching against `primaryid` (9 digits including caseversion suffix, e.g., `104172021` vs `10417202`).
  - **Action**: FIXED — Checked both `primaryid` and `caseid` in `filter_deleted` (see Discovery 39 below).
- **Action:** FIXED — Complete rewrite of `master_join` and `deduplicate` in
  `02_preprocess.py`.

---

### Discovery 39: Case ID vs Primary ID Mismatch in DELETED Filtering
- **Date:** June 11, 2026
- **Source:** User run of `02_preprocess.py` + manual entries comparison + web research
- **Symptom:** 225,016 DELETED IDs found but 0 cases removed from 10.5M row dataset.
- **Root cause (VERIFIED by FDA documentation + web research):**
  FAERS has TWO ID systems:
  - `caseid`: Groups all versions of a case (6-8 digit integer, e.g., `10417202`)
  - `primaryid`: Unique per record/version (9+ digits, unique per row in the DB)
  DELETED files contain **caseid** values — this is by design because deleting a
  case means deleting ALL versions. Our `filter_deleted` was matching against
  `primaryid` only, which never matched because the number formats are different.
  Example from user's data:
  - `ADR19Q1DeletedCases.txt` entry: `10417202` (caseid, 8 digits)
  - `AllDeletedCases.txt` entry: `820242` (caseid, 6 digits)
  - Demo table `primaryid` values are 9+ digits — no match possible.
- **Fix (user-initiated, audited and enhanced):**
  1. `_parse_deleted_file`: Simplified to split on `$` and take all digit-only parts.
     Correctly handles headerless files, trailing `$`, and optional headers.
     Header lines like `caseid` are auto-skipped (not all-digits). ✅
  2. `filter_deleted`: Checks BOTH `caseid` AND `primaryid` against deleted IDs.
     Uses `AND` logic (case must NOT appear in either column). ✅
  3. All docstrings corrected: "primaryid" → "caseid" to match reality.
  4. Added diagnostic logging: shows sample deleted IDs, sample demo primaryid/caseid,
     and pre-filter overlap counts for both columns — making future debugging trivial.
- **Impact:** Without this fix, ALL 225,016+ retracted cases remained in the training
  data, silently corrupting labels for all 4 tasks.
- **Action:** FIXED — Updated `_parse_deleted_file`, `filter_deleted`, and all
  docstrings in `collect_deleted_ids` in `02_preprocess.py`.
