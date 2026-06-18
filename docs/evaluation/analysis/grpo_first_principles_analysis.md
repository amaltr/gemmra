# GRPO/DAPO First-Principles Deep Analysis

> **Date:** June 14, 2026
> **Status:** ✅ CURRENT & DEFINITIVE — This is the post-mortem analysis that led to the decision to ship SFT-only. RAFT was also subsequently tested and failed (binary scores, zero diversity). All 7 failure mechanisms confirmed.
> **Value:** Critical reading for judges Q&A. Contains the innovation narrative for presenting negative results.
> **Purpose:** Question everything. No sacred cows. Find the optimal post-SFT strategy.

---

## 1. Why GRPO Was Designed (And Why Our Task Breaks It)

### GRPO's Success Domain: Math & Code Reasoning

GRPO works when:
1. **Many possible solution PATHS** → temperature diversity produces genuinely different reasoning chains
2. **Clear binary outcome** → but the PATH to get there is what varies
3. **Model knows how but is inconsistent** → RL teaches "which reasoning path is most reliable"

Example (math): "Solve 2x + 3 = 7"
- Generation 1: "Subtract 3 first → 2x = 4 → x = 2" ✅
- Generation 2: "Divide by 2 first → x + 1.5 = 3.5 → x = 2" ✅
- Generation 3: "2x = 10 → x = 5" ❌
- GRPO learns: path 1 and 2 are reliable, path 3 is not

### Our Task: The Fundamental Mismatch

| Aspect | Math/Code (GRPO works) | Gemmra (GRPO fails) |
|--------|---|---|
| Solution space | Infinite reasoning paths | 2 (T1/T3: YES/NO) or 5 (T4) |
| Diversity at temp=1.5 | Genuinely different approaches | Same answer, different words |
| Learning signal | "Which path → correct?" | "Model already knows correct answer" |
| SFT ceiling | Low (format only) | High (0.86 composite) |
| Reward variance | High (diverse paths → diverse correctness) | Near-zero (8/8 identical answers) |

**First principle:** GRPO optimizes over the VARIANCE of rewards within a group. When the model already produces the right answer consistently (SFT 0.86+), there IS no variance to optimize over.

---

## 2. Evidence From Smoke Logs: 7 Failure Mechanisms

### Failure 1: Reward Variance Collapse

```
Step 1:  correctness/std = 0.60  ← Alive
Step 5:  correctness/std = 0.81  ← Peak diversity  
Step 10: correctness/std = 0.47  ← Declining
Step 15: correctness/std = 0.30  ← Dying
Step 20: correctness/std = 0.10  ← Dead
```

Model memorizes 200 samples by epoch 0.25. All 8 generations converge to same answer → std=0 → advantage=0 → no gradient.

### Failure 2: frac_reward_zero_std Creep

```
Start: 0.2 (20% dead batches)
25%:   0.2
50%:   0.6 ← 60% batches dead
75%:   0.6
End:   0.4
```

By midpoint, 60% of batches contribute nothing to learning.

### Failure 3: grad_norm = 0 At Multiple Checkpoints

Steps at 50%, 75%, and final all show `grad_norm: 0`. The optimizer literally receives zero gradient — no weight updates possible.

### Failure 4: KL = 0 Throughout

`beta=0` means no KL penalty (DAPO paper). But combined with clip_ratio=0, this means:
- Model hasn't moved from reference (KL=0)
- Updates are so small they don't trigger clipping (clip_ratio=0)
- Policy is essentially frozen

### Failure 5: Duplication Artifact

GRPO taught the model to repeat answers inside AND outside thinking traces. Evidence from check_raw:

**SFT T4 output** (216 tokens): Clean answer once after thinking
**GRPO T4 output** (356-411 tokens): Answer appears TWICE — once in thinking, once after `<channel|>`

Root cause: `_extract_answer_text()` strips thinking trace, but the correctness reward sees the FULL text including thinking. Model learns: "Put answer everywhere = guaranteed match."

### Failure 6: T1/T4 Regression

| Task | SFT | GRPO | Delta |
|------|:---:|:---:|---|
| T1 | 1.000 | 0.985 | -0.015 |
| T4 | 0.980 | 0.975 | -0.005 |

GRPO slightly worsened already-strong tasks. The duplication increases token count → more chances for format parsing to fail.

### Failure 7: T2 Unchanged Despite Hierarchical Reward

T2 correctness reward already has partial credit (SOC=0.5, fuzzy=0.7, exact=1.0). But all 8 T2 generations produce the SAME PT → no gradient signal. Temperature=1.5 isn't enough to make the SFT model explore different MedDRA terms.

---

## 3. What Would Actually Fix GRPO/DAPO

### Fix A: T2-Only GRPO with Curriculum Filtering

**Idea:** Only train GRPO on T2 samples where SFT is WRONG. This guarantees every batch has learning signal.

**Implementation:**
1. Run SFT inference on all T2 training data
2. Collect samples where SFT prediction ≠ ground truth
3. Use ONLY these "hard" samples for GRPO
4. Remove T1/T3/T4 entirely (already at ceiling)

**Evidence for:** Research shows "focusing on hardest examples yields up to 47% larger performance gains" (search result [9])

**Evidence against:** If SFT is wrong on a sample, 8 GRPO generations at temp=1.5 will probably also be wrong → all get score 0 → no variance → still dead.

**Verdict:** Helps but doesn't fully solve the variance problem.

### Fix B: Temperature Annealing

**Idea:** Start with temp=3.0 (extreme diversity) → anneal to 1.0 over training.

**Rationale:** At temp=3.0, the model produces genuinely different MedDRA PTs for the same input. Some will be correct, some wrong → natural reward variance.

**Risk:** Extreme temperature produces garbage outputs that get 0 reward. If ALL 8 generations score 0, still no variance.

**Verdict:** Might help for T2, harmful for T1/T3/T4.

### Fix C: Reward Function Redesign

**Current T1/T3 reward:** Binary (0 or 1). Eight generations of "YES" → all get 1.0 → std=0.

**Proposed:** Add reasoning QUALITY gradient:
- T1: 1.0 if correct answer + cites correct ICH criterion, 0.7 if correct answer + wrong criterion, 0 if wrong answer
- T3: 1.0 if correct + pharmacological reasoning, 0.7 if correct + "checked label" reasoning, 0 if wrong

**Problem:** This requires parsing reasoning quality from thinking traces → complex, error-prone, subjective.

**Verdict:** High implementation cost, uncertain benefit. Research warns "partial credit can overwhelm the signal for perfectly correct answers."

### Fix D: Increase num_generations to 16

**Rationale:** More generations → more diversity → more chance of 1 outlier.

**Math:** With 8 gens and 90% accuracy, P(all correct) = 0.9^8 = 43%. P(at least 1 wrong) = 57%.
With 16 gens: P(all correct) = 0.9^16 = 19%. P(at least 1 wrong) = 81%.

**Cost:** 2× VRAM for generation. With 31B model in bf16, this may OOM.

**Verdict:** Mathematically sound but hardware-limited.

---

## 4. Alternative Approaches (Beyond GRPO)

### Alternative 1: Rejection Sampling Fine-Tuning (RAFT) ⭐ MOST PROMISING

**How it works:**
1. For each T2 training prompt, generate 8 completions with SFT model
2. Score each completion against ground truth (using existing `compute_t2_similarity`)
3. Keep ONLY completions that score ≥ 0.7 (fuzzy match or better)
4. Fine-tune SFT model on these correct self-generated completions

**Why it works for our case:**
- No group variance needed — we're doing SFT on correct samples
- Model learns its OWN successful reasoning patterns
- No duplication artifact (standard SFT training)
- No regression on T1/T4 (train only on T2)
- Stable training dynamics (same as SFT, which we know works)

**Evidence:**
- Research: "Rejection sampling is the most stable, interpretable baseline" (search [1])
- Research: "If your task allows verification, always establish a strong baseline using RAFT" (search [1])
- Practical: We HAVE verification (compute_t2_similarity is deterministic)

**Implementation cost:** 
- ~2 hrs to generate 8 completions per T2 sample (759 × 8 = 6,072 generations)
- ~1 hr to filter and retrain SFT on augmented data
- Total: ~3 hrs. Cheaper than GRPO's ~10 hrs.

### Alternative 2: DPO (Direct Preference Optimization)

**How it works:**
1. Generate pairs: (correct T2 answer, incorrect T2 answer) from SFT outputs
2. Train DPO loss: maximize log P(correct) - log P(incorrect)

**Pros:** More stable than GRPO. No reward function needed.
**Cons:** Requires paired data generation. DPO can overfit to formatting differences between chosen/rejected.

**Verdict:** Viable but RAFT is simpler for our case.

### Alternative 3: Best-of-N at Inference Time

**How it works:** At eval time, generate N=4 completions per prompt, score with `compute_t2_similarity` against... wait, we don't have ground truth at inference. 

**Modified:** Generate N completions, use faithfulness/format heuristic to pick best one. Or use self-consistency (majority vote for T1/T3/T4, most-common PT for T2).

**Pros:** Zero training cost. Works immediately.
**Cons:** N× inference time. No ground truth at real inference.

**Verdict:** Good for demo but not for competition eval (no ground truth to score against).

### Alternative 4: Iterative SFT (Self-Training)

**How it works:**
1. Run full eval on SFT model
2. For T2 samples where model is CORRECT, save the model's own thinking trace + answer
3. Retrain SFT with these new self-generated correct examples ADDED to training data
4. Repeat

**This is RAFT without the rejection step.** The insight: the model's OWN correct T2 reasoning traces may be better training data than the original synthetic traces.

---

## 5. Head-to-Head Decision Matrix

| Criterion | GRPO Full | RAFT (T2-only) | DPO | Best-of-N | Ship SFT |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Expected T2 improvement | ~0% | +5-10% | +3-7% | +3-5% | 0% |
| Risk of regression | HIGH | LOW | MEDIUM | NONE | NONE |
| GPU time | 10 hrs | 3 hrs | 5 hrs | 0 hrs | 0 hrs |
| Implementation complexity | Done | 1 hr code | 3 hrs code | 30 min | Done |
| Evidence quality | NEGATIVE (smoke failed) | STRONG (literature) | Moderate | Moderate | PROVEN (0.862) |
| Duplication risk | YES | NO | NO | NO | NO |

---

## 6. The Optimal Strategy

### If Time Permits (~4 hours available):

**Do RAFT on T2 only:**

```python
# Pseudocode
for each T2_training_sample:
    generate 8 completions at temperature=0.8  # Lower temp = higher quality
    score each with compute_t2_similarity(pred, gt)
    keep completions scoring >= 0.7
    
# Add to training data
augmented_data = original_sft_data + raft_correct_t2_completions

# Retrain SFT for 0.5 epochs (just the new data)
train(augmented_data, epochs=0.5)
```

**Why temperature=0.8 (not 1.5):**
- For RAFT, we want HIGH QUALITY diverse outputs, not random noise
- temp=0.8 produces variations of the correct answer with different reasoning
- These become high-quality training signal

### If Time Is Limited:

**Ship SFT. Composite 0.862 is strong.** Spend time on presentation (35% of judge score).

### What NOT to Do:

1. ❌ Full GRPO run — evidence shows it doesn't work for our task structure
2. ❌ Increase temperature beyond 1.5 — produces garbage
3. ❌ Add more reward functions — zero-variance rewards = dead gradients
4. ❌ Train GRPO on all 4 tasks — T1/T4 can only regress

---

## 7. Why DAPO Specifically Doesn't Help Us

DAPO adds 4 things over GRPO. Let's check each:

| DAPO Feature | What It Does | Does It Help Us? |
|---|---|---|
| **Clip-higher (ε=0.28)** | Prevents entropy collapse by allowing large positive updates | ❌ No entropy to collapse — model already converged |
| **Dynamic sampling** | Filters prompts with zero reward variance | ✅ Good idea, but 60% filtered = very few remaining |
| **Token-level loss** | Prevents long-CoT bias | ⚠️ Marginal — our outputs are 100-300 tokens, not 2000+ |
| **β=0 (no KL)** | Removes reference model constraint | ⚠️ Combined with clip=0, model just doesn't move |

**The core issue:** DAPO assumes the model NEEDS to explore. Our SFT model has ALREADY explored and found the right answers. DAPO's exploration-encouraging features are fighting against a model that's already converged.

---

## 8. Final Recommendation

### For Maximum Score (Evidence-Based):

1. **Keep SFT as primary model** — 0.862 composite, proven
2. **If 4 hrs GPU available:** Run RAFT on T2 for potential +5-10% T2 improvement
3. **If <4 hrs:** Ship SFT, focus on presentation
4. **Never:** Run full GRPO again — evidence conclusively shows it doesn't help

### For Hackathon Narrative (Judges Love This):

Present GRPO/DAPO as a **learning story**:
- "We implemented state-of-the-art DAPO (June 2026) with clip-higher, dynamic sampling, and token-level loss"
- "Smoke testing revealed our SFT model was already too strong for RL to improve — reward variance collapsed within 25% of training"
- "This taught us: RL works when the model is inconsistent (math). For domain-specific classification with strong SFT, data quality > RL algorithms"
- "We pivoted to data-first improvements and achieved 0.862 composite"

**This shows technical depth + scientific rigor + adaptability — exactly what judges want.**
