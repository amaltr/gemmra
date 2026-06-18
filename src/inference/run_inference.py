"""
Gemmra — Interactive Inference Console & Demo Recorder.

Designed for the TCS & AMD AI Hackathon to demonstrate:
1. Real-time streaming output (including Gemma 4 thinking trace).
2. End-to-end latency and throughput (tokens/second).
3. Exact token counts (Input, Output, Total).
4. GPU VRAM utilization (Allocated, Reserved, Peak).
5. Cross-task worked examples (T1, T2, T3, T4) and Case-specific demos (A-E).
6. Cinematic demo recording mode (--demo) for screen capture.

Usage:
  python src/inference/run_inference.py              # Interactive console
  python src/inference/run_inference.py --demo        # Auto-run 3 curated cases for recording
  python src/inference/run_inference.py --demo --demo-save transcript.txt  # Save clean transcript
"""

import os
import sys
import re
import time
import argparse
import subprocess
import threading
import torch
from pathlib import Path

# AMD/ROCm override for Instinct GPUs
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.4.2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# Checkpoint Discovery
# ============================================================

def discover_checkpoints(override_path=None):
    """Find available model checkpoints."""
    paths = [
        "checkpoints/sft_raft",
        "checkpoints/grpo_final",
        "checkpoints/local_grpo",
        "checkpoints/sft",
        "checkpoints/local_sft",
    ]
    if override_path:
        paths = [override_path] + paths
        
    found = []
    for p in paths:
        if Path(p).exists():
            found.append(p)
    return found

# ============================================================
# Demo Preset Prompts
# ============================================================

PRESETS_BY_TASK = {
    "1": {
        "name": "Task 1: Seriousness Assessment (Warfarin GI Bleed)",
        "task": "T1",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: "
            "Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). "
            "Base your assessment on the clinical narrative provided. Think step by step."
        ),
        "user": (
            "Patient: 68-year-old female, currently taking WARFARIN SODIUM for Atrial fibrillation. "
            "The patient developed Gastrointestinal haemorrhage during treatment. Inpatient hospitalization was required."
        )
    },
    "2": {
        "name": "Task 2: MedDRA Coding (Breast Lymphoma spironolactone context)",
        "task": "T2",
        "system": (
            "You are a medical coder specializing in MedDRA terminology. Given an adverse event description from "
            "clinical or patient-reported text, map it to the correct MedDRA Preferred Term (PT). "
            "Think step by step about the medical terminology and clinical context."
        ),
        "user": (
            "From biomedical literature about Unknown: \"BACKGROUND\n"
            "Whereas lymphoma of the female breast is already rare, lymphoma of the male breast has only anecdotally "
            "been reported. Within a study of 32 lymphoma of the breast reported between 1973 and 2014 as Burkitt "
            "lymphoma, we observed a single male case, which we report here.\n\n\n"
            "METHODS\n"
            "A 72-years-old Caucasian man presented with a mass in his left breast. Clinical history included prior "
            "basal cell carcinoma, leiomyosarcoma, and administration of spironolactone. The reference pathology diagnos\"\n\n"
            "Identify and code the adverse drug reaction described in this text using MedDRA Preferred Term (PT)."
        )
    },
    "3": {
        "name": "Task 3: Labelling Status (Warfarin GI Bleed expectation)",
        "task": "T3",
        "system": (
            "You are a pharmacovigilance expert. Determine if the reported adverse event is listed in the drug's approved "
            "product label. Consider the drug's pharmacological class and mechanism of action in your assessment. "
            "An unlabelled adverse event requires expedited 15-day reporting. Think step by step."
        ),
        "user": (
            "Drug: WARFARIN SODIUM\n"
            "Pharmacological class: Vitamin K Antagonist\n"
            "Mechanism of action: Inhibits Vitamin K epoxide reductase\n"
            "Indication: Atrial fibrillation\n"
            "Reported adverse event: Gastrointestinal haemorrhage\n\n"
            "Based on the drug's pharmacological class, mechanism of action, and known safety profile, "
            "determine whether this adverse event is listed in the drug's approved product label."
        )
    },
    "4": {
        "name": "Task 4: Causality Assessment (Warfarin GI Bleed temporal/dechallenge)",
        "task": "T4",
        "system": (
            "You are a pharmacovigilance expert. Read the clinical case narrative carefully, extract the relevant "
            "evidence (temporal relationship, dechallenge, rechallenge, concomitant medications, confounding factors), "
            "and assess drug-event causality using WHO-UMC criteria: Certain, Probable, Possible, Unlikely, "
            "Conditional, Unassessable. Think step by step."
        ),
        "user": (
            "A 68-year-old female patient began treatment with WARFARIN SODIUM (Atrial fibrillation) on 2023-12-15. "
            "The patient developed Gastrointestinal haemorrhage on 2024-06-15. The drug was discontinued and the event "
            "resolved within 5 days. No rechallenge was performed. No other suspect drugs or confounding medical "
            "conditions were reported.\n\n"
            "Assess the causal relationship between WARFARIN SODIUM and Gastrointestinal haemorrhage using WHO-UMC criteria."
        )
    }
}

PRESETS_BY_CASE = {
    "A": {
        "name": "Case A: Warfarin / GI Bleed (Standard)",
        "task": "T1 & T4",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria "
            "and determine the causal relationship using WHO-UMC criteria. Think step by step."
        ),
        "user": (
            "Patient: 68-year-old female. Drug: WARFARIN SODIUM. Indication: Atrial fibrillation. "
            "Adverse reaction: Gastrointestinal haemorrhage. Hospitalization was required. The drug was stopped, "
            "and the bleeding resolved within 5 days. No rechallenge was performed."
        )
    },
    "B": {
        "name": "Case B: Ibuprofen / Mild Rash (Non-Serious)",
        "task": "T1 & T4",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria "
            "and determine the causal relationship using WHO-UMC criteria. Think step by step."
        ),
        "user": (
            "Patient: 34-year-old male. Drug: IBUPROFEN. Indication: Headache. "
            "Adverse reaction: Mild skin rash. No hospital admission, no disability, no life-threatening outcomes. "
            "The rash resolved when drug was stopped."
        )
    },
    "C": {
        "name": "Case C: Methotrexate / Liver Failure (Severe, Positive Rechallenge)",
        "task": "T1 & T4",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria "
            "and determine the causal relationship using WHO-UMC criteria. Think step by step."
        ),
        "user": (
            "Patient: 45-year-old female. Drug: METHOTREXATE. Indication: Rheumatoid arthritis. "
            "Adverse reaction: Hepatic failure. The event was life-threatening and required immediate ICU admission. "
            "The drug was stopped and liver enzymes normalized. Later, the drug was restarted, and liver enzymes "
            "spiked again immediately."
        )
    },
    "D": {
        "name": "Case D: Atorvastatin / Muscle Pain (MedDRA Challenge)",
        "task": "T2",
        "system": (
            "You are a medical coder specializing in MedDRA terminology. Given an adverse event description, "
            "map it to the correct MedDRA Preferred Term (PT) and System Organ Class (SOC). Think step by step."
        ),
        "user": (
            "Patient reported severe, aching muscle pain in both calves and thighs after taking Atorvastatin for 2 months. "
            "Identify and code the adverse drug reaction described."
        )
    },
    "E": {
        "name": "Case E: Infliximab / Infusion Reaction (Labelling)",
        "task": "T3",
        "system": (
            "You are a pharmacovigilance expert. Determine if the reported adverse event is listed in the drug's approved product label. "
            "Think step by step."
        ),
        "user": (
            "Drug: INFLIXIMAB. Adverse event: Infusion related reaction. "
            "Determine whether this adverse event is listed in the approved product label."
        )
    }
}

# ============================================================
# Custom Prompt Presets (Edge Cases for Demo)
# ============================================================

CUSTOM_PRESETS = {
    "C1": {
        "name": "Edge Case: Non-Serious Assessment (harder than YES)",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: "
            "Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). "
            "Base your assessment on the clinical narrative provided. Think step by step."
        ),
        "user": (
            "Patient: 29-year-old male, currently taking IBUPROFEN for Headache. "
            "The patient developed mild nausea and transient dizziness lasting 2 hours. "
            "No medical attention was sought. The symptoms resolved spontaneously."
        )
    },
    "C2": {
        "name": "Edge Case: Unassessable Causality (missing data)",
        "system": (
            "You are a pharmacovigilance expert. Read the clinical case narrative carefully, extract the relevant "
            "evidence (temporal relationship, dechallenge, rechallenge, concomitant medications, confounding factors), "
            "and assess drug-event causality using WHO-UMC criteria: Certain, Probable, Possible, Unlikely, "
            "Conditional, Unassessable. Think step by step."
        ),
        "user": (
            "A patient of unknown age and sex was prescribed UNKNOWN DRUG for an unknown indication. "
            "An adverse event of 'Headache' was reported. No temporal information is available. "
            "No dechallenge or rechallenge data exists. No concomitant medications were documented.\n\n"
            "Assess the causal relationship using WHO-UMC criteria."
        )
    },
    "C3": {
        "name": "Edge Case: Lay Language MedDRA Coding",
        "system": (
            "You are a medical coder specializing in MedDRA terminology. Given an adverse event description from "
            "clinical or patient-reported text, map it to the correct MedDRA Preferred Term (PT). "
            "Think step by step about the medical terminology and clinical context."
        ),
        "user": (
            "Patient-reported complaint: 'My muscles have been aching really bad in both legs ever since "
            "I started taking that cholesterol pill. It hurts to walk up stairs.'\n\n"
            "Identify and code the adverse drug reaction described in this text using MedDRA Preferred Terms."
        )
    },
}

# ============================================================
# Output Analysis Helpers
# ============================================================

def analyze_output(generated_text):
    """Analyze model output for thinking trace ratio and format compliance.
    
    Returns:
        thinking_ratio: fraction of output that is reasoning trace (0.0 - 1.0)
        format_compliant: whether output matches expected structured format
        thinking_chars: char count of thinking trace
        answer_chars: char count of final answer
    """
    # Normalize newlines to avoid platform-specific bugs
    generated_text = generated_text.replace('\r\n', '\n')
    
    # Try to match with closing tag
    think_match = re.search(r'<\|channel>thought\s*(.*?)\s*<channel\|>', generated_text, re.DOTALL)
    if think_match:
        thinking_chars = len(think_match.group(1).strip())
        parts = generated_text.split('<channel|>', 1)
        answer = parts[1].strip()
    else:
        # Fallback if no closing tag exists yet
        if '<|channel>thought' in generated_text:
            parts = generated_text.split('<|channel>thought', 1)
            thinking_content = parts[1].strip()
            # If the model didn't output <channel|> but outputted the structured fields directly,
            # we can split by the first field (e.g. SERIOUS:, MedDRA PT:, LABELLED:, WHO-UMC:)
            field_match = re.search(r'\b(SERIOUS:|MedDRA PT:|LABELLED:|WHO-UMC Causality:)', thinking_content, re.IGNORECASE)
            if field_match:
                split_idx = field_match.start()
                thinking_chars = len(thinking_content[:split_idx].strip())
                answer = thinking_content[split_idx:].strip()
            else:
                thinking_chars = len(thinking_content)
                answer = ""
        else:
            thinking_chars = 0
            answer = generated_text.strip()
            
    answer_chars = len(answer)
    
    total_chars = thinking_chars + answer_chars
    thinking_ratio = thinking_chars / total_chars if total_chars > 0 else 0.0
    
    # Format compliance: does output have expected structured fields?
    format_compliant = bool(
        re.search(r'SERIOUS:\s*(YES|NO)', generated_text, re.IGNORECASE) or
        re.search(r'LABELLED:\s*(YES|NO)', generated_text, re.IGNORECASE) or
        re.search(r'MedDRA PT:', generated_text, re.IGNORECASE) or
        re.search(r'WHO-UMC Causality:', generated_text, re.IGNORECASE)
    )
    
    return {
        "thinking_ratio": thinking_ratio,
        "thinking_chars": thinking_chars,
        "answer_chars": answer_chars,
        "format_compliant": format_compliant,
    }


def get_gpu_utilization():
    """Get GPU compute utilization % via rocm-smi (AMD) or nvidia-smi.
    
    Returns utilization as float (0-100) or None if unavailable.
    """
    # Try rocm-smi first (AMD MI300X)
    for cmd in [
        ["rocm-smi", "--showuse", "--json"],
        ["rocm-smi", "-u"],
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout
                # Try to parse percentage
                match = re.search(r'(\d+\.?\d*)\s*%', output)
                if match:
                    return float(match.group(1))
                # nvidia-smi csv output
                try:
                    return float(output.strip())
                except ValueError:
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


# ============================================================
# Inference Execution & Metrics Collection
# ============================================================

def warmup_model(model, tokenizer):
    """Run dummy generations to JIT-compile CUDA kernels for realistic input shapes.
    
    First .generate() call on ROCm/CUDA compiles fused attention kernels,
    adding 5-15s overhead. torch.compile also caches graph by input shape,
    so we warm up with a realistic-length input (~128 tokens) to prevent
    recompilation on the first real inference call.
    """
    print("  Warming up (compiling CUDA/ROCm kernels)...", end=" ", flush=True)
    # Use a realistic-length prompt to populate torch.compile graph cache
    dummy_text = "Warm up prompt. " * 30  # ~120 tokens
    dummy = tokenizer(text=[dummy_text], return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.inference_mode():
        model.generate(**dummy, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print("done.")


def run_inference(model, tokenizer, system_prompt, user_prompt, max_new_tokens=512):
    """Executes model inference with real-time streaming and accurate metrics.
    
    Architecture:
      - TextIteratorStreamer runs model.generate() in a BACKGROUND THREAD
      - Main thread consumes tokens from iterator → prints them live (streaming)
      - TTFT is measured precisely: time from generate() start to first yielded token
      - E2E latency, TPOT, throughput all measured from the streaming timestamps
      - stdout I/O happens in the main thread, NOT in the generation thread,
        so it cannot block or slow down the GPU computation
    """
    from transformers import TextIteratorStreamer
    
    # 1. Format prompt
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    if user_prompt:
        formatted_messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
    
    try:
        prompt = tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )
        
    print("\n" + "-" * 50)
    print("  INPUT PROMPT (First 200 & Last 200 chars)")
    print("-" * 50)
    if len(prompt) > 400:
        print(f"{prompt[:200]}\n\n... [TRUNCATED] ...\n\n{prompt[-200:]}")
    else:
        print(prompt)
    print("-" * 50)

    # 2. Tokenize
    inputs = tokenizer(
        text=[prompt],
        return_tensors="pt",
        truncation=True,
        max_length=8192
    )
    input_len = inputs["input_ids"].shape[1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # 4. Setup TextIteratorStreamer for non-blocking streaming
    #    - generate() runs in a background thread
    #    - Main thread iterates over streamer, prints tokens live
    #    - Generation thread is NOT blocked by stdout I/O
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=False
    )
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )
    
    # Launch generation in background thread
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    gen_thread = threading.Thread(
        target=lambda: model.generate(**generation_kwargs),
        daemon=True
    )
    gen_thread.start()
    
    # 5. Consume tokens from streamer — live streaming + real TTFT
    print("\n[STREAMING OUTPUT]")
    print("-" * 50)
    
    generated_chunks = []
    ttft = None  # Will be set on first token
    
    for chunk in streamer:
        if ttft is None and chunk:  # First non-empty chunk = first token
            ttft = time.perf_counter() - start_time
        print(chunk, end="", flush=True)
        generated_chunks.append(chunk)
    
    # Wait for generation thread to finish
    gen_thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    print("\n" + "-" * 50)
    
    # 6. Reconstruct full output for analysis
    generated_text = "".join(generated_chunks)
    generated_text_clean = generated_text.replace(tokenizer.eos_token or '', '')
    if tokenizer.pad_token and tokenizer.pad_token != tokenizer.eos_token:
        generated_text_clean = generated_text_clean.replace(tokenizer.pad_token, '')
    generated_text_clean = generated_text_clean.strip()
    
    # 7. Calculate Metrics
    latency = end_time - start_time
    # Count output tokens by re-encoding the generated text
    if hasattr(tokenizer, "encode"):
        output_token_ids = tokenizer.encode(generated_text_clean, add_special_tokens=False)
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
        output_token_ids = tokenizer.tokenizer.encode(generated_text_clean, add_special_tokens=False)
    else:
        try:
            encoded = tokenizer(text=[generated_text_clean], add_special_tokens=False)
            ids = encoded.get("input_ids", [])
            if len(ids) > 0 and isinstance(ids[0], (list, torch.Tensor)):
                output_token_ids = ids[0]
            else:
                output_token_ids = ids
        except Exception:
            output_token_ids = [0] * int(len(generated_text_clean.split()) * 1.3)
            
    output_tokens = len(output_token_ids)
    if output_tokens == 0:
        output_tokens = 1  # Avoid division by zero
    total_tokens = input_len + output_tokens
    throughput = output_tokens / latency if latency > 0 else 0
    
    # TPOT — Time Per Output Token (MLPerf Inference primary decode metric)
    # Decode time = E2E - prefill. TPOT = decode_time / (output_tokens - 1)
    if ttft is None:
        ttft = latency  # Fallback if no tokens were streamed
    decode_time = latency - ttft
    tpot = (decode_time / max(output_tokens - 1, 1)) * 1000  # ms
    
    # Prefill throughput — input tokens processed before first output token
    prefill_throughput = input_len / ttft if ttft > 0 else 0
    
    # Output quality analysis
    output_analysis = analyze_output(generated_text_clean)
    
    # GPU compute utilization
    gpu_util = get_gpu_utilization()
    
    # VRAM stats
    gpu_name = "CPU"
    vram_alloc = 0.0
    vram_res = 0.0
    vram_peak = 0.0
    vram_total = 0.0
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
    return {
        "latency": latency,
        "ttft": ttft,
        "tpot": tpot,
        "prefill_throughput": prefill_throughput,
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "thinking_ratio": output_analysis["thinking_ratio"],
        "thinking_chars": output_analysis["thinking_chars"],
        "answer_chars": output_analysis["answer_chars"],
        "format_compliant": output_analysis["format_compliant"],
        "gpu_name": gpu_name,
        "gpu_utilization": gpu_util,
        "vram_allocated": vram_alloc,
        "vram_reserved": vram_res,
        "vram_peak": vram_peak,
        "vram_total": vram_total
    }

# ============================================================
# Demo Recording Mode — Cinematic Auto-Run
# ============================================================

# ANSI escape codes for cinematic terminal output
class C:
    """ANSI color codes for demo output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"
    BG_CYAN = "\033[46m"
    BG_GREEN = "\033[42m"
    BCYAN   = "\033[1;36m"
    BGREEN  = "\033[1;32m"
    BYELLOW = "\033[1;33m"
    BRED    = "\033[1;31m"
    BWHITE  = "\033[1;97m"


DEMO_SEQUENCE = [
    {
        "banner_num": "1",
        "banner_total": "4",
        "drug": "ACTEMRA (tocilizumab)",
        "task": "T1 — Seriousness Assessment (ICH E2A)",
        "patient_summary": "69-year-old female · Cardiac arrest, PE, AKI · Fatal outcome",
        "system": (
            "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: "
            "Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). "
            "Base your assessment on the clinical narrative provided. Think step by step."
        ),
        "user": (
            "Patient: 69-year-old female, currently taking ACTEMRA (tocilizumab) for Rheumatoid arthritis.\n"
            "Adverse events reported: Acute kidney injury, ALT increased, AST increased, Blood glucose decreased, "
            "Blood potassium increased, CRP increased, Cardiac arrest, Fibrin D dimer increased, Haemodialysis, "
            "Inflammatory marker increased, Platelet count decreased, Pulmonary embolism, Serum ferritin increased, "
            "Therapy non-responder.\n"
            "Outcome: The patient did not survive the clinical episode. Permanent functional limitation was noted "
            "prior to death. Haemodialysis was administered indicating acute renal failure requiring intensive care."
        )
    },
    {
        "banner_num": "2",
        "banner_total": "4",
        "drug": "SPIRONOLACTONE",
        "task": "T2 — MedDRA Coding",
        "patient_summary": "72-year-old male · Breast mass with prior malignancies · Spironolactone use",
        "system": (
            "You are a medical coder specializing in MedDRA terminology. Given an adverse event description "
            "from clinical or patient-reported text, map it to the correct MedDRA Preferred Term (PT). "
            "Think step by step about the medical terminology and clinical context."
        ),
        "user": (
            "From biomedical literature about SPIRONOLACTONE:\n"
            "\"A 72-year-old Caucasian man presented with a mass in his left breast. "
            "Clinical history included prior basal cell carcinoma, leiomyosarcoma, and administration of spironolactone. "
            "Histological examination revealed a diffuse lymphoid infiltrate with medium-sized cells exhibiting "
            "a high mitotic rate. The final diagnosis was Burkitt lymphoma of the male breast.\"\n\n"
            "Identify and code the adverse drug reaction described in this text using MedDRA Preferred Term (PT)."
        )
    },
    {
        "banner_num": "3",
        "banner_total": "4",
        "drug": "WARFARIN SODIUM",
        "task": "T3 — Labelling Assessment",
        "patient_summary": "68-year-old female · GI haemorrhage · Check if event is labelled",
        "system": (
            "You are a pharmacovigilance expert. Determine whether the reported adverse event is already listed "
            "in the drug's approved labelling (package insert / SmPC). Respond with LABELLED: YES or LABELLED: NO, "
            "and provide the evidence or rationale. Think step by step."
        ),
        "user": (
            "Drug: WARFARIN SODIUM. Indication: Atrial fibrillation.\n"
            "Reported adverse event: Gastrointestinal haemorrhage.\n\n"
            "Is gastrointestinal haemorrhage listed in the approved labelling for Warfarin Sodium?"
        )
    },
    {
        "banner_num": "4",
        "banner_total": "4",
        "drug": "YESCARTA (axicabtagene ciloleucel)",
        "task": "T4 — Causality Assessment (WHO-UMC)",
        "patient_summary": "31-year-old female · Cytokine release syndrome · 5 days post-infusion",
        "system": (
            "You are a pharmacovigilance expert. Read the clinical case narrative carefully, extract the relevant "
            "evidence (temporal relationship, dechallenge, rechallenge, concomitant medications, confounding factors), "
            "and assess drug-event causality using WHO-UMC criteria: Certain, Probable, Possible, Unlikely, "
            "Conditional, Unassessable. Think step by step."
        ),
        "user": (
            "A 31-year-old female patient received YESCARTA (axicabtagene ciloleucel) for Follicular lymphoma. "
            "The patient developed Cytokine release syndrome 5 days after treatment administration. "
            "No dechallenge information is available. The drug was not reintroduced (no rechallenge). "
            "No concomitant medications were reported. No obvious confounding medical conditions were identified.\n\n"
            "Assess the causal relationship between YESCARTA and Cytokine release syndrome using WHO-UMC criteria."
        )
    },
]


def run_demo_mode(model, tokenizer, selected_path, args):
    """Run curated demo cases with cinematic output for screen recording.
    
    Designed to produce terminal output that looks professional in a demo video:
    - Clear header banner with GPU/model info
    - Per-case banners with drug name and task
    - Streaming output with full thinking trace
    - Compact metric box after each case
    - Aggregate summary table at end
    """
    import io
    
    transcript_lines = []
    
    def tprint(text=""):
        """Print and optionally capture to transcript."""
        print(text)
        transcript_lines.append(text)
    
    # Resolve GPU info
    gpu_name = "CPU"
    vram_total = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # ── Header Banner ──
    tprint()
    tprint(f"{C.BCYAN}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    tprint(f"{C.BCYAN}║{C.RESET}  {C.BWHITE}GEMMRA — Live Inference Demo{C.RESET}                                {C.BCYAN}║{C.RESET}")
    tprint(f"{C.BCYAN}║{C.RESET}  {C.DIM}Model: team-gemmra/gemmra (SFT LoRA r=64, bf16){C.RESET}             {C.BCYAN}║{C.RESET}")
    tprint(f"{C.BCYAN}║{C.RESET}  {C.DIM}GPU:   {gpu_name} · {vram_total:.0f} GB HBM3{C.RESET}                        {C.BCYAN}║{C.RESET}")
    tprint(f"{C.BCYAN}║{C.RESET}  {C.DIM}Checkpoint: {selected_path}{C.RESET}                                   {C.BCYAN}║{C.RESET}")
    tprint(f"{C.BCYAN}╚══════════════════════════════════════════════════════════════╝{C.RESET}")
    tprint()
    
    metrics_list = []
    
    for i, case in enumerate(DEMO_SEQUENCE):
        # ── Case Banner ──
        tprint(f"{C.BYELLOW}━━━━ CASE {case['banner_num']} of {case['banner_total']} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
        tprint(f"  {C.BWHITE}Drug:{C.RESET} {case['drug']}")
        tprint(f"  {C.BWHITE}Task:{C.RESET} {case['task']}")
        tprint(f"  {C.DIM}Patient: {case['patient_summary']}{C.RESET}")
        tprint(f"{C.BYELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
        tprint()
        
        # ── Run Inference (uses existing streaming function) ──
        metrics = run_inference(
            model, tokenizer,
            case["system"], case["user"],
            args.max_tokens
        )
        metrics_list.append((case["drug"], case["task"], metrics))
        
        # ── Compact Metric Box ──
        think_pct = f"{metrics['thinking_ratio']*100:.0f}%"
        fmt_status = f"{C.BGREEN}✅ Format OK{C.RESET}" if metrics['format_compliant'] else f"{C.BRED}❌ Format FAIL{C.RESET}"
        
        tprint()
        tprint(f"  {C.DIM}┌─────────────────────────────────────────────────────┐{C.RESET}")
        tprint(f"  {C.DIM}│{C.RESET} {C.BCYAN}⏱ {metrics['latency']:.1f}s{C.RESET}  │  {C.BCYAN}🧠 {think_pct} thinking{C.RESET}  │  {fmt_status}  {C.DIM}│{C.RESET}")
        tprint(f"  {C.DIM}│{C.RESET} {C.BYELLOW}📊 {metrics['throughput']:.1f} tok/s{C.RESET}  │  {C.BYELLOW}💾 {metrics['vram_peak']:.1f} GB peak VRAM{C.RESET}           {C.DIM}│{C.RESET}")
        tprint(f"  {C.DIM}│{C.RESET} {C.DIM}TTFT: {metrics['ttft']*1000:.0f}ms  │  TPOT: {metrics['tpot']:.1f}ms  │  Tokens: {metrics['input_tokens']}→{metrics['output_tokens']}{C.RESET}  {C.DIM}│{C.RESET}")
        tprint(f"  {C.DIM}└─────────────────────────────────────────────────────┘{C.RESET}")
        
        # ── Pause between cases (for narration overlay) ──
        if i < len(DEMO_SEQUENCE) - 1:
            tprint()
            time.sleep(args.demo_pause)
    
    # ── Aggregate Summary ──
    tprint()
    tprint(f"{C.BGREEN}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    tprint(f"{C.BGREEN}║{C.RESET}  {C.BWHITE}DEMO COMPLETE — AGGREGATE PERFORMANCE{C.RESET}                       {C.BGREEN}║{C.RESET}")
    tprint(f"{C.BGREEN}╚══════════════════════════════════════════════════════════════╝{C.RESET}")
    tprint()
    
    total_latency = sum(m['latency'] for _, _, m in metrics_list)
    total_out = sum(m['output_tokens'] for _, _, m in metrics_list)
    total_in = sum(m['input_tokens'] for _, _, m in metrics_list)
    compliant = sum(1 for _, _, m in metrics_list if m['format_compliant'])
    avg_throughput = total_out / total_latency if total_latency > 0 else 0
    
    tprint(f"  {C.BWHITE}{'Drug':<30} {'Task':<20} {'E2E':>6} {'Tok/s':>6} {'Format':>8}{C.RESET}")
    tprint(f"  {C.DIM}{'─' * 75}{C.RESET}")
    for drug, task, m in metrics_list:
        fmt = f"{C.BGREEN}PASS{C.RESET}" if m['format_compliant'] else f"{C.BRED}FAIL{C.RESET}"
        task_short = task.split("—")[0].strip() if "—" in task else task[:18]
        tprint(f"  {drug:<30} {task_short:<20} {m['latency']:>5.1f}s {m['throughput']:>5.1f} {fmt:>8}")
    tprint(f"  {C.DIM}{'─' * 75}{C.RESET}")
    tprint(f"  {C.BWHITE}{'TOTAL':<30} {f'{len(metrics_list)} cases':<20} {total_latency:>5.1f}s {avg_throughput:>5.1f} {compliant}/{len(metrics_list)}{C.RESET}")
    
    tprint()
    if metrics_list:
        last = metrics_list[-1][2]
        tprint(f"  {C.DIM}GPU:  {last['gpu_name']}{C.RESET}")
        tprint(f"  {C.DIM}VRAM: {last['vram_peak']:.1f} / {last['vram_total']:.1f} GB ({last['vram_peak']/last['vram_total']*100:.0f}% utilization){C.RESET}")
        tprint(f"  {C.DIM}Model: team-gemmra/gemmra (LoRA r=64, α=128, bf16){C.RESET}")
        if last['gpu_utilization'] is not None:
            tprint(f"  {C.DIM}GPU Compute: {last['gpu_utilization']:.0f}%{C.RESET}")
    
    tprint()
    tprint(f"  {C.BCYAN}Gemmra — from weeks of manual review to seconds of AI-assisted assessment.{C.RESET}")
    tprint(f"  {C.DIM}Powered by AMD MI300X  ·  huggingface.co/team-gemmra/gemmra  ·  gemmra.bhaskarjha.dev{C.RESET}")
    tprint()
    
    # ── Save transcript if requested ──
    if args.demo_save:
        # Strip ANSI codes for clean text file
        import re as re_mod
        ansi_escape = re_mod.compile(r'\033\[[0-9;]*m')
        clean_lines = [ansi_escape.sub('', line) for line in transcript_lines]
        with open(args.demo_save, 'w', encoding='utf-8') as f:
            f.write('\n'.join(clean_lines))
        print(f"\n  \U0001f4c4 Transcript saved to: {args.demo_save}")
    
    # ── Custom Input Loop (for live demo) ──
    CUSTOM_SYSTEM_PROMPTS = {
        "T1": ("Seriousness (ICH E2A)",
               "You are a pharmacovigilance expert. Assess whether this adverse event case is SERIOUS per ICH E2A criteria: "
               "Death (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA). Think step by step."),
        "T2": ("MedDRA Coding",
               "You are a medical coder specializing in MedDRA terminology. Given an adverse event description, "
               "map it to the correct MedDRA Preferred Term (PT). Think step by step."),
        "T3": ("Labelling Assessment",
               "You are a pharmacovigilance expert. Determine whether the reported adverse event is already listed "
               "in the drug's approved labelling. Respond with LABELLED: YES or LABELLED: NO. Think step by step."),
        "T4": ("Causality (WHO-UMC)",
               "You are a pharmacovigilance expert. Assess drug-event causality using WHO-UMC criteria: "
               "Certain, Probable, Possible, Unlikely, Conditional, Unassessable. Think step by step."),
    }
    
    while True:
        print()
        print(f"\033[1;33m\u2501\u2501\u2501\u2501 CUSTOM INPUT \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\033[0m")
        print(f"  \033[1mT1\033[0m Seriousness  |  \033[1mT2\033[0m MedDRA  |  \033[1mT3\033[0m Labelling  |  \033[1mT4\033[0m Causality")
        print(f"  \033[2mQ  Quit\033[0m")
        print(f"\033[1;33m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\033[0m")
        
        try:
            task_choice = input(f"\n  \033[1;36mSelect task:\033[0m ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break
        
        if task_choice in ('Q', 'QUIT', 'EXIT', ''):
            break
        
        if task_choice not in CUSTOM_SYSTEM_PROMPTS:
            print(f"  \033[1;31mInvalid choice.\033[0m Enter T1, T2, T3, T4, or Q.")
            continue
        
        task_name, system_prompt = CUSTOM_SYSTEM_PROMPTS[task_choice]
        print(f"\n  \033[2mUsing {task_choice} ({task_name}). Paste your case below (press Enter twice to submit):\033[0m")
        
        # Multi-line input
        input_lines = []
        try:
            while True:
                line = input()
                if line == '' and input_lines and input_lines[-1] == '':
                    input_lines.pop()
                    break
                input_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
        
        user_input = '\n'.join(input_lines).strip()
        if not user_input:
            print(f"  \033[2mEmpty input, skipping.\033[0m")
            continue
        
        print(f"\n  \033[1mTask:\033[0m {task_choice} \u2014 {task_name}")
        print()
        
        # Run inference
        metrics = run_inference(
            model, tokenizer,
            system_prompt, user_input,
            args.max_tokens
        )
        
        # Compact metric box
        think_pct = f"{metrics['thinking_ratio']*100:.0f}%"
        fmt_ok = metrics['format_compliant']
        fmt_label = "\033[1;32m\u2705 Format OK\033[0m" if fmt_ok else "\033[1;31m\u274c Format FAIL\033[0m"
        
        print()
        print(f"  \033[2m\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\033[0m")
        print(f"  \033[2m\u2502\033[0m \033[1;36m\u23f1 {metrics['latency']:.1f}s\033[0m  |  \033[1;36m\U0001f9e0 {think_pct} thinking\033[0m  |  {fmt_label}  \033[2m\u2502\033[0m")
        print(f"  \033[2m\u2502\033[0m \033[1;33m\U0001f4ca {metrics['throughput']:.1f} tok/s\033[0m  |  \033[1;33m\U0001f4be {metrics['vram_peak']:.1f} GB peak VRAM\033[0m           \033[2m\u2502\033[0m")
        print(f"  \033[2m\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\033[0m")
    
    print(f"\n  \033[2mDemo session ended. Thank you!\033[0m\n")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gemmra — Interactive Inference Console & Demo Recorder")
    parser.add_argument('--ckpt', type=str, default=None, help='Override path to model checkpoint')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--demo', action='store_true',
                        help='Demo recording mode: auto-runs curated cases with cinematic output')
    parser.add_argument('--demo-pause', type=int, default=3,
                        help='Seconds to pause between demo cases (default: 3)')
    parser.add_argument('--demo-save', type=str, default=None,
                        help='Save demo transcript to file (e.g. demo_transcript.txt)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("          GEMMRA — INFERENCE CONSOLE")
    print("=" * 60)
    
    # 1. Discover Checkpoints
    discovered = discover_checkpoints(args.ckpt)
    if not discovered:
        print("ERROR: No checkpoints found! Searched in:")
        print("  - checkpoints/sft_raft")
        print("  - checkpoints/grpo_final")
        print("  - checkpoints/local_grpo")
        print("  - checkpoints/sft")
        print("  - checkpoints/local_sft")
        sys.exit(1)
        
    print("\nDetected Checkpoints:")
    for idx, path in enumerate(discovered):
        print(f"  [{idx + 1}] {path}")
        
    # Choose checkpoint
    if len(discovered) == 1:
        selected_path = discovered[0]
    else:
        try:
            choice = input(f"\nSelect checkpoint to load (1-{len(discovered)}) [Default: 1]: ").strip()
            if not choice:
                selected_path = discovered[0]
            else:
                selected_path = discovered[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Loading default.")
            selected_path = discovered[0]
            
    print(f"\n-> Loading model checkpoint: {selected_path}")
    
    # 2. Check VRAM and loading options
    load_in_4bit = "local" in selected_path
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB total VRAM)")
        # Automatically load in 4-bit if VRAM is tight (e.g. local test environment)
        if vram_gb < 40:
            load_in_4bit = True
    else:
        print("No CUDA device detected. Running on CPU (Expect slow performance).")
        
    print(f"Configured 4-bit quantization: {load_in_4bit}")
    
    # 3. Load model and tokenizer
    start_load = time.time()
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=selected_path,
            max_seq_length=8192,
            load_in_4bit=load_in_4bit,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        FastLanguageModel.for_inference(model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Failed to load via FastLanguageModel: {e}")
        print("Attempting standard transformers fallback...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(selected_path)
        model = AutoModelForCausalLM.from_pretrained(
            selected_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
    print(f"Loaded successfully in {time.time() - start_load:.2f} seconds.")
    
    # torch.compile — fuse attention kernels for faster decode on ROCm
    # reduce-overhead mode minimizes framework overhead for autoregressive generation
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile(mode='reduce-overhead') applied.")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")
    
    # Warmup pass — JIT-compile CUDA/ROCm kernels before any timed inference
    if torch.cuda.is_available():
        warmup_model(model, tokenizer)
    
    # 4. Demo mode or interactive loop
    if args.demo:
        run_demo_mode(model, tokenizer, selected_path, args)
        return
    
    # 5. Interactive loop
    while True:
        print("\n" + "=" * 60)
        print("  MAIN MENU")
        print("=" * 60)
        print("  [1] worked_examples.md - Task 1 (Seriousness Assessment)")
        print("  [2] worked_examples.md - Task 2 (MedDRA Coding)")
        print("  [3] worked_examples.md - Task 3 (Labelling Status)")
        print("  [4] worked_examples.md - Task 4 (Causality Assessment)")
        print("  ---")
        print("  [A] Demo Case A: Warfarin / GI Bleed (T1 & T4)")
        print("  [B] Demo Case B: Ibuprofen / Mild Rash (T1 & T4)")
        print("  [C] Demo Case C: Methotrexate / Liver Failure (T1 & T4)")
        print("  [D] Demo Case D: Atorvastatin / Muscle Pain (T2)")
        print("  [E] Demo Case E: Infliximab / Infusion Reaction (T3)")
        print("  ---")
        print("  ---")
        print("  [C1] Edge: Non-Serious Assessment (model says NO)")
        print("  [C2] Edge: Unassessable Causality (missing data)")
        print("  [C3] Edge: Lay Language MedDRA ('muscle ache' -> Myalgia)")
        print("  ---")
        print("  [CUST] Free-form Custom Prompt")
        print("  [ALL]  Run all Tasks sequentially (T1 -> T2 -> T3 -> T4)")
        print("  [EXIT] Quit Console")
        print("=" * 60)
        
        choice = input("Enter choice: ").strip().upper()
        
        if choice == "EXIT":
            print("Goodbye!")
            break
            
        system_prompt = ""
        user_prompt = ""
        name = ""
        
        if choice in PRESETS_BY_TASK:
            preset = PRESETS_BY_TASK[choice]
            name = preset["name"]
            system_prompt = preset["system"]
            user_prompt = preset["user"]
            
        elif choice in PRESETS_BY_CASE:
            preset = PRESETS_BY_CASE[choice]
            name = preset["name"]
            system_prompt = preset["system"]
            user_prompt = preset["user"]
            
        elif choice in CUSTOM_PRESETS:
            preset = CUSTOM_PRESETS[choice]
            name = preset["name"]
            system_prompt = preset["system"]
            user_prompt = preset["user"]
            
        elif choice == "CUST":
            name = "Custom User Scenario"
            system_prompt = input("\nEnter System Prompt:\n").strip()
            user_prompt = input("\nEnter User Prompt:\n").strip()
            
        elif choice == "ALL":
            print("\nRunning pipeline demonstration: Tasks 1 through 4...")
            metrics_list = []
            for t_idx in sorted(PRESETS_BY_TASK.keys()):
                preset = PRESETS_BY_TASK[t_idx]
                print(f"\n==========================================")
                print(f" Executing: {preset['name']}")
                print(f"==========================================")
                metrics = run_inference(
                    model, tokenizer, preset["system"], preset["user"], args.max_tokens
                )
                metrics_list.append((preset['name'], metrics))
                
            # Print comparative report at the end
            print("\n" + "=" * 70)
            print("       PIPELINE PERFORMANCE REPORT")
            print("=" * 70)
            # Table 1: Latency & Throughput
            print(f"{'Task':<40} | {'I/O Tok':<10} | {'E2E':<7} | {'Tok/s':<6} | {'TPOT':<7} | {'TTFT':<6} | {'Prefill':<8}")
            print("-" * 105)
            total_latency = 0.0
            total_out_tokens = 0
            total_in_tokens = 0
            for t_name, m in metrics_list:
                tok = f"{m['input_tokens']}/{m['output_tokens']}"
                print(f"{t_name[:38]:<40} | {tok:<10} | {m['latency']:.2f}s | {m['throughput']:.1f}  | {m['tpot']:.1f}ms | {m['ttft']*1000:.0f}ms | {m['prefill_throughput']:.0f} t/s")
                total_latency += m['latency']
                total_out_tokens += m['output_tokens']
                total_in_tokens += m['input_tokens']
            print("-" * 105)
            avg_tp = total_out_tokens / total_latency if total_latency > 0 else 0
            avg_tpot = (total_latency / total_out_tokens * 1000) if total_out_tokens > 0 else 0
            print(f"{'AGGREGATE':<40} | {f'{total_in_tokens}/{total_out_tokens}':<10} | {total_latency:.2f}s | {avg_tp:.1f}  | {avg_tpot:.1f}ms |      |")
            
            # Table 2: Output Quality
            print(f"\n{'Task':<40} | {'Think%':<8} | {'Think/Ans':<12} | {'Format':<8}")
            print("-" * 80)
            compliant_count = 0
            for t_name, m in metrics_list:
                think_pct = f"{m['thinking_ratio']*100:.0f}%"
                think_ans = f"{m['thinking_chars']}/{m['answer_chars']}"
                fmt = "PASS" if m['format_compliant'] else "FAIL"
                if m['format_compliant']:
                    compliant_count += 1
                print(f"{t_name[:38]:<40} | {think_pct:<8} | {think_ans:<12} | {fmt:<8}")
            print("-" * 80)
            compliance_rate = compliant_count / len(metrics_list) * 100 if metrics_list else 0
            print(f"Format Compliance: {compliant_count}/{len(metrics_list)} ({compliance_rate:.0f}%)")
            
            # Hardware footer
            if metrics_list:
                last_m = metrics_list[-1][1]
                print(f"\n  Model:    {selected_path}")
                print(f"  GPU:      {last_m['gpu_name']}")
                print(f"  Peak VRAM:{last_m['vram_peak']:.2f} / {last_m['vram_total']:.2f} GB ({last_m['vram_peak']/last_m['vram_total']*100:.1f}%)")
                if last_m['gpu_utilization'] is not None:
                    print(f"  GPU Util: {last_m['gpu_utilization']:.1f}%")
            print("=" * 70)
            continue
            
        else:
            print("Invalid choice, please try again.")
            continue
            
        # Run inference
        print(f"\n[Active Scenario: {name}]")
        print(f"System Prompt:\n{system_prompt}")
        print(f"User Prompt:\n{user_prompt}")
        
        metrics = run_inference(model, tokenizer, system_prompt, user_prompt, args.max_tokens)
        
        # Display Metrics Report
        print("\n" + "=" * 60)
        print("          INFERENCE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"  Model Checkpoint:     {selected_path}")
        print(f"  GPU Hardware:         {metrics['gpu_name']}")
        print("-" * 60)
        print("  LATENCY & THROUGHPUT")
        print(f"  End-to-End Latency:   {metrics['latency']:.3f} seconds")
        print(f"  TTFT (approx):        {metrics['ttft']*1000:.1f} ms")
        print(f"  TPOT (per token):     {metrics['tpot']:.1f} ms")
        print(f"  Decode Throughput:    {metrics['throughput']:.2f} tokens/second")
        print(f"  Prefill Throughput:   {metrics['prefill_throughput']:.0f} tokens/second")
        print("-" * 60)
        print("  TOKEN COUNTS")
        print(f"  Input Tokens:         {metrics['input_tokens']}")
        print(f"  Generated Tokens:     {metrics['output_tokens']}")
        print(f"  Total Tokens:         {metrics['total_tokens']}")
        print("-" * 60)
        print("  OUTPUT QUALITY")
        print(f"  Thinking Ratio:       {metrics['thinking_ratio']*100:.1f}% ({metrics['thinking_chars']} think / {metrics['answer_chars']} answer chars)")
        print(f"  Format Compliant:     {'PASS' if metrics['format_compliant'] else 'FAIL'}")
        print("-" * 60)
        print("  GPU RESOURCES")
        if torch.cuda.is_available():
            print(f"  VRAM Allocated:       {metrics['vram_allocated']:.3f} GB")
            print(f"  VRAM Reserved:        {metrics['vram_reserved']:.3f} GB")
            print(f"  Peak VRAM:            {metrics['vram_peak']:.3f} GB / {metrics['vram_total']:.3f} GB ({metrics['vram_peak']/metrics['vram_total']*100:.1f}%)")
            if metrics['gpu_utilization'] is not None:
                print(f"  GPU Compute Util:     {metrics['gpu_utilization']:.1f}%")
        else:
            print("  VRAM:                 N/A (CPU execution)")
        print("=" * 60)

if __name__ == "__main__":
    main()
