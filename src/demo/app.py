"""
Gemmra — Streamlit Demo Application
Run: streamlit run src/demo/app.py

This demo showcases the 4 pharmacovigilance tasks with:
- Pre-loaded example cases
- Mock mode (works without GPU — uses pre-computed responses)
- Live mode (requires model loaded via vLLM or direct inference)
"""

import streamlit as st
import json
import time
from pathlib import Path

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Gemmra",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .thinking-box {
        background-color: #1e1e2e;
        color: #a6e3a1;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        border-left: 4px solid #89b4fa;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #a6e3a1;
        margin: 0.5rem 0;
    }
    .task-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .badge-t1 { background-color: #dbeafe; color: #1e40af; }
    .badge-t2 { background-color: #fce7f3; color: #9d174d; }
    .badge-t3 { background-color: #d1fae5; color: #065f46; }
    .badge-t4 { background-color: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Demo Cases (Pre-loaded)
# ============================================================

DEMO_CASES = {
    "Case A: ACTEMRA \u2014 Cardiac Arrest (Multi-Criteria)": {
        "patient": "69 year-old female",
        "drug": "ACTEMRA",
        "indication": "Not specified",
        "event": "Cardiac arrest, Pulmonary embolism, Acute kidney injury, + 11 more AEs",
        "outcomes": "DE (Death), LT (Life-threatening), HO (Hospitalization), DS (Disability)",
        "dechallenge": "N/A (patient deceased)",
        "rechallenge": "N/A",
        "temporal": "Not recorded",
        "mock_t1": {
            "thinking": "Pharmacovigilance expert.\nAssess seriousness per ICH E2A criteria.\nDeath (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA).\n\nPatient: 69-year-old female.\nDrug: ACTEMRA.\nAdverse events: Acute kidney injury, ALT increased, AST increased, Blood glucose decreased, Blood potassium increased, CRP increased, Cardiac arrest, Fibrin D dimer increased, Haemodialysis, Inflammatory marker increased, Platelet count decreased, Pulmonary embolism, Serum ferritin increased, Therapy non-responder.\n\nOutcome: \"The patient did not survive the clinical episode.\" -> Death (DE).\nOutcome: \"The patient suffered permanent functional limitation.\" -> Disability (DS).\nClinical severity: Cardiac arrest and pulmonary embolism are life-threatening conditions. -> Life-threatening (LT).\nTreatment: Haemodialysis suggests hospitalization or intensive care. -> Hospitalization (HO).\n\nDE (Death): Yes. The patient died.\nLT (Life-threatening): Yes. The clinical situation was critical.\nHO (Hospitalization): Yes. The severity of the event required hospitalization.\nDS (Disability): Yes. Permanent functional limitation is noted.\nCA (Congenital anomaly): No. Not applicable.\n\nThe case meets multiple seriousness criteria (DE, LT, HO, DS).\nFinal classification: Serious.",
            "answer": "SERIOUS: YES\nCriteria met: DE (Death), LT (Life-threatening), HO (Hospitalization), DS (Disability)\nRationale: The clinical outcome meets multiple seriousness categories, confirming serious classification."
        },
        "mock_t2": {
            "thinking": "The primary life-threatening event is cardiac arrest. In MedDRA terminology, this maps directly to the Preferred Term 'Cardiac arrest'. The patient also experienced pulmonary embolism, acute kidney injury, and multiple laboratory abnormalities.",
            "answer": "MedDRA PT: Cardiac arrest\nDrug context: ACTEMRA (tocilizumab)\nRationale: Cardiac arrest is the most clinically significant adverse event in this case."
        },
        "mock_t4": {
            "thinking": "Temporal: The exact timing between drug initiation and event onset is not recorded.\nDechallenge: Not applicable \u2014 patient deceased.\nRechallenge: Not applicable.\nConfounders: Multiple co-occurring adverse events complicate attribution.\nAlternatives: No concomitant medications reported.\n\nThe available information is insufficient \u2014 causality is Unassessable.",
            "answer": "WHO-UMC Causality: Unassessable\nEvidence:\n  - Temporal: No timing data available\n  - Dechallenge: Not applicable (patient deceased)\n  - Rechallenge: Not applicable\n  - Confounders: Multiple co-occurring AEs\n  - Alternatives: No concomitant medications reported"
        },
        "mock_t3": {
            "thinking": "",
            "answer": "LABELLED: YES\nDrug: ACTEMRA (tocilizumab)\nAdverse event: Cardiac arrest\nRationale: Cardiac events including cardiac arrest are documented in the ACTEMRA product label under Warnings and Precautions."
        }
    },
    "Case B: YESCARTA \u2014 Cytokine Release Syndrome": {
        "patient": "31 year-old female",
        "drug": "YESCARTA (axicabtagene ciloleucel)",
        "indication": "Follicular lymphoma",
        "event": "Cytokine release syndrome",
        "outcomes": "HO (Hospitalization)",
        "dechallenge": "N/A (single-dose CAR-T)",
        "rechallenge": "N/A",
        "temporal": "5 days after treatment",
        "mock_t1": {
            "thinking": "The adverse event led to hospitalization \u2014 serious per HO criterion. Cytokine release syndrome is a known serious reaction to CAR-T therapy. The case is serious.",
            "answer": "SERIOUS: YES\nCriteria met: HO (Hospitalization)\nRationale: The clinical outcome meets the hospitalization seriousness criterion."
        },
        "mock_t2": {
            "thinking": "The patient received YESCARTA (CAR-T therapy) for follicular lymphoma and developed cytokine release syndrome 5 days post-treatment. CRS is a well-characterized reaction to CAR-T therapies. MedDRA PT: Cytokine release syndrome.",
            "answer": "MedDRA PT: Cytokine release syndrome\nDrug context: YESCARTA (axicabtagene ciloleucel, CAR-T therapy)\nRationale: CRS is a recognized and expected adverse reaction to CAR-T cell therapies."
        },
        "mock_t4": {
            "thinking": "Pharmacovigilance expert.\nExtract evidence and assess causality using WHO-UMC criteria.\n31-year-old female. YESCARTA. Follicular lymphoma.\nCytokine release syndrome. 5 days.\n\nTemporal Relationship: The 5-day interval between drug start and event onset is within the expected window for this type of reaction.\nDechallenge: No dechallenge information is available \u2014 this key criterion cannot be evaluated.\nRechallenge: The drug was not reintroduced, so rechallenge evidence is absent.\nConfounders: No obvious confounders identified between the drug indication and the adverse event.\nAlternatives: The adverse event is distinct from the treated condition, reducing confounding risk.\nConcomitants: No concomitant medications were reported \u2014 the suspect drug is the sole agent.\n\nTemporal: Strong (5 days).\nDechallenge: Unknown.\nRechallenge: Not performed.\nConfounders: None.\nThe evidence is suggestive but not conclusive. The temporal link is strong, but the absence of dechallenge/rechallenge data limits the assessment to 'Possible'.",
            "answer": "WHO-UMC Causality: Possible\nEvidence:\n  - Temporal: Strong (5-day interval, within expected window)\n  - Dechallenge: Unknown (key criterion cannot be evaluated)\n  - Rechallenge: Not performed\n  - Confounders: None identified\n  - Alternatives: AE distinct from treated condition\n  - Concomitants: Sole agent"
        },
        "mock_t3": {
            "thinking": "",
            "answer": "LABELLED: YES\nDrug: YESCARTA (axicabtagene ciloleucel)\nAdverse event: Cytokine release syndrome\nRationale: CRS is prominently listed in the YESCARTA product label as a Boxed Warning."
        }
    },
    "Case C: HUMIRA \u2014 Nasopharyngitis (Non-Serious)": {
        "patient": "81 year-old female",
        "drug": "HUMIRA (adalimumab)",
        "indication": "Psoriatic arthropathy",
        "event": "Nasopharyngitis",
        "outcomes": "OT (Other)",
        "dechallenge": "Drug continued",
        "rechallenge": "N/A",
        "temporal": "Not recorded",
        "mock_t1": {
            "thinking": "Pharmacovigilance expert.\nAssess seriousness per ICH E2A criteria.\nDeath (DE), Life-threatening (LT), Hospitalization (HO), Disability (DS), Congenital anomaly (CA).\n\nPatient: 81-year-old female.\nDrug: HUMIRA.\nIndication: Psoriatic arthropathy.\nAdverse event: Nasopharyngitis.\nOutcome: The patient continued treatment and the symptoms subsided.\n\nDeath (DE): No death reported.\nLife-threatening (LT): No indication of immediate danger of death.\nHospitalization (HO): No mention of hospital admission.\nDisability (DS): No mention of long-term disability or functional impairment.\nCongenital anomaly (CA): No mention of pregnancy or birth defect.\n\nThe event was transient and resolved without meeting any serious criteria.\nClassification: Non-serious.",
            "answer": "SERIOUS: NO\nCriteria met: None\nRationale: The case does not meet any ICH E2A seriousness criteria."
        },
        "mock_t2": {
            "thinking": "The patient experienced nasopharyngitis (common cold). This maps directly to MedDRA PT 'Nasopharyngitis'. It is a frequently reported adverse event with HUMIRA.",
            "answer": "MedDRA PT: Nasopharyngitis\nDrug context: HUMIRA (adalimumab)\nRationale: Nasopharyngitis is a common adverse reaction with TNF inhibitors including adalimumab."
        },
        "mock_t4": {
            "thinking": "Temporal: The exact timing between drug initiation and event onset is not recorded.\nDechallenge: Drug was continued \u2014 no dechallenge performed.\nRechallenge: Not applicable.\nConfounders: No obvious confounders identified.\nAlternatives: No concomitant medications reported.\n\nThe available information is insufficient \u2014 causality is Unassessable.",
            "answer": "WHO-UMC Causality: Unassessable\nEvidence:\n  - Temporal: Not recorded\n  - Dechallenge: Drug continued (no dechallenge)\n  - Rechallenge: Not applicable\n  - Confounders: None identified\n  - Alternatives: No concomitant medications"
        },
        "mock_t3": {
            "thinking": "",
            "answer": "LABELLED: YES\nDrug: HUMIRA (adalimumab)\nAdverse event: Nasopharyngitis\nRationale: Upper respiratory tract infections including nasopharyngitis are listed in the HUMIRA product label as common adverse reactions."
        }
    },
}

# ============================================================
# App Layout
# ============================================================

def main():
    # Header
    st.markdown('<p class="main-header">💊 Gemmra</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Task Pharmacovigilance Assessment • Powered by Gemma 4 31B on AMD MI300X</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📂 Select Case")
        case_name = st.selectbox("Pre-loaded cases:", list(DEMO_CASES.keys()))
        
        st.divider()
        st.header("⚙️ Settings")
        mode = st.radio("Inference mode:", ["🎭 Mock (no GPU)", "🔥 Live (requires model)"])
        show_thinking = st.checkbox("Show thinking traces", value=True)
        
        st.divider()
        st.header("📊 Model Info")
        st.markdown("""
        - **Model:** Gemma 4 31B-IT
        - **Training:** SFT (LoRA r=64, bf16)
        - **Precision:** bf16 LoRA (zero quantization)
        - **LoRA:** r=64, α=128, all linear layers
        - **Data:** 32,355 pairs (FAERS + BioDEX + OnSIDES)
        - **Eval:** 3,645 decontaminated samples
        - **GPU:** AMD MI300X (192 GB HBM3)
        - **Inference:** ~10-20 sec/case
        """)
        
        st.divider()
        st.caption("TCS & AMD AI Hackathon 2026")
        st.caption("Track: Fine-Tuning (FINETUNING_005)")

    # Main content
    case = DEMO_CASES[case_name]
    
    # Case summary
    st.subheader(f"📋 {case_name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Patient:** {case['patient']}\n\n**Drug:** {case['drug']}")
    with col2:
        st.info(f"**Event:** {case['event']}\n\n**Indication:** {case['indication']}")
    with col3:
        st.info(f"**Outcomes:** {case['outcomes']}\n\n**Temporal:** {case['temporal']}")
    
    st.divider()
    
    # Task tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Task 1: Seriousness", 
        "🟣 Task 2: MedDRA Coding", 
        "🟢 Task 3: Labelling",
        "🟡 Task 4: Causality"
    ])
    
    with tab1:
        render_task(case, "T1", "Seriousness Assessment", "mock_t1", show_thinking, mode)
    
    with tab2:
        render_task(case, "T2", "MedDRA Code Suggestion", "mock_t2", show_thinking, mode)
    
    with tab3:
        render_task(case, "T3", "Labelling Status Evaluation", "mock_t3", show_thinking, mode)
    
    with tab4:
        render_task(case, "T4", "Causality Assessment", "mock_t4", show_thinking, mode)


def render_task(case, task_id, task_name, mock_key, show_thinking, mode):
    """Render a single task assessment."""
    st.subheader(f"{task_name}")
    
    is_live = "Live" in mode
    
    if st.button(f"▶️ Run {task_name}", key=f"btn_{task_id}"):
        if is_live:
            # Live mode: requires model loaded via vLLM or direct inference
            st.warning(
                "🔥 Live inference requires a running model server.\n\n"
                "Start the model with:\n"
                "`python -m vllm.entrypoints.openai.api_server --model checkpoints/sft/`\n\n"
                "Falling back to mock mode for this demo."
            )
        
        mock = case[mock_key]
        
        if show_thinking and mock.get("thinking"):
            st.markdown("**🧠 Thinking Trace:**")
            thinking_text = mock["thinking"]
            
            # Simulate streaming
            placeholder = st.empty()
            displayed = ""
            for word in thinking_text.split():
                displayed += word + " "
                placeholder.markdown(
                    f'<div class="thinking-box">{displayed}▌</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.03)
            placeholder.markdown(
                f'<div class="thinking-box">{thinking_text}</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("**📋 Assessment:**")
        st.markdown(
            f'<div class="answer-box">{mock["answer"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
        
        st.success(f"✅ {task_name} complete")


if __name__ == "__main__":
    main()
