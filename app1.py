import os
import json
import requests
import datetime
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION & SECURITY ---
st.set_page_config(
    page_title="VA NeuroMetabolic Triage | Secure",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING (The "Competition Winner" Look) ---
st.markdown("""
    <style>
    /* Main Background - Soft Medical Gray */
    .stApp {background-color: #f0f2f6;}
    
    /* Header Bar - Clinical Navy */
    header[data-testid="stHeader"] {background-color: #0e1117;}
    
    /* Sidebar - Darker Contrast */
    section[data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #dcdcdc;}
    
    /* Card Styling for Results */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #0078d4; /* Microsoft Blue */
    }
    
    /* Badges */
    .badge-secure {
        background-color: #d4edda; 
        color: #155724; 
        padding: 5px 10px; 
        border-radius: 15px; 
        font-size: 0.8em; 
        font-weight: bold;
        display: inline-block;
    }
    .badge-risk-high {
        background-color: #f8d7da; 
        color: #721c24; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold;
        border: 1px solid #f5c6cb;
    }
    .badge-risk-low {
        background-color: #d1e7dd; 
        color: #0f5132; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        font-weight: bold;
        border: 1px solid #badbcc;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def reset_app():
    """Clears the risk score whenever inputs change."""
    st.session_state['risk_score'] = None

def get_safe_diagnosis_explanation(medical_data_json):
    """
    Connects to Azure OpenAI + Content Safety to explain the results.
    UPDATED: Uses a dynamic prompt based on the selected disease protocol.
    """
    # Check for keys
    if not os.getenv("AZURE_OPENAI_KEY") or not os.getenv("CONTENT_SAFETY_KEY"):
        return "⚠️ SYSTEM NOTE: AI Companion disabled (Keys missing)."

    try:
        # 1. TRANSLATE (Azure OpenAI)
        client_gpt = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
        
        # Extract the context from the package (defaults to GWI if missing)
        patient_context = medical_data_json.get("context", "Patient is a Gulf War Veteran")

        # --- UPDATED DYNAMIC PROMPT ---
        prompt = f"""
        You are a Clinical Metabolic Pathologist providing an objective analysis.
        
        CONTEXT: {patient_context}
        
        HEALTHY REFERENCE TARGETS (Higher values indicate superior health):
        - NAD/NADH Ratio: Target > 5.0 (Values < 3.0 indicate mitochondrial failure)
        - PCr/ATP Ratio: Target > 3.0 (Values > 5.0 indicate elite energy reserves)
        - GSH/GSSG Ratio: Target > 10.0 (Higher values confirm superior antioxidant capacity)
        
        INSTRUCTIONS:
        1. Begin with "Metabolic Analysis:"
        2. Analyze each biomarker in the context of the specific disease protocol selected.
        3. State the patient's value in **bold** and compare to reference ranges.
        4. If the Final Diagnosis is POSITIVE but biomarkers are normal, explain that the diagnosis is driven by the high symptom burden.
        5. Suggest next steps (e.g., "Consider confirmatory 31P-MRS").
        6. DO NOT use conversation fillers.
        
        RAW DATA: {medical_data_json}
        """
        
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation_text = response.choices[0].message.content

        # 2. SAFETY CHECK (Azure Content Safety)
        client_safety = ContentSafetyClient(
            endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("CONTENT_SAFETY_KEY"))
        )

        request = AnalyzeTextOptions(text=explanation_text)
        safety_result = client_safety.analyze_text(request)

        for analysis in safety_result.categories_analysis:
            if analysis.category == TextCategory.SELF_HARM and analysis.severity > 0:
                return "DETECTED_RISK: Please contact the Veteran Crisis Line: 988."

        return explanation_text

    except Exception as e:
        return f"System Note: AI Explanation unavailable ({str(e)})"

# --- LOGIC & API ---

def get_model_weights(protocol):
    """
    Returns the logistic regression coefficients based on the disease profile.
    NOTE: GWI weights are statistically derived from Haley et al.
    Long COVID and Alzheimer's weights are 'Literature-Derived Heuristics' for the prototype.
    """
    if protocol == "Gulf War Illness (Haley)":
        # Original validated weights
        return {
            "intercept": -18.07175,
            "pain": 0.86589,
            "confusion": 1.53029,
            "dizziness": 1.25769,
            "fatigue": 1.08384
        }
    elif protocol == "Long COVID / PASC (CDC)":
        # Hypothesis: Driven by PEM (Fatigue) and POTS (Dizziness)
        return {
            "intercept": -18.0,
            "pain": 0.4,
            "confusion": 1.2,
            "dizziness": 1.8,
            "fatigue": 2.5
        }
    elif protocol == "Early-Onset Alzheimer's":
        # Hypothesis: Driven almost entirely by Cognitive Decline
        return {
            "intercept": -20.0,
            "pain": 0.1,
            "confusion": 4.5,
            "dizziness": 0.8,
            "fatigue": 0.2
        }
    else:
        # Fallback
        return {"intercept": -18, "pain": 1, "confusion": 1, "dizziness": 1, "fatigue": 1}

def calculate_stage1_risk(pain, confusion, dizziness, fatigue, protocol):
    # 1. Get the weights for the ACTIVE protocol
    w = get_model_weights(protocol)
    
    # 2. Apply the Logistic Regression Formula
    logit = (
        w["intercept"] +
        (w["pain"] * pain) +
        (w["confusion"] * confusion) +
        (w["dizziness"] * dizziness) +
        (w["fatigue"] * fatigue)
    )
    
    # 3. Sigmoid Function
    try:
        probability = 1 / (1 + np.exp(-logit))
    except OverflowError:
        probability = 0.0 if logit < 0 else 1.0
        
    return probability

def call_azure_api(biomarkers, prior_prob, symptoms):
    # Retrieves Azure ML Model Credentials from Azure instead of Hard Coding
    endpoint = os.getenv("AZURE_ML_ENDPOINT")
    api_key = os.getenv("AZURE_ML_KEY")

    if not endpoint or not api_key:
        st.error("🚨 CRITICAL: Azure ML Environment Variables are missing!")
        return "Error: Missing Keys"
    
    # PAYLOAD (Correctly formatted for Azure Managed Endpoints)
    # We use "input_data" instead of "Inputs"
    payload = {
        "input_data": [
            {
                "nad_nadh": biomarkers['nad'],
                "pcr_atp": biomarkers['pcr'],
                "gsh_gssg": biomarkers['gsh'],
                "symptom_pain": symptoms['pain'],
                "symptom_cognitive": symptoms['cognitive'],
                "symptom_fatigue": symptoms['fatigue'],
                "symptom_vestibular": symptoms['vestibular'],
                "prior_probability": prior_prob
            }
        ],
        "GlobalParameters": 1.0
    }

    # Headers for Bearer Auth
    headers = {
        'Content-Type': 'application/json',
        'Authorization': (f'Bearer {api_key}')
    }

    # ACTUAL API CALL
    try:
        # Timeout set to 30s to prevent hanging
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30.0) 
        
        if response.status_code == 200:
            result = response.json()
            # Handle standard Azure ML return format (usually a list)
            if isinstance(result, list):
                return result[0] 
            return result
        else:
            # Show the specific error from Azure if it fails
            error_msg = f"API Error {response.status_code}: {response.text}"
            st.error(f"⚠️ {error_msg}")
            print(error_msg)
            return error_msg

    except Exception as e:
        st.error(f"🚨 Connection Failure: {str(e)}")
        return f"Error: {str(e)}"
        
# --- SIDEBAR (THE UNIVERSAL CHART) ---
with st.sidebar:
    # 1. THE LOGO (Accessible)
    st.markdown(
        """
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Microsoft_icon.svg/1024px-Microsoft_icon.svg.png" 
             alt="Microsoft Healthcare Logo" 
             width="50">
        """,
        unsafe_allow_html=True
    )
    
    # 2. THE PROTOCOL SELECTOR (The "Magic Trick")
    st.markdown("### ⚙️ Protocol Selector")
    protocol_mode = st.selectbox(
        "Select Diagnostic Kernel:",
        ["Gulf War Illness (Haley)", "Long COVID / PASC (CDC)", "Early-Onset Alzheimer's"],
        key="protocol_selector",
        on_change=reset_app
    )
    
    # --- DYNAMIC CONFIGURATION ENGINE ---
    config = {
        "Gulf War Illness (Haley)": {
            "title": "VA NeuroMetabolic Triage",
            "s1": "Joint Pain / Myalgia",
            "h1": "Rate severity (0-10) per Haley Protocol",
            "s2": "Chronic Fatigue",
            "h2": "Impact of fatigue on daily activities",
            "s3": "Cognitive Impairment",
            "h3": "Brain fog, memory loss, confusion",
            "s4": "Vestibular Dysfunction",
            "h4": "Dizziness, ataxia, balance issues"
        },
        "Long COVID / PASC (CDC)": {
            "title": "Post-Viral Triage Platform",
            "s1": "Post-Exertional Malaise",
            "h1": "Worsening of symptoms after minor exertion (PEM)",
            "s2": "Respiratory / Dyspnea",
            "h2": "Shortness of breath or air hunger",
            "s3": "Brain Fog / Neurocognitive",
            "h3": "Difficulty concentrating or finding words",
            "s4": "Dysautonomia / POTS",
            "h4": "Heart rate spikes, dizziness on standing"
        },
        "Early-Onset Alzheimer's": {
            "title": "Neuro-Degenerative Screen",
            "s1": "Agitation / Aggression",
            "h1": "Restlessness or behavioral outbursts",
            "s2": "Apathy / Withdrawal",
            "h2": "Lack of motivation or emotional flattening",
            "s3": "Short-Term Memory Loss",
            "h3": "Repetitive questions, forgetting recent events",
            "s4": "Spatial Disorientation",
            "h4": "Getting lost in familiar places"
        }
    }
    
    current_settings = config[protocol_mode]
    
    # 3. DYNAMIC HEADER & SECURITY BADGE
    st.markdown(f"### 🏥 {current_settings['title']}")
    st.markdown(f"<div class='badge-secure'>🔒 HIPAA SECURE | ID: {datetime.date.today()}</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 4. DYNAMIC SYMPTOM SLIDERS
    st.subheader("Patient Symptom Profile")
    st.info(f"Protocol: {protocol_mode}")
    
    input_pain = st.slider(
        current_settings['s1'], 0, 10, 1, key="pain",
        help=current_settings['h1'], on_change=reset_app
    )
    input_fatigue = st.slider(
        current_settings['s2'], 0, 10, 1, key="fatigue",
        help=current_settings['h2'], on_change=reset_app
    )
    input_confusion = st.slider(
        current_settings['s3'], 0, 10, 1, key="confusion",
        help=current_settings['h3'], on_change=reset_app
    )
    input_dizziness = st.slider(
        current_settings['s4'], 0, 10, 1, key="dizziness",
        help=current_settings['h4'], on_change=reset_app
    )
    
    st.markdown("---")
    analyze_btn = st.button("RUN TRIAGE PROTOCOL", type="primary")

# --- MAIN DASHBOARD ---
st.title("Clinical Decision Support Dashboard")

if 'risk_score' not in st.session_state:
    st.session_state['risk_score'] = None

if analyze_btn:
    st.session_state['risk_score'] = calculate_stage1_risk(
        input_pain, 
        input_confusion, 
        input_dizziness, 
        input_fatigue, 
        protocol_mode
    )

if st.session_state['risk_score'] is None:
    st.info("Please input patient symptoms in the left sidebar to begin triage.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("System Status", "Online", delta="Azure East US 2")
    with c2:
        st.metric("Model Version", "v2.0 (Universal)", delta="Active")
    with c3:
        st.metric("Pending Claims", "214,005", delta_color="inverse")

else:
    score = st.session_state['risk_score']
    score_pct = round(score * 100, 1)
    
    # RISK BANNER
    if score < 0.65:
        st.markdown(f"<div class='badge-risk-low'>✅ LOW PROBABILITY ({score_pct}%)<br>Recommendation: Standard of Care</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge-risk-high'>⚠️ HIGH PROBABILITY ({score_pct}%)<br>METABOLIC DYSFUNCTION DETECTED</div>", unsafe_allow_html=True)
        
    st.write("")
    st.subheader("🔓 Confirmatory Lab Interface")
    
    # The "Card" Look
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        val_nad = st.number_input("NAD / NADH Ratio", value=0.0)
    with c2:
        val_pcr = st.number_input("PCr / ATP Ratio", value=0.0)
        
    val_gsh = st.number_input("GSH / GSSG Ratio", value=0.0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    val_meta_index = (val_nad/2.0) + (val_pcr/1.8) + (val_gsh/30.0)
    st.write("")
    
    if st.button("ORDER CONFIRMATORY ANALYSIS"):
        if val_nad < 0 or val_pcr < 0 or val_gsh < 0:
            st.error("⚠️ Invalid Input: Biomarker ratios cannot be negative.")
            st.stop()
            
        biomarkers = {
            'nad': val_nad,
            'pcr': val_pcr,
            'gsh': val_gsh,
            'meta_index': val_meta_index
        }
        
        # --- LOGIC UPDATE: THE 'HARD OVERRIDE' ---
        is_metabolic_optimal = (
            (val_nad >= 8.0) and             
            (val_pcr >= 3.5 and val_pcr <= 6.0) and 
            (val_gsh >= 50.0)                
        )
        
        if is_metabolic_optimal:
            st.info("✅ **Clinical Note:** Biomarkers indicate optimal mitochondrial function. Symptom score overridden by objective metabolic data.")
            result = "Negative (Healthy)"
        else:
            current_score = score
            
            # 2. CALL AZURE NEURAL NETWORK
            with st.spinner(f"Connecting to Azure Neural Network (Prior Score: {round(current_score*100, 1)}%)..."):
                # Map dynamic inputs to standard keys
                symptoms_map = {
                    'vestibular': input_dizziness,
                    'pain': input_pain,
                    'cognitive': input_confusion,
                    'fatigue': input_fatigue
                }
                result = call_azure_api(biomarkers, current_score, symptoms_map)
            
            st.success(f"**FINAL DIAGNOSIS:** {result}")
            if "POSITIVE" in result:
                st.error("ACTION REQUIRED: Refer to Neurology.")
        
        st.markdown("---")
        st.subheader("🤖 AI Clinical Companion (Azure OpenAI)")
        
        with st.spinner("Generating plain-English explanation..."):
            data_package = {
                "diagnosis_result": result,
                "metabolic_index": val_meta_index,
                "biomarkers": biomarkers,
                "context": f"Patient is being screened for {protocol_mode}" 
            }
            
            explanation = get_safe_diagnosis_explanation(data_package)
            
            if "DETECTED_RISK" in explanation:
                st.error(explanation)
            elif "System Note" in explanation:
                st.warning(explanation)
            else:
                st.info(explanation)




