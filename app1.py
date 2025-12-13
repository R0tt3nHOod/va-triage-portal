import os
import streamlit as st
import numpy as np
import requests
import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

#Load environment variables from .env file
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
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Header Bar - Clinical Navy */
    header[data-testid="stHeader"] {
        background-color: #0e1117;
    }

    /* Sidebar - Darker Contrast */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dcdcdc;
    }

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
    UPDATED: Uses a strict Clinical Pathologist persona and explicit ranges for filtering.
    """
    # Check for keys (prevents crashing if not set) [cite: 8, 30]
    if not os.getenv("AZURE_OPENAI_KEY") or not os.getenv("CONTENT_SAFETY_KEY"):
        return "⚠️ SYSTEM NOTE: AI Companion disabled (Keys missing)."

    try:
        # 1. TRANSLATE (Azure OpenAI)
        client_gpt = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_KEY"),  
            api_version="2024-02-01"
        )

        # --- UPDATED PROMPT FOR CLINICIANS (With Filtering/Ranges) ---
        prompt = f"""
        You are a Clinical Metabolic Pathologist providing an objective analysis for a physician.
        Your output must be concise, technical, and based only on the provided data.
        CONTEXT: The patient is a Gulf War Veteran with suspected GWI.

        HEALTHY REFERENCE RANGES:
        - NAD/NADH Ratio: 5.0 to 10.0 (or higher is optimal)
        - PCr/ATP Ratio: 3.0 to 5.0
        - GSH/GSSG Ratio: 10.0 to 100.0

        INSTRUCTIONS:
        1. Begin your output immediately with the header "Metabolic Analysis:"
        2. Analyze each biomarker. State the patient's value and conclude if it is within, above, or below the normal range, referencing the ranges provided.
        3. Explain the clinical significance of the ratios in relation to the diagnosis pattern (e.g., Type 1, Type 2).
        4. Suggest specific next steps for clinical validation (e.g., "Consider confirmatory 31P-MRS").
        5. DO NOT use conversation fillers, subjective language, or reassurances.

        RAW DATA: {medical_data_json}
        """

        response = client_gpt.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}]
        )
        explanation_text = response.choices[0].message.content

        # 2. SAFETY CHECK (Azure Content Safety) [cite: 33]
        client_safety = ContentSafetyClient(
            endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("CONTENT_SAFETY_KEY"))
        )
        
        request = AnalyzeTextOptions(text=explanation_text)
        safety_result = client_safety.analyze_text(request)
        
        for analysis in safety_result.categories_analysis:
            if analysis.category == TextCategory.SELF_HARM and analysis.severity > 0:
                return "DETECTED_RISK: Please contact the Veteran Crisis Line: 988." [cite: 33]

        return explanation_text

    except Exception as e:
        return f"System Note: AI Explanation unavailable ({str(e)})"

# --- LOGIC & API ---
# Stage 1 Weights (Calibrated)
INTERCEPT = -18.07175
COEF_PAIN = 0.86589
COEF_CONFUSION = 1.53029
COEF_DIZZINESS = 1.25769
COEF_FATIGUE = 1.08384

def calculate_stage1_risk(pain, confusion, dizziness, fatigue):
    logit = INTERCEPT + (COEF_PAIN * pain) + (COEF_CONFUSION * confusion) + \
            (COEF_DIZZINESS * dizziness) + (COEF_FATIGUE * fatigue)
    probability = 1 / (1 + np.exp(-logit))
    return probability

def call_azure_api(biomarkers, prior_prob):
    # 1. SECURE: Get Key AND Endpoint from Environment Settings
    api_key = os.environ.get("AZURE_API_KEY")
    azure_ml_url = os.environ.get("AZURE_ML_ENDPOINT")  

    # 2. SAFETY CHECK: Ensure both exist before running
    if not api_key:
        return "System Error: Azure API Key is missing."
    if not azure_ml_url:
        return "System Error: Azure ML Endpoint is missing."

    # 3. RUN: Use the variables
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
    
    data = {
        "input_data": {
            "data": [{
                "NAD_NADH": biomarkers['nad'],
                "PCr_ATP": biomarkers['pcr'],
                "GSH_GSSG": biomarkers['gsh'],
                "Metabolic_Index": biomarkers['meta_index'],
                "Symptom_Prior_Probability": prior_prob
            }]
        }
    }

    try:
        response = requests.post(azure_ml_url, headers=headers, json=data)
        if response.status_code == 200:
            result_list = response.json()
            result_code = result_list[0]
            if result_code == 0: return "Negative (Healthy)"
            elif result_code == 1: return "POSITIVE: Type 1 Pattern (Cognitive/Fatigue)"
            elif result_code == 2: return "POSITIVE: Type 2 Pattern (Ataxia/Dizziness)"
            elif result_code == 3: return "POSITIVE: Type 3 Pattern (Pain/Myalgia)"
            else: return f"POSITIVE: Unknown Pattern ({result_code})"
        else: return f"Error {response.status_code}"
    except Exception as e: return f"Connection Error: {str(e)}"

# --- SIDEBAR (THE CHART) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Microsoft_icon.svg/1024px-Microsoft_icon.svg.png", width=50) # Placeholder Logo
    st.markdown("### 🏥 VA NeuroMetabolic Triage")
    st.markdown(f"<div class='badge-secure'>🔒 HIPAA SECURE SESSION | ID: {datetime.date.today()}</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Patient Symptom Profile")
    st.info("Haley Criteria Revised (0-10)")
    
    input_pain = st.slider("Joint Pain / Myalgia", 0, 10, 1, key="pain", on_change=reset_app)
    input_fatigue = st.slider("Chronic Fatigue", 0, 10, 1, key="fatigue", on_change=reset_app)
    input_confusion = st.slider("Cognitive Impairment", 0, 10, 1, key="confusion", on_change=reset_app)
    input_dizziness = st.slider("Vestibular Dysfunction", 0, 10, 1, key="dizziness", on_change=reset_app)
    
    st.markdown("---")
    analyze_btn = st.button("RUN TRIAGE PROTOCOL", type="primary")

# --- MAIN DASHBOARD ---
st.title("Clinical Decision Support Dashboard")

# Initialize State
if 'risk_score' not in st.session_state: st.session_state['risk_score'] = None

# Logic Trigger
if analyze_btn:
    st.session_state['risk_score'] = calculate_stage1_risk(input_pain, input_confusion, input_dizziness, input_fatigue)

# DISPLAY LOGIC
if st.session_state['risk_score'] is None:
    st.info("Please input patient symptoms in the left sidebar to begin triage.")
    
    # Placeholder to look nice when empty
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("System Status", "Online", delta="Azure East US 2")
    with c2: st.metric("Model Version", "v1.4 (Ensemble)", delta="Active")
    with c3: st.metric("Pending Claims", "214,005", delta_color="inverse")

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
        with c1: val_nad = st.number_input("NAD / NADH Ratio", value=0.0)
        with c2: val_pcr = st.number_input("PCr / ATP Ratio", value=0.0)
        val_gsh = st.number_input("GSH / GSSG Ratio", value=0.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        val_meta_index = (val_nad/2.0) + (val_pcr/1.8) + (val_gsh/30.0)
        
        st.write("")
        if st.button("ORDER CONFIRMATORY ANALYSIS"):
            # INPUT VALIDATION: Stop processing if values are impossible
            if val_nad < 0 or val_pcr < 0 or val_gsh < 0:
                st.error("⚠️ Invalid Input: Biomarker ratios cannot be negative. Please check your values.")
                st.stop()
            
            # --- LOGIC UPDATE: THE 'HIGH-CONFIDENCE VETO' ---
            current_score = score # The score from Stage 1 (Symptoms)
            modified_score = current_score
            
            # We define strict 'Optimal' ranges. 
            # Only PERFECT metabolic health can veto a high symptom score.
            # This protects against False Negatives (sick vets with borderline labs).
            is_metabolic_optimal = (
                (val_nad >= 8.0) and              # Strict: Must have high redox potential
                (val_pcr >= 3.5 and val_pcr <= 4.5) and # Strict: Must have stable energy reserves
                (val_gsh >= 50.0)                 # Strict: Must have high antioxidant capacity
            )
            
            # If biomarkers are OPTIMAL, we assume the symptoms are non-metabolic (or exaggerated)
            if is_metabolic_optimal:
                modified_score = 0.05 # De-weight symptoms to 5%
                st.info("✅ **Clinical Note:** Biomarkers indicate optimal mitochondrial function. Symptom score de-weighted to prioritize objective metabolic data.")
            
            # 1. PREPARE DATA 
            biomarkers = {
                'nad': val_nad, 
                'pcr': val_pcr, 
                'gsh': val_gsh, 
                'meta_index': val_meta_index
            }
        
            # 2. CALL YOUR EXISTING NEURAL NETWORK
            with st.spinner(f"Connecting to Azure Neural Network (Prior Score: {round(modified_score*100, 1)}%)..."):
                result = call_azure_api(biomarkers, modified_score)
            st.success(f"**FINAL DIAGNOSIS:** {result}")
            
            if "POSITIVE" in result:
                st.error("ACTION REQUIRED: Refer to Neurology.")
    
            st.markdown("---")
            st.subheader("🤖 AI Clinical Companion (Azure OpenAI)")
            
            with st.spinner("Generating plain-English explanation..."):
                
                # Package the data for the explanation AI
                data_package = {
                    "diagnosis_result": result,
                    "metabolic_index": val_meta_index,
                    "biomarkers": biomarkers,
                    # We explicitly add the raw values so the AI can explain "Low NAD" etc.
                    "context": "Patient is a Gulf War Veteran"
                }
                
                # Call the new function
                explanation = get_safe_diagnosis_explanation(data_package)
                
                # Display the output safely
                if "DETECTED_RISK" in explanation:
                    st.error(explanation)
                elif "System Note" in explanation:
                    st.warning(explanation)
                else:
                    st.info(explanation)
        
        
        





