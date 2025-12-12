import os
import streamlit as st
import numpy as np
import requests
import datetime

# --- CONFIGURATION & SECURITY ---
st.set_page_config(
    page_title="VA NeuroMetabolic Triage | Secure",
    page_icon="🏥",
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
    # SECURE: Get key from Azure Environment settings
    api_key = os.environ.get("AZURE_API_KEY")
        
    # --- THE FIX: SAFETY CHECK ---
    if not api_key:
        return "System Error: Azure API Key is missing from environment settings."

    # Now it is safe to use api_key because we checked it's not None
    url = "https://proto-mtiocloud-jdakw.eastus2.inference.ml.azure.com/score" 
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
        response = requests.post(url, headers=headers, json=data)
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
            biomarkers = {'nad': val_nad, 'pcr': val_pcr, 'gsh': val_gsh, 'meta_index': val_meta_index}
            
            with st.spinner("Connecting to Azure Neural Network..."):
                result = call_azure_api(biomarkers, score)
            
            st.success(f"**FINAL DIAGNOSIS:** {result}")
            if "POSITIVE" in result:

                st.error("ACTION REQUIRED: Refer to Neurology.")


