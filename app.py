import streamlit as st
import pandas as pd
import joblib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Naija Telecom Churn Predictor",
    page_icon="📶",
    layout="wide"
)

# ============================================
# CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1000px;}
    h1 {font-weight: 700;}
    .stButton>button {
        background-color: #2C8C5C;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover {
        background-color: #236B47;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL + ENCODERS
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    encoders = joblib.load("encoders.pkl")
    columns = joblib.load("columns.pkl")
    return model, encoders, columns

model, encoders, columns = load_artifacts()

# ============================================
# MAPPINGS
# ============================================
contract_map = {
    "Prepaid": "Month-to-month",
    "Postpaid - 1 year": "One year",
    "Postpaid - 2 year": "Two year"
}
internet_map = {
    "No mobile data": "No",
    "3G / DSL-equivalent": "DSL",
    "4G / Fiber-equivalent": "Fiber optic"
}
payment_map = {
    "USSD / Bank Transfer": "Bank transfer (automatic)",
    "Card Payment": "Credit card (automatic)",
    "Electronic Check": "Electronic check",
    "Mailed Check": "Mailed check"
}

# Sample profiles for one-click testing (great for demo video too)
SAMPLES = {
    "High-risk prepaid user": dict(
        gender="Female", senior="No", partner="No", dependents="No", tenure=3,
        phone_service="Yes", multiple_lines="No", plan_type="Prepaid",
        network_type="4G / Fiber-equivalent", online_security="No", online_backup="No",
        device_protection="No", tech_support="No", streaming_tv="Yes", streaming_movies="Yes",
        paperless_billing="Yes", payment_method="Electronic Check",
        monthly_charges=85.0, total_charges=250.0
    ),
    "Loyal postpaid user": dict(
        gender="Male", senior="No", partner="Yes", dependents="Yes", tenure=60,
        phone_service="Yes", multiple_lines="Yes", plan_type="Postpaid - 2 year",
        network_type="3G / DSL-equivalent", online_security="Yes", online_backup="Yes",
        device_protection="Yes", tech_support="Yes", streaming_tv="Yes", streaming_movies="Yes",
        paperless_billing="No", payment_method="USSD / Bank Transfer",
        monthly_charges=45.0, total_charges=2700.0
    )
}

# ============================================
# INITIALIZE SESSION STATE (for sample-fill buttons)
# ============================================
defaults = dict(
    gender="Male", senior="No", partner="Yes", dependents="No", tenure=12,
    phone_service="Yes", multiple_lines="No", plan_type="Prepaid",
    network_type="4G / Fiber-equivalent", online_security="No", online_backup="No",
    device_protection="No", tech_support="No", streaming_tv="No", streaming_movies="No",
    paperless_billing="Yes", payment_method="USSD / Bank Transfer",
    monthly_charges=25.0, total_charges=300.0
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_sample(name):
    for k, v in SAMPLES[name].items():
        st.session_state[k] = v

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 📶 About this tool")
    st.write(
        "Predicts churn risk for Nigerian mobile subscribers (MTN, Glo, Airtel, 9mobile) "
        "using a model trained on real-world telecom churn patterns."
    )
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**ROC-AUC:** 0.839")
    st.markdown("**Recall (churners caught):** 80%")
    st.markdown("---")
    st.markdown("### ⚡ Try a sample profile")
    for name in SAMPLES:
        st.button(name, on_click=load_sample, args=(name,), use_container_width=True)
    st.markdown("---")
    st.caption("3MTT NextGen Capstone Project · Built with Streamlit")

# ============================================
# HEADER
# ============================================
st.title("📶 Telecom Churn Predictor")
st.write("Enter a subscriber's details below to predict their risk of switching networks.")
st.markdown("")

# ============================================
# FORM — grouped into tabs
# ============================================
tab1, tab2, tab3 = st.tabs(["👤 Subscriber Profile", "📡 Plan & Services", "💰 Spend"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        partner = st.selectbox("Has Partner", ["Yes", "No"], key="partner")
        tenure = st.slider("Tenure with network (months)", 0, 72, key="tenure",
                            help="How long the subscriber has been with the network")
    with c2:
        senior = st.selectbox("Senior Citizen (60+)", ["No", "Yes"], key="senior")
        dependents = st.selectbox("Has Dependents", ["Yes", "No"], key="dependents")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone_service")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        plan_type = st.selectbox("Plan Type", list(contract_map.keys()), key="plan_type",
                                  help="Prepaid = no contract lock-in, easiest to churn")
        network_type = st.selectbox("Network / Data Type", list(internet_map.keys()), key="network_type")
        multiple_lines = st.selectbox("Multiple SIM Lines", ["No", "Yes", "No phone service"], key="multiple_lines")
        online_security = st.selectbox("Data/Account Security Add-on", ["No", "Yes", "No internet service"], key="online_security")
        online_backup = st.selectbox("Cloud Backup Add-on", ["No", "Yes", "No internet service"], key="online_backup")
    with c2:
        tech_support = st.selectbox("Customer Care Support Plan", ["No", "Yes", "No internet service"], key="tech_support")
        device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"], key="device_protection")
        streaming_tv = st.selectbox("Streaming TV Bundle", ["No", "Yes", "No internet service"], key="streaming_tv")
        streaming_movies = st.selectbox("Streaming Movies Bundle", ["No", "Yes", "No internet service"], key="streaming_movies")
        payment_method = st.selectbox("Payment Channel", list(payment_map.keys()), key="payment_method")
        paperless_billing = st.selectbox("Paperless/Digital Billing", ["Yes", "No"], key="paperless_billing")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        monthly_charges = st.number_input("Average Monthly Spend (₦, in thousands)",
                                           min_value=0.0, max_value=100.0, step=0.5, key="monthly_charges")
    with c2:
        total_charges = st.number_input("Total Lifetime Spend (₦, in thousands)",
                                         min_value=0.0, max_value=5000.0, step=10.0, key="total_charges")

st.markdown("")
predict_clicked = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

# ============================================
# PREDICT
# ============================================
if predict_clicked:
    input_data = {
        'gender': st.session_state.gender,
        'SeniorCitizen': 1 if st.session_state.senior == "Yes" else 0,
        'Partner': st.session_state.partner,
        'Dependents': st.session_state.dependents,
        'tenure': st.session_state.tenure,
        'PhoneService': st.session_state.phone_service,
        'MultipleLines': st.session_state.multiple_lines,
        'InternetService': internet_map[st.session_state.network_type],
        'OnlineSecurity': st.session_state.online_security,
        'OnlineBackup': st.session_state.online_backup,
        'DeviceProtection': st.session_state.device_protection,
        'TechSupport': st.session_state.tech_support,
        'StreamingTV': st.session_state.streaming_tv,
        'StreamingMovies': st.session_state.streaming_movies,
        'Contract': contract_map[st.session_state.plan_type],
        'PaperlessBilling': st.session_state.paperless_billing,
        'PaymentMethod': payment_map[st.session_state.payment_method],
        'MonthlyCharges': st.session_state.monthly_charges,
        'TotalCharges': st.session_state.total_charges
    }

    input_df = pd.DataFrame([input_data])
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Result")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Churn Probability", f"{probability:.0%}",
                   delta="High Risk" if prediction == 1 else "Low Risk",
                   delta_color="inverse" if prediction == 1 else "normal")
    with r2:
        st.progress(float(probability))
        if prediction == 1:
            st.error("⚠️ **High churn risk.** This subscriber is likely to switch networks soon.")
            st.markdown("**Suggested action:** Offer bonus data, a loyalty discount, or migrate them to a postpaid plan to reduce switching risk.")
        else:
            st.success("✅ **Low churn risk.** This subscriber is likely to stay.")
            st.markdown("**Suggested action:** No urgent intervention needed — continue standard engagement.")

st.markdown("---")
st.caption("Model trained on the IBM Telco Customer Churn dataset, adapted to reflect Nigerian mobile network usage patterns. Built as a 3MTT NextGen Capstone Project.")
