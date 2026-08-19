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
    .field-error {
        color: #D93025;
        font-size: 12.5px;
        margin-top: -12px;
        margin-bottom: 8px;
        font-weight: 600;
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

PLACEHOLDER = "-- Select --"

def sel_options(opts):
    return [PLACEHOLDER] + opts

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

FIELD_KEYS = [
    "gender", "senior", "partner", "dependents", "tenure", "phone_service",
    "multiple_lines", "plan_type", "network_type", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "paperless_billing", "payment_method", "monthly_charges", "total_charges"
]
for k in FIELD_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

if "submit_attempted" not in st.session_state:
    st.session_state.submit_attempted = False

def load_sample(name):
    for k, v in SAMPLES[name].items():
        st.session_state[k] = v
    st.session_state.submit_attempted = False

def clear_form():
    for k in FIELD_KEYS:
        st.session_state[k] = None
    st.session_state.submit_attempted = False

def is_empty(key):
    return st.session_state[key] in (None, PLACEHOLDER)

def error_note(key):
    """Show a red 'required' note under a field if it's empty AND the user already tried to submit."""
    if st.session_state.submit_attempted and is_empty(key):
        st.markdown("<div class='field-error'>⚠️ This field is required</div>", unsafe_allow_html=True)

# ============================================
# HELPER WIDGETS (wrap + auto-show error note)
# ============================================
def field_select(label, options, key, help=None):
    st.selectbox(label, sel_options(options), key=key, help=help)
    error_note(key)

def field_number(label, key, min_value, max_value, step, placeholder, help=None):
    st.number_input(label, min_value=min_value, max_value=max_value, value=None,
                     step=step, placeholder=placeholder, key=key, help=help)
    error_note(key)

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
    st.caption("Auto-fills every field for a quick demo")
    for name in SAMPLES:
        st.button(name, on_click=load_sample, args=(name,), use_container_width=True)
    st.button("🧹 Clear form", on_click=clear_form, use_container_width=True)
    st.markdown("---")
    st.caption("3MTT NextGen Capstone Project · Built with Streamlit")

# ============================================
# HEADER
# ============================================
st.title("📶 Telecom Churn Predictor")
st.write("Fill in every field below, then click **Predict Churn Risk**. All fields are required.")

missing_count = sum(1 for k in FIELD_KEYS if is_empty(k))
if st.session_state.submit_attempted and missing_count > 0:
    st.error(f"⚠️ {missing_count} field(s) still empty — check the fields marked in red below.")

st.markdown("")

# ============================================
# FORM — grouped into tabs
# ============================================
tab1, tab2, tab3 = st.tabs(["👤 Subscriber Profile", "📡 Plan & Services", "💰 Spend"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        field_select("Gender", ["Male", "Female"], "gender")
        field_select("Has Partner", ["Yes", "No"], "partner")
        field_number("Tenure with network (months)", "tenure", 0, 72, 1, "Enter number of months",
                     help="How long the subscriber has been with the network")
    with c2:
        field_select("Senior Citizen (60+)", ["No", "Yes"], "senior")
        field_select("Has Dependents", ["Yes", "No"], "dependents")
        field_select("Phone Service", ["Yes", "No"], "phone_service")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        field_select("Plan Type", list(contract_map.keys()), "plan_type",
                     help="Prepaid = no contract lock-in, easiest to churn")
        field_select("Network / Data Type", list(internet_map.keys()), "network_type")
        field_select("Multiple SIM Lines", ["No", "Yes", "No phone service"], "multiple_lines")
        field_select("Data/Account Security Add-on", ["No", "Yes", "No internet service"], "online_security")
        field_select("Cloud Backup Add-on", ["No", "Yes", "No internet service"], "online_backup")
    with c2:
        field_select("Customer Care Support Plan", ["No", "Yes", "No internet service"], "tech_support")
        field_select("Device Protection Plan", ["No", "Yes", "No internet service"], "device_protection")
        field_select("Streaming TV Bundle", ["No", "Yes", "No internet service"], "streaming_tv")
        field_select("Streaming Movies Bundle", ["No", "Yes", "No internet service"], "streaming_movies")
        field_select("Payment Channel", list(payment_map.keys()), "payment_method")
        field_select("Paperless/Digital Billing", ["Yes", "No"], "paperless_billing")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        field_number("Average Monthly Spend (₦, in thousands)", "monthly_charges",
                     0.0, 100.0, 0.5, "e.g. 25.0")
    with c2:
        field_number("Total Lifetime Spend (₦, in thousands)", "total_charges",
                     0.0, 5000.0, 10.0, "e.g. 300.0")

st.markdown("")
predict_clicked = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

# ============================================
# VALIDATE + PREDICT
# ============================================
if predict_clicked:
    st.session_state.submit_attempted = True
    missing = [k for k in FIELD_KEYS if is_empty(k)]

    if missing:
        st.rerun()  # rerun so red notes appear under each empty field immediately

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
