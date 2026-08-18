import streamlit as st
import pandas as pd
import joblib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Naija Telecom Churn Predictor",
    page_icon="📶",
    layout="centered"
)

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
# HEADER
# ============================================
st.title("📶 Telecom Churn Predictor")
st.caption("Predict which mobile subscribers are at risk of switching networks — built for the Nigerian telecom market (MTN, Glo, Airtel, 9mobile)")

st.markdown("---")

# ============================================
# MAPPINGS: Nigerian labels -> original dataset labels
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

yes_no_map = {"Yes": "Yes", "No": "No"}

# ============================================
# FORM
# ============================================
st.subheader("Subscriber Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen (60+)", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure with network (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple SIM Lines", ["No", "Yes", "No phone service"])
    plan_type = st.selectbox("Plan Type", list(contract_map.keys()))
    network_type = st.selectbox("Network / Data Type", list(internet_map.keys()))

with col2:
    online_security = st.selectbox("Data/Account Security Add-on", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Cloud Backup Add-on", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Customer Care Support Plan", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV Bundle", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies Bundle", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless/Digital Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Channel", list(payment_map.keys()))

st.markdown("---")
st.subheader("Spend")

col3, col4 = st.columns(2)
with col3:
    monthly_charges = st.number_input("Average Monthly Spend (₦, in thousands)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
with col4:
    total_charges = st.number_input("Total Lifetime Spend (₦, in thousands)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0)

st.markdown("---")

# ============================================
# PREDICT
# ============================================
if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):

    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_map[network_type],
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract_map[plan_type],
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_map[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Apply saved encoders (never re-fit — always transform)
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    # Match training column order
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.error(f"⚠️ HIGH CHURN RISK — {probability:.0%} likelihood this subscriber switches networks")
        st.markdown("**Suggested action:** Proactively offer bonus data, loyalty discount, or migrate to a postpaid plan to reduce switching risk.")
    else:
        st.success(f"✅ LOW CHURN RISK — {probability:.0%} likelihood this subscriber switches networks")
        st.markdown("**Suggested action:** No urgent intervention needed. Continue standard engagement.")

    st.progress(float(probability))

st.markdown("---")
st.caption("Model trained on the IBM Telco Customer Churn dataset, adapted to reflect Nigerian mobile network usage patterns (prepaid/postpaid plans, network type, and spend in ₦). Built as a 3MTT NextGen Capstone Project.")
