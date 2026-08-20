import streamlit as st
import pandas as pd
import numpy as np
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
    .main .block-container {padding-top: 2rem; max-width: 1100px;}
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

# Known evaluation metrics from training (Logistic Regression, held-out test set, n=1409)
METRICS = {
    "accuracy": 0.74,
    "roc_auc": 0.839,
    "class0": {"label": "Will Stay", "precision": 0.91, "recall": 0.71, "f1": 0.80, "support": 1035},
    "class1": {"label": "Will Churn", "precision": 0.50, "recall": 0.80, "f1": 0.62, "support": 374},
}

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
FRIENDLY_FEATURE_NAMES = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple SIM Lines",
    "InternetService": "Network / Data Type",
    "OnlineSecurity": "Account Security Add-on",
    "OnlineBackup": "Cloud Backup Add-on",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Customer Care Support",
    "StreamingTV": "Streaming TV Bundle",
    "StreamingMovies": "Streaming Movies Bundle",
    "Contract": "Plan Type (Prepaid/Postpaid)",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Channel",
    "MonthlyCharges": "Monthly Spend",
    "TotalCharges": "Total Lifetime Spend"
}

NETWORK_PROVIDERS = ["MTN", "Glo", "Airtel", "9mobile", "Other / Not sure"]

def naira(amount):
    """Format a number as a Naira amount with comma separators."""
    return f"₦{amount:,.0f}"

def get_retention_suggestion(raw_input: dict, monthly_naira: float, tenure_months: int) -> str:
    """
    Return a tailored retention suggestion based on which risk drivers are present,
    instead of one generic message for every high-risk case.
    """
    is_prepaid = raw_input.get("Contract") == "Month-to-month"
    no_support = raw_input.get("TechSupport") == "No"
    no_security = raw_input.get("OnlineSecurity") == "No"
    is_new = tenure_months < 12
    high_spender = monthly_naira > 50000
    paperless = raw_input.get("PaperlessBilling") == "Yes"

    suggestions = []

    if is_prepaid and is_new:
        suggestions.append(
            "This is a new, prepaid subscriber — the highest-risk combination. "
            "Offer a discounted postpaid migration or a loyalty data bundle within their first 90 days."
        )
    elif is_prepaid and high_spender:
        suggestions.append(
            "High-spending prepaid subscriber — a strong candidate for a postpaid upgrade with better rates, "
            "since they already spend enough to benefit from a bundled plan."
        )
    elif is_prepaid:
        suggestions.append(
            "Prepaid subscriber with no contract lock-in — offer bonus data or a short-term loyalty discount to build commitment."
        )

    if no_support:
        suggestions.append(
            "No active customer care support plan — proactively reach out to check on service satisfaction before they consider switching."
        )

    if no_security:
        suggestions.append(
            "No account security add-on — bundling this in at low/no cost can increase perceived value without a big discount."
        )

    if not suggestions:
        suggestions.append(
            "Monitor this subscriber and consider a light-touch loyalty offer, e.g. bonus data for their next renewal."
        )

    return " ".join(suggestions[:2])  # keep it to at most 2 combined points


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
        monthly_charges=85000.0, total_charges=250000.0
    ),
    "Loyal postpaid user": dict(
        gender="Male", senior="No", partner="Yes", dependents="Yes", tenure=60,
        phone_service="Yes", multiple_lines="Yes", plan_type="Postpaid - 2 year",
        network_type="3G / DSL-equivalent", online_security="Yes", online_backup="Yes",
        device_protection="Yes", tech_support="Yes", streaming_tv="Yes", streaming_movies="Yes",
        paperless_billing="No", payment_method="USSD / Bank Transfer",
        monthly_charges=45000.0, total_charges=2700000.0
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

if "network_provider" not in st.session_state:
    st.session_state.network_provider = "Not specified"

if "submit_attempted" not in st.session_state:
    st.session_state.submit_attempted = False

def load_sample(name):
    for k, v in SAMPLES[name].items():
        st.session_state[k] = v
    st.session_state.submit_attempted = False

def clear_form():
    for k in FIELD_KEYS:
        st.session_state[k] = None
    st.session_state.network_provider = "Not specified"
    st.session_state.submit_attempted = False

def is_empty(key):
    return st.session_state[key] in (None, PLACEHOLDER)

def error_note(key):
    if st.session_state.submit_attempted and is_empty(key):
        st.markdown("<div class='field-error'>⚠️ This field is required</div>", unsafe_allow_html=True)

def field_select(label, options, key, help=None):
    st.selectbox(label, sel_options(options), key=key, help=help)
    error_note(key)

def field_number(label, key, min_value, max_value, step, placeholder, help=None, note=None):
    st.number_input(label, min_value=min_value, max_value=max_value, value=None,
                     step=step, placeholder=placeholder, key=key, help=help)
    if note:
        st.caption(note)
    error_note(key)

def build_input_row(vals: dict) -> pd.DataFrame:
    """Takes a dict of raw (already dataset-label) values and returns an encoded, column-ordered row."""
    row = pd.DataFrame([vals])
    for col in encoders:
        if col in row.columns:
            row[col] = encoders[col].transform(row[col])
    return row[columns]

def get_feature_importance(top_n=8):
    """Global feature importance from Logistic Regression coefficients."""
    if not hasattr(model, "coef_"):
        return None
    coefs = model.coef_[0]
    fi = pd.DataFrame({
        "feature": [FRIENDLY_FEATURE_NAMES.get(c, c) for c in columns],
        "impact": coefs
    })
    fi["abs_impact"] = fi["impact"].abs()
    fi = fi.sort_values("abs_impact", ascending=False).head(top_n)
    fi["direction"] = fi["impact"].apply(lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk")
    return fi.sort_values("abs_impact")

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
    st.markdown(f"**ROC-AUC:** {METRICS['roc_auc']}")
    st.markdown(f"**Recall (churners caught):** {METRICS['class1']['recall']:.0%}")
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
st.write("Predict churn risk for one subscriber, upload a batch of subscribers, or explore how the model makes decisions.")
st.markdown("")

main_tab1, main_tab2, main_tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction (CSV)", "📊 Model Insights"])

# ============================================
# TAB 1: SINGLE PREDICTION
# ============================================
with main_tab1:
    missing_count = sum(1 for k in FIELD_KEYS if is_empty(k))
    if st.session_state.submit_attempted and missing_count > 0:
        st.error(f"⚠️ {missing_count} field(s) still empty — check the fields marked in red below.")

    st.write("Fill in every field below, then click **Predict Churn Risk**. All fields are required.")

    tab1, tab2, tab3 = st.tabs(["👤 Subscriber Profile", "📡 Plan & Services", "💰 Spend"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            field_select("Gender", ["Male", "Female"], "gender")
            field_select("Has Partner", ["Yes", "No"], "partner")
            field_number("Tenure with network (months)", "tenure", 0, 72, 1, "Enter number of months",
                         help="How long the subscriber has been with the network")
            st.selectbox("Network Provider (optional)", ["Not specified"] + NETWORK_PROVIDERS,
                        key="network_provider",
                        help="For labeling/record-keeping only — does not affect the prediction.")
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
        st.info(
            "📊 **Why these caps?** The model was trained on the IBM Telco Customer Churn dataset, "
            "where Monthly Charges ranged **$18.25–$118.75** and Total Charges ranged **$18.80–$8,684.80** "
            "(scaled ×1000 here to represent Naira). Input limits are capped at this range because a Logistic "
            "Regression model extrapolates unreliably beyond the data it was trained on."
        )
        c1, c2 = st.columns(2)
        with c1:
            field_number("Average Monthly Spend (₦)", "monthly_charges",
                         0.0, 120000.0, 500.0, "e.g. 25000",
                         note="Capped at ₦120,000 — matches the training data's maximum monthly charge ($118.75 × 1000).")
        with c2:
            field_number("Total Lifetime Spend (₦)", "total_charges",
                         0.0, 8700000.0, 5000.0, "e.g. 300000",
                         note="Capped at ₦8,700,000 — matches the training data's maximum total charge ($8,684.80 × 1000).")

    st.markdown("")
    predict_clicked = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

    if predict_clicked:
        st.session_state.submit_attempted = True
        missing = [k for k in FIELD_KEYS if is_empty(k)]

        if missing:
            st.rerun()

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
            'MonthlyCharges': st.session_state.monthly_charges / 1000,
            'TotalCharges': st.session_state.total_charges / 1000
        }

        try:
            input_df = build_input_row(input_data)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.subheader("Result")

            monthly_naira = st.session_state.monthly_charges
            annual_revenue_at_risk = monthly_naira * 12
            tenure_months = st.session_state.tenure

            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Churn Probability", f"{probability:.0%}",
                           delta="High Risk" if prediction == 1 else "Low Risk",
                           delta_color="inverse" if prediction == 1 else "normal")
                if st.session_state.network_provider != "Not specified":
                    st.caption(f"📡 Network: {st.session_state.network_provider}")
            with r2:
                st.progress(float(probability))
                if prediction == 1:
                    st.error("⚠️ **High churn risk.** This subscriber is likely to switch networks soon.")
                    st.markdown(
                        f"**💰 Estimated revenue at risk:** {naira(annual_revenue_at_risk)}/year "
                        f"(based on {naira(monthly_naira)}/month spend, if this subscriber churns)"
                    )
                    suggestion = get_retention_suggestion(input_data, monthly_naira, tenure_months)
                    st.markdown(f"**Suggested action:** {suggestion}")
                else:
                    st.success("✅ **Low churn risk.** This subscriber is likely to stay.")
                    st.markdown("**Suggested action:** No urgent intervention needed — continue standard engagement.")

            # ---- Why this prediction (top global drivers, for context) ----
            with st.expander("💡 What typically drives this prediction?"):
                fi = get_feature_importance(top_n=6)
                if fi is not None:
                    st.caption("These are the factors that most influence churn risk across all subscribers in the model (not just this one).")
                    st.bar_chart(fi.set_index("feature")["impact"])
                    st.caption("Positive bars increase churn risk; negative bars decrease it.")
        except Exception as e:
            st.error(f"Something went wrong while generating the prediction: {e}")

# ============================================
# TAB 2: BATCH PREDICTION (CSV)
# ============================================
with main_tab2:
    st.subheader("Upload a CSV of multiple subscribers")
    st.write(
        "Predict churn risk for many subscribers at once. Upload a CSV with the columns listed below "
        "(same format as the original training dataset)."
    )

    template_df = pd.DataFrame([{
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 5, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0, 'TotalCharges': 425.0
    }])

    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="churn_prediction_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_cols = [c for c in columns if c not in batch_df.columns]

            if missing_cols:
                st.error(f"⚠️ Your CSV is missing these required columns: {', '.join(missing_cols)}")
            else:
                encoded_batch = batch_df.copy()
                for col in encoders:
                    if col in encoded_batch.columns:
                        encoded_batch[col] = encoders[col].transform(encoded_batch[col])
                encoded_batch = encoded_batch[columns]

                preds = model.predict(encoded_batch)
                probs = model.predict_proba(encoded_batch)[:, 1]

                results = batch_df.copy()
                results["Churn_Probability_%"] = (probs * 100).round(1)
                results["Churn_Risk"] = np.where(preds == 1, "High Risk", "Low Risk")
                # MonthlyCharges in the template/dataset is in raw model-scale units (matches $ scale);
                # multiply by 1000 to express as Naira, consistent with the single-prediction conversion.
                results["Est_Annual_Revenue_Naira"] = (results["MonthlyCharges"] * 1000 * 12).round(0)

                st.success(f"✅ Predicted churn risk for {len(results)} subscribers.")

                high_risk_mask = results["Churn_Risk"] == "High Risk"
                high_risk_count = high_risk_mask.sum()
                total_revenue_at_risk = results.loc[high_risk_mask, "Est_Annual_Revenue_Naira"].sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Subscribers", len(results))
                m2.metric("High Risk", int(high_risk_count))
                m3.metric("High Risk %", f"{high_risk_count / len(results):.0%}")
                m4.metric("Est. Annual Revenue at Risk", naira(total_revenue_at_risk))

                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=results.to_csv(index=False),
                    file_name="churn_predictions_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")

# ============================================
# TAB 3: MODEL INSIGHTS
# ============================================
with main_tab3:
    st.subheader("Model Performance")
    st.write("Evaluated on a held-out test set of 1,409 subscribers (20% of the data), not seen during training.")

    m1, m2 = st.columns(2)
    m1.metric("Overall Accuracy", f"{METRICS['accuracy']:.0%}")
    m2.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")

    st.markdown("#### Precision, Recall & F1-score by class")
    metrics_df = pd.DataFrame({
        "Will Stay": [METRICS["class0"]["precision"], METRICS["class0"]["recall"], METRICS["class0"]["f1"]],
        "Will Churn": [METRICS["class1"]["precision"], METRICS["class1"]["recall"], METRICS["class1"]["f1"]],
    }, index=["Precision", "Recall", "F1-score"])
    st.bar_chart(metrics_df)

    st.info(
        f"**Why Logistic Regression over Random Forest?** Random Forest had higher raw accuracy (79% vs 74%), "
        f"but only caught **48%** of actual churners. Logistic Regression catches **{METRICS['class1']['recall']:.0%}** "
        f"of churners and has a higher ROC-AUC ({METRICS['roc_auc']}). In churn prediction, missing an at-risk "
        f"customer (a false negative) is more costly than a false alarm, so recall was prioritized over accuracy."
    )

    st.markdown("---")
    st.subheader("What drives churn predictions?")
    st.write("Based on the model's learned coefficients — larger bars mean a stronger influence on churn risk.")

    fi_full = get_feature_importance(top_n=10)
    if fi_full is not None:
        st.bar_chart(fi_full.set_index("feature")["impact"])
        st.caption("Positive values increase predicted churn risk; negative values decrease it. "
                   "Note: since categorical fields are label-encoded (not one-hot), these values are an approximate guide, not an exact ranking.")

st.markdown("---")
st.caption("Model trained on the IBM Telco Customer Churn dataset, adapted to reflect Nigerian mobile network usage patterns. Built as a 3MTT NextGen Capstone Project.")
