# 📶 Telecom Churn Predictor

A machine learning web app that predicts customer churn risk for telecom subscribers, adapted to reflect the **Nigerian mobile market** (MTN, Glo, Airtel, 9mobile).

Built as a capstone project for the **3MTT NextGen Fellowship Program**.

**🔗 Live App:** [Add your Streamlit link here]
**🎥 Demo Video:** [Add your video link here]

---

## Problem Statement

Nigerian telecom operators face persistent subscriber churn driven by:
- **SIM switching** — most Nigerians are prepaid, with no contract lock-in, making it easy to switch networks
- **Network quality complaints** — dropped calls and poor 4G coverage push users to competitors
- **Price sensitivity** — sharp reactions to data/call tariff increases
- **Poor customer support experience** — unresolved complaints accelerate switching

This app helps identify subscribers at high risk of churning *before* they leave, so telcos can proactively intervene (bonus data, loyalty discounts, plan upgrades) rather than losing the customer entirely.

---

## Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Size:** 7,043 customers, 21 features
- **Why this dataset:** Clean, well-documented, and its churn patterns map naturally onto the Nigerian market — e.g. "Month-to-month" contracts (highest churn in the original data) directly parallel Nigeria's dominant prepaid model.

**Nigerian-context adaptation:** Since Nigeria-specific public churn datasets aren't available, the original dataset's labels were reframed at the UI layer to reflect local terminology (e.g. "Month-to-month" → "Prepaid", "Fiber optic" → "4G/Fiber-equivalent network", charges shown in ₦). The underlying model and data remain unchanged — only how information is presented and collected from the user was localized.

---

## Model

Two models were trained and compared:

| Model | Accuracy | ROC-AUC | Recall (Churners) |
|---|---|---|---|
| **Logistic Regression** ✅ (used) | 74% | **0.839** | **80%** |
| Random Forest | 79% | 0.824 | 48% |

**Why Logistic Regression was chosen over Random Forest:**
Although Random Forest has higher raw accuracy, it misses more than half of actual churners (48% recall). For a churn predictor, failing to flag a customer who is about to leave is the costliest type of error — the entire value of the tool is catching at-risk customers early. Logistic Regression catches 80% of churners and has a higher ROC-AUC, making it the better model for this business problem despite its lower headline accuracy.

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Logistic Regression), pandas
- **App/UI:** Streamlit
- **Hosting:** Streamlit Community Cloud (free)
- **Code hosting:** GitHub
- **Development environment:** Google Colab

---

## How It Works

1. User opens the app and fills in subscriber details (demographics, plan type, service add-ons, spend) via an interactive form
2. On clicking **"Predict Churn Risk"**, inputs are encoded using the same encoders used during training
3. The trained Logistic Regression model predicts churn probability
4. The app displays a **High Risk** or **Low Risk** result with the probability and a suggested retention action

---

## Project Structure

```
telecom-churn-predictor/
├── app.py              # Streamlit app (UI + inference)
├── requirements.txt    # Python dependencies
├── churn_model.pkl     # Trained Logistic Regression model
├── encoders.pkl        # Label encoders for categorical features
├── columns.pkl         # Feature column order (for consistent inference)
└── README.md           # This file
```

---

## Run Locally

```bash
git clone https://github.com/yourusername/telecom-churn-predictor.git
cd telecom-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Author

**Daborn**
3MTT NextGen Fellowship — Capstone Project
