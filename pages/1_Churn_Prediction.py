import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")

st.title("🎯 Customer Churn Prediction")
st.markdown("Fill in a customer's profile to predict whether they'll churn.")


PROJECT_ROOT = Path(__file__).parent.parent

@st.cache_resource
def load_artifacts():
    """Load the trained model and preprocessor. Runs once per session."""
    model        = joblib.load(PROJECT_ROOT / "models" / "best_classifier.pkl")
    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model files not found. Run the classification notebook first to "
             "create `models/best_classifier.pkl` and `models/preprocessor.pkl`.")
    st.stop()

st.subheader("Customer profile")

# Use a form so all inputs are submitted at once (cleaner UX than re-running on every change)
with st.form("churn_form"):
    # ---- Layout: 3 columns of inputs ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender         = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner        = st.selectbox("Has Partner", ["No", "Yes"])
        dependents     = st.selectbox("Has Dependents", ["No", "Yes"])

        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone_service     = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**More Services**")
        tech_support      = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("**Billing**")
        contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method    = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    # ---- Numerical inputs at the bottom (full width) ----
    st.markdown("**Charges**")
    cc1, cc2 = st.columns(2)
    with cc1:
        monthly_charges = st.number_input("Monthly Charges ($)",
                                          min_value=0.0, max_value=200.0, value=70.0, step=5.0)
    with cc2:
        total_charges = st.number_input("Total Charges ($)",
                                        min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")


if submitted:
    # ---- Assemble all inputs into a single-row DataFrame ----
    input_data = pd.DataFrame([{
        "gender":           gender,
        "SeniorCitizen":    senior_citizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }])

    # ---- Run prediction ----
    X_processed = preprocessor.transform(input_data)
    proba       = model.predict_proba(X_processed)[0, 1]
    prediction  = int(proba >= 0.5)

    # ---- Show result ----
    st.divider()
    st.subheader("Prediction result")

    rcol1, rcol2 = st.columns([1, 2])

    with rcol1:
        if prediction == 1:
            st.error("### ⚠️ Likely to Churn")
            st.markdown(f"**Probability:** {proba:.1%}")
        else:
            st.success("### ✅ Likely to Stay")
            st.markdown(f"**Churn probability:** {proba:.1%}")

    with rcol2:
        st.markdown("**Churn probability**")
        st.progress(float(proba))

        if proba < 0.30:
            risk = "🟢 Low risk"
        elif proba < 0.60:
            risk = "🟡 Medium risk"
        else:
            risk = "🔴 High risk"
        st.markdown(f"**Risk level:** {risk}")

    # ---- Recommendation ----
    st.markdown("##### Recommendation")
    if proba >= 0.60:
        st.warning("Reach out proactively. Consider a retention offer, contract upgrade, "
                   "or addressing service issues.")
    elif proba >= 0.30:
        st.info("Monitor this customer. Send a check-in survey or loyalty perk.")
    else:
        st.success("Customer is satisfied. Focus retention budget elsewhere.")

    # ---- Submitted inputs ----
    with st.expander("See submitted inputs"):
        display_df = input_data.T.rename(columns={0: "value"})
        display_df["value"] = display_df["value"].astype(str)
        st.dataframe(display_df, use_container_width=True)

# ---- Sidebar (outside if-block, runs every time) ----
st.sidebar.markdown("### About this page")
st.sidebar.info(
    "This page uses the best classifier from the experiment "
    "(Gradient Boosting with tuned hyperparameters) trained on 7,043 telecom customers. "
    "Test F1 ≈ 0.625, ROC-AUC ≈ 0.84."
)
st.sidebar.markdown("### Try these scenarios")
st.sidebar.markdown(
    "- **High risk:** Month-to-month contract, Fiber optic, no Tech Support, "
    "Electronic check, low tenure\n"
    "- **Low risk:** Two-year contract, long tenure, automatic payment"
)
