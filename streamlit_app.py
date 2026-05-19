import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Minging Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Header ----
st.title("📊 CSC 588 — Data Mining Project")
st.markdown("### Classification • Clustering • Association Analysis")

st.markdown("""
This interactive system demonstrates three core data mining techniques applied to
real-world business datasets. Use the sidebar to navigate between modules.
""")

st.divider()


# ---- Project overview in three columns ----
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Classification")
    st.markdown("""
    **Telco Customer Churn**
    - 7,043 customers × 21 features
    - Goal: predict who will leave
    - 11 algorithms compared
    - **Best model:** Gradient Boosting (F1 = 0.625)
    """)

with col2:
    st.subheader("🔍 Clustering")
    st.markdown("""
    **Mall Customer Segmentation**
    - 200 customers × 5 features
    - Goal: discover natural segments
    - 4 algorithms compared
    - **Result:** 5 distinct customer groups
    """)

with col3:
    st.subheader("🛒 Association Rules")
    st.markdown("""
    **Grocery Transactions**
    - 14,963 baskets × 167 items
    - Goal: find co-purchase patterns
    - Apriori vs FP-Growth
    - **Result:** 8 strong rules, top lift = 2.18
    """)

st.divider()

# ---- Method overview ----
st.subheader("Methodology")
st.markdown("""
- **Preprocessing:** column-specific scaling/encoding via `ColumnTransformer`,
  stratified train/test splits, SMOTE for class imbalance.
- **Validation:** 5-fold stratified cross-validation during hyperparameter tuning,
  held-out test sets for final evaluation.
- **Hyperparameter tuning:** GridSearchCV on the top-3 baseline classifiers.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, training time
  (classification); silhouette, Davies-Bouldin, Calinski-Harabasz (clustering);
  support, confidence, lift (association).
""")

# ---- Sidebar info ----
st.sidebar.title("Navigation")
st.sidebar.success("Select a module above ☝️")
st.sidebar.markdown("---")
st.sidebar.info("""
**Project files**
- 📓 Notebooks in `/notebooks/`
- 🤖 Trained models in `/models/`
- 📈 Results in `/results/`
""")
