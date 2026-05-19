import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Customer Segmentation", page_icon="🔍", layout="wide")

st.title("🔍 Customer Segmentation")
st.markdown("Enter a customer's profile to see which segment they belong to.")

PROJECT_ROOT = Path(__file__).parent.parent

@st.cache_resource
def load_artifacts():
    """Load the trained KMeans model, scaler, and original Mall data."""
    kmeans  = joblib.load(PROJECT_ROOT / "models" / "kmeans_clusterer.pkl")
    scaler  = joblib.load(PROJECT_ROOT / "models" / "cluster_scaler.pkl")
    mall_df = pd.read_csv(PROJECT_ROOT / "datasets" / "Mall_Customers.csv")
    return kmeans, scaler, mall_df

try:
    kmeans, scaler, mall_df = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e}. "
             f"Run the clustering notebook first to create the model artifacts.")
    st.stop()

# Pre-compute the cluster labels for all existing customers (for plotting)
X_existing = mall_df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_existing_scaled = scaler.transform(X_existing)
mall_df["Cluster"] = kmeans.predict(X_existing_scaled)


# Business-friendly labels for each cluster (based on the profile analysis from the notebook)
CLUSTER_PROFILES = {
    0: {"name": "Average Shoppers",     "emoji": "🛍️", "color": "#3498db"},
    1: {"name": "Cautious Wealthy",     "emoji": "💎", "color": "#9b59b6"},
    2: {"name": "Young Spenders",       "emoji": "🎉", "color": "#e74c3c"},
    3: {"name": "Premium Customers",    "emoji": "⭐", "color": "#27ae60"},
    4: {"name": "Budget-Conscious",     "emoji": "💰", "color": "#f39c12"},
}

st.subheader("Customer profile")

with st.form("segmentation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", min_value=18, max_value=80, value=35)
    with col2:
        income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=60)
    with col3:
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    submitted = st.form_submit_button("🔮 Find Segment", use_container_width=True, type="primary")


if submitted:
    # Scale the new point and predict its cluster
    new_point = np.array([[income, spending]])
    new_point_scaled = scaler.transform(new_point)
    cluster_id = int(kmeans.predict(new_point_scaled)[0])

    profile = CLUSTER_PROFILES.get(cluster_id, {"name": f"Cluster {cluster_id}", "emoji": "❓", "color": "#7f8c8d"})

    # ---- Result banner ----
    st.divider()
    st.subheader("Result")

    rcol1, rcol2 = st.columns([1, 2])

    with rcol1:
        st.markdown(f"### {profile['emoji']} {profile['name']}")
        st.markdown(f"**Cluster ID:** {cluster_id}")
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Income:** ${income}k")
        st.markdown(f"**Spending Score:** {spending}/100")

    with rcol2:
        # ---- Scatter plot: all customers, colored by cluster, with the new point ----
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot each cluster
        for cid, prof in CLUSTER_PROFILES.items():
            cluster_data = mall_df[mall_df["Cluster"] == cid]
            ax.scatter(
                cluster_data["Annual Income (k$)"],
                cluster_data["Spending Score (1-100)"],
                c=prof["color"],
                s=60, alpha=0.6, edgecolor="black",
                label=f"{prof['emoji']} {prof['name']}",
            )

        # Plot the new customer as a big black star
        ax.scatter(income, spending, c="black", marker="*", s=400,
                   edgecolor="yellow", linewidth=2, label="New Customer", zorder=10)

        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title("Customer Segments")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # ---- Recommendation per segment ----
        st.markdown("##### Marketing recommendation")

        recommendations = {
            "Premium Customers":   "Offer VIP perks, exclusive previews, and concierge service. "
                                   "These customers are your highest-value group — focus on retention.",
            "Cautious Wealthy":    "Untapped revenue potential. Try premium product trials, "
                                   "loyalty programs, and curated recommendations to nudge spending.",
            "Young Spenders":      "Engage with trendy products, social campaigns, and limited-time offers. "
                                   "Build long-term loyalty before income grows.",
            "Average Shoppers":    "The mall's bread and butter. Broad seasonal campaigns and consistent value messaging.",
            "Budget-Conscious":    "Focus on discounts, value bundles, and essentials. Avoid premium upsells.",
        }
        st.info(recommendations.get(profile["name"], "Tailor strategy based on segment characteristics."))

    # ---- Recommendation per segment ----
    st.markdown("##### Marketing recommendation")

    recommendations = {
        "Premium Customers":   "Offer VIP perks, exclusive previews, and concierge service. "
                                   "These customers are your highest-value group — focus on retention.",
            "Cautious Wealthy":    "Untapped revenue potential. Try premium product trials, "
                                   "loyalty programs, and curated recommendations to nudge spending.",
            "Young Spenders":      "Engage with trendy products, social campaigns, and limited-time offers. "
                                   "Build long-term loyalty before income grows.",
            "Average Shoppers":    "The mall's bread and butter. Broad seasonal campaigns and consistent value messaging.",
            "Budget-Conscious":    "Focus on discounts, value bundles, and essentials. Avoid premium upsells.",
        }
    st.info(recommendations.get(profile["name"], "Tailor strategy based on segment characteristics."))

st.sidebar.markdown("### About this page")
st.sidebar.info(
    "Clusters were discovered by KMeans (k=5) on Annual Income and Spending Score, "
    "validated by Silhouette ≈ 0.55 and confirmed by Elbow and Hierarchical clustering."
)
st.sidebar.markdown("### Try these profiles")
st.sidebar.markdown(
    "- **Premium:** Income 90+, Spending 80+\n"
    "- **Cautious Wealthy:** Income 90+, Spending under 40\n"
    "- **Young Spenders:** Income under 40, Spending 70+\n"
    "- **Budget-Conscious:** Income under 40, Spending under 40"
)
