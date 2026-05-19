import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import ast

st.set_page_config(page_title="Association Rules", page_icon="🛒", layout="wide")

st.title("🛒 Association Rules — Grocery Basket Analysis")
st.markdown(
    "Explore co-purchase patterns discovered from 14,963 grocery transactions. "
    "Use the filters in the sidebar to refine which rules you see."
)

PROJECT_ROOT = Path(__file__).parent.parent

@st.cache_data
def load_rules():
    """Load saved association rules and parse frozensets back from strings."""
    rules_path = PROJECT_ROOT / "results" / "association_rules.csv"
    if not rules_path.exists():
        return None
    df = pd.read_csv(rules_path)

    # When saved to CSV, frozensets become strings like "frozenset({'whole milk'})".
    # Parse them back into clean Python sets, then format for display.
    def parse(x):
        if isinstance(x, str) and x.startswith("frozenset"):
            return ast.literal_eval(x.replace("frozenset(", "").rstrip(")"))
        return x

    df["antecedents_set"] = df["antecedents"].apply(parse)
    df["consequents_set"] = df["consequents"].apply(parse)
    df["antecedents_str"] = df["antecedents_set"].apply(lambda s: ", ".join(sorted(s)))
    df["consequents_str"] = df["consequents_set"].apply(lambda s: ", ".join(sorted(s)))
    df["rule"] = df["antecedents_str"] + " → " + df["consequents_str"]
    return df

@st.cache_data
def load_sensitivity():
    path = PROJECT_ROOT / "results" / "association_sensitivity.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_timing():
    path = PROJECT_ROOT / "results" / "association_timing.csv"
    return pd.read_csv(path) if path.exists() else None

rules_df       = load_rules()
sensitivity_df = load_sensitivity()
timing_df      = load_timing()

if rules_df is None:
    st.error("❌ `results/association_rules.csv` not found. "
             "Run the association notebook first to generate the rules.")
    st.stop()

st.sidebar.header("🔍 Filters")

# Compute the actual min/max of available rules so the sliders never go out of range
min_lift = float(rules_df["lift"].min())
max_lift = float(rules_df["lift"].max())
min_conf = float(rules_df["confidence"].min())
max_conf = float(rules_df["confidence"].max())

lift_threshold = st.sidebar.slider(
    "Minimum lift",
    min_value=round(min_lift, 2),
    max_value=round(max_lift, 2),
    value=round(min_lift, 2),
    step=0.05,
)

conf_threshold = st.sidebar.slider(
    "Minimum confidence",
    min_value=round(min_conf, 2),
    max_value=round(max_conf, 2),
    value=round(min_conf, 2),
    step=0.01,
)

# Build a sorted list of all unique items appearing anywhere
all_items = sorted(
    set().union(*rules_df["antecedents_set"].tolist(), *rules_df["consequents_set"].tolist())
)
item_filter = st.sidebar.selectbox(
    "Filter by item (optional)",
    ["(any)"] + all_items,
)


# Start with the full set, narrow it down by user choices
filtered = rules_df[
    (rules_df["lift"] >= lift_threshold) &
    (rules_df["confidence"] >= conf_threshold)
].copy()

if item_filter != "(any)":
    mask = (
        filtered["antecedents_set"].apply(lambda s: item_filter in s) |
        filtered["consequents_set"].apply(lambda s: item_filter in s)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values("lift", ascending=False).reset_index(drop=True)

# ---- Top-line metrics ----
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Rules shown", f"{len(filtered):,}")
mcol2.metric("Total rules",  f"{len(rules_df):,}")
mcol3.metric("Max lift",     f"{filtered['lift'].max():.2f}" if len(filtered) else "—")
mcol4.metric("Max confidence", f"{filtered['confidence'].max():.1%}" if len(filtered) else "—")

st.divider()

# ---- Rules table ----
st.subheader("Rules")

if len(filtered) == 0:
    st.warning("No rules match the current filters. Try lowering the thresholds.")
else:
    display_df = filtered[["rule", "support", "confidence", "lift"]].rename(
        columns={"rule": "Rule", "support": "Support",
                 "confidence": "Confidence", "lift": "Lift"}
    )
    st.dataframe(
        display_df.style.format({
            "Support":    "{:.4f}",
            "Confidence": "{:.2%}",
            "Lift":       "{:.3f}",
        }),
        use_container_width=True,
        height=400,
    )


if len(filtered) > 0:
    st.divider()
    st.subheader("Visualizations")

    vcol1, vcol2 = st.columns(2)

    # ---- Scatter plot ----
    with vcol1:
        st.markdown("**Support vs Confidence (color = Lift)**")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        scatter = ax1.scatter(
            filtered["support"],
            filtered["confidence"],
            c=filtered["lift"],
            cmap="viridis",
            s=80, alpha=0.7, edgecolor="black",
        )
        plt.colorbar(scatter, ax=ax1, label="Lift")
        ax1.set_xlabel("Support")
        ax1.set_ylabel("Confidence")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # ---- Top rules bar chart ----
    with vcol2:
        st.markdown("**Top 10 rules by lift**")
        top10 = filtered.head(10).copy()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.barh(top10["rule"][::-1], top10["lift"][::-1], color="teal")
        ax2.set_xlabel("Lift")
        ax2.tick_params(axis="y", labelsize=8)
        ax2.grid(axis="x", alpha=0.3)
        st.pyplot(fig2)

st.divider()
st.subheader("Methodology details")

mcol1, mcol2 = st.columns(2)

# ---- Algorithm comparison table ----
with mcol1:
    st.markdown("##### Algorithm comparison")
    if timing_df is not None:
        st.dataframe(timing_df, use_container_width=True, hide_index=True)
        speedup = timing_df.loc[timing_df["Algorithm"] == "Apriori", "Time (s)"].values[0] / \
                  timing_df.loc[timing_df["Algorithm"] == "FP-Growth", "Time (s)"].values[0]
        st.caption(f"FP-Growth was {speedup:.1f}× faster on this dataset.")
    else:
        st.info("Run the association notebook to generate timing data.")

# ---- Sensitivity analysis ----
with mcol2:
    st.markdown("##### Sensitivity to min_support")
    if sensitivity_df is not None:
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
        st.caption("Lower min_support → more itemsets, but more noise.")
    else:
        st.info("Run the association notebook to generate sensitivity data.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About this page")
st.sidebar.info(
    "Rules were mined from 14,963 grocery transactions using Apriori and FP-Growth "
    "(min_support = 0.001). Both algorithms found 750 frequent itemsets; "
    "FP-Growth ran ~3× faster."
)
