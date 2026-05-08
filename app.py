"""
app.py  –  Customer Churn Prediction Dashboard
===============================================
Multi-page Streamlit app with four sections:

  1. Overview & EDA       – combined dataset stats + interactive Plotly charts
  2. Model Performance    – confusion matrix, ROC curve, feature importance
  3. Single Prediction    – form input → churn probability gauge
  4. Batch Prediction     – upload CSV → scored table → download

Launch
------
    streamlit run app.py
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config  (must be the very first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data file paths ───────────────────────────────────────────────────────
TELCO_PATH        = "data/telco_churn.csv"
ORANGE_TRAIN_PATH = "data/orange_churn_train.csv"
ORANGE_TEST_PATH  = "data/orange_churn_test.csv"
IRANIAN_PATH      = "data/iranian_churn.csv"

DATASET_REGISTRY = {
    "IBM Telco (Kaggle)":     TELCO_PATH,
    "Orange Telecom (Kaggle)":[ORANGE_TRAIN_PATH, ORANGE_TEST_PATH],
    "Iranian Churn (UCI)":    IRANIAN_PATH,
}


# ═══════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading & harmonising datasets …")
def load_combined() -> pd.DataFrame:
    """Load all available datasets and return the unified DataFrame."""
    from data_loader import load_all_datasets
    return load_all_datasets()


@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artifacts():
    """Return (model, scaler, metrics, feature_cols) or None-tuple if not trained."""
    required = [
        "models/xgb_model.pkl",
        "models/scaler.pkl",
        "models/metrics.json",
        "models/feature_cols.json",
    ]
    if not all(os.path.exists(p) for p in required):
        return None, None, None, None

    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, metrics, feature_cols


def trigger_training() -> None:
    """Run train_model.train() and invalidate cached resources."""
    import train_model
    train_model.train()
    load_artifacts.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Page helpers – colour palette
# ═══════════════════════════════════════════════════════════════════════════
CHURN_COLOURS = {"Yes": "#EF553B", "No": "#636EFA",
                 1: "#EF553B",     0: "#636EFA"}


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview & EDA
# ═══════════════════════════════════════════════════════════════════════════

def page_eda(df: pd.DataFrame) -> None:
    st.header("📊 Dataset Overview & Exploratory Analysis")
    st.caption(
        "Unified view across IBM Telco, Orange Telecom, and Iranian Churn datasets"
    )

    # ── Dataset availability panel ────────────────────────────────────────
    with st.expander("Dataset availability", expanded=False):
        cols = st.columns(3)
        status = {
            "IBM Telco":    os.path.exists(TELCO_PATH),
            "Orange Train": os.path.exists(ORANGE_TRAIN_PATH),
            "Orange Test":  os.path.exists(ORANGE_TEST_PATH),
            "Iranian":      os.path.exists(IRANIAN_PATH),
        }
        for i, (name, ok) in enumerate(status.items()):
            cols[i % 3].metric(name, "✅ Loaded" if ok else "❌ Missing")

    # ── KPI row ───────────────────────────────────────────────────────────
    total    = len(df)
    churned  = int(df["churn"].sum())
    retained = total - churned

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers",    f"{total:,}")
    k2.metric("Churned",            f"{churned:,}")
    k3.metric("Retained",           f"{retained:,}")
    k4.metric("Churn Rate",         f"{churned/total:.1%}")
    k5.metric("Avg Monthly Charge", f"${df['monthly_charges'].mean():.2f}")

    st.divider()

    # ── Row 1 : churn donut + tenure histogram ────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        pie_df = df["churn"].map({1: "Yes", 0: "No"}).value_counts().reset_index()
        pie_df.columns = ["Churn", "Count"]
        fig = px.pie(
            pie_df, names="Churn", values="Count",
            title="Overall Churn Distribution",
            color="Churn", color_discrete_map=CHURN_COLOURS,
            hole=0.40,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        hist_df = df.copy()
        hist_df["Churn Label"] = hist_df["churn"].map({1: "Yes", 0: "No"})
        fig = px.histogram(
            hist_df, x="tenure", color="Churn Label",
            color_discrete_map=CHURN_COLOURS,
            barmode="overlay", opacity=0.72, nbins=36,
            title="Tenure Distribution by Churn Status",
            labels={"tenure": "Tenure (months)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2 : box plots ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    plot_df = df.copy()
    plot_df["Churn"] = plot_df["churn"].map({1: "Yes", 0: "No"})

    with c1:
        fig = px.box(
            plot_df, x="Churn", y="monthly_charges",
            color="Churn", color_discrete_map=CHURN_COLOURS,
            title="Monthly Charges vs Churn", points="outliers",
            labels={"monthly_charges": "Monthly Charges ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            plot_df, x="Churn", y="cust_service_calls",
            color="Churn", color_discrete_map=CHURN_COLOURS,
            title="Customer Service Calls vs Churn",
            labels={"cust_service_calls": "Support Calls"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3 : churn rate per binary / categorical feature ───────────────
    st.subheader("Churn Rate by Feature")

    binary_features = [
        "has_internet", "has_phone", "is_monthly_contract",
        "has_complaints", "is_senior",
    ]
    fig = make_subplots(
        rows=1, cols=len(binary_features),
        subplot_titles=[f.replace("_", " ").title() for f in binary_features],
    )
    for i, feat in enumerate(binary_features, start=1):
        rate = df.groupby(feat)["churn"].mean().mul(100).reset_index()
        rate.columns = [feat, "Churn Rate %"]
        fig.add_trace(
            go.Bar(
                x=rate[feat].map({0: "No", 1: "Yes"}).astype(str),
                y=rate["Churn Rate %"],
                marker_color=["#636EFA", "#EF553B"],
                showlegend=False,
            ),
            row=1, col=i,
        )
    fig.update_layout(height=360, title_text="Churn Rate (%) — Binary Features")
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4 : scatter + correlation heatmap ─────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        sample = plot_df.sample(min(2000, len(plot_df)), random_state=42)
        fig = px.scatter(
            sample, x="tenure", y="monthly_charges",
            color="Churn", color_discrete_map=CHURN_COLOURS,
            opacity=0.45, title="Tenure vs Monthly Charges",
            labels={"monthly_charges": "Monthly Charges ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        corr_cols = ["tenure", "monthly_charges", "total_charges",
                     "num_services", "cust_service_calls", "churn"]
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── num_services distribution ─────────────────────────────────────────
    st.subheader("Number of Add-on Services by Churn Status")
    fig = px.histogram(
        plot_df, x="num_services", color="Churn",
        color_discrete_map=CHURN_COLOURS,
        barmode="group", title="Add-on Services Count Distribution",
        labels={"num_services": "Number of Services"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data preview ──────────────────────────────────────────────────
    with st.expander("Preview harmonised data (first 200 rows)"):
        st.dataframe(df.head(200), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – Model Performance
# ═══════════════════════════════════════════════════════════════════════════

def page_performance(metrics: dict) -> None:
    st.header("📈 Model Performance")
    st.caption(
        "XGBoost  ·  3-dataset augmented training  ·  SMOTE balanced  ·  80/20 stratified split"
    )

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    k2.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")
    k3.metric("F1 Score", f"{metrics['f1']:.4f}")

    st.divider()

    # ── Confusion matrix + ROC curve ──────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True,
            labels={"x": "Predicted", "y": "Actual"},
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            aspect="equal",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"XGBoost  (AUC = {metrics['roc_auc']:.3f})",
            line=dict(color="#636EFA", width=2.5),
            fill="tozeroy", fillcolor="rgba(99,110,250,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random  (AUC = 0.500)",
            line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────
    st.subheader("Feature Importances (XGBoost gain)")
    fi_path = "models/feature_importance.csv"
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h", color="importance",
            color_continuous_scale="Viridis",
            title="XGBoost Feature Importance",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Classification report ──────────────────────────────────────────────
    st.subheader("Classification Report")
    cr = metrics["classification_report"]
    rows = []
    for label, display in [("0", "No Churn"), ("1", "Churn"),
                            ("macro avg", "Macro Avg"),
                            ("weighted avg", "Weighted Avg")]:
        rows.append({
            "Class":     display,
            "Precision": cr[label]["precision"],
            "Recall":    cr[label]["recall"],
            "F1-Score":  cr[label]["f1-score"],
            "Support":   int(cr[label]["support"]) if "support" in cr[label] else "",
        })
    report_df = pd.DataFrame(rows).set_index("Class")
    st.dataframe(
        report_df.style.format(
            {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}
        ),
        use_container_width=True,
    )

    # ── Interpretation guide ───────────────────────────────────────────────
    with st.expander("Metric Interpretation Guide"):
        st.markdown(
            """
**Accuracy** – fraction of all predictions that are correct.

**ROC-AUC** – measures separability between churners and non-churners,
independent of the decision threshold. 0.5 = random; 1.0 = perfect.

**F1 Score** – harmonic mean of Precision and Recall for the Churn class.
Important here because the class distribution is imbalanced (~26–30 % churn).

**SMOTE** – Synthetic Minority Over-sampling Technique was applied to the
training set only to balance churn/non-churn before fitting XGBoost.
The test set is kept as-is to give an honest estimate of real-world performance.

**Multi-dataset training** – combining IBM Telco, Orange Telecom, and Iranian
Churn data gives the model exposure to diverse churn signals (billing patterns,
call behaviour, subscription length) reducing single-domain overfitting.
            """
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – Single Customer Prediction
# ═══════════════════════════════════════════════════════════════════════════

def page_single(model, scaler, feature_cols: list) -> None:
    from data_loader import FEATURE_LABELS

    st.header("🔮 Single Customer Churn Prediction")
    st.markdown("Fill in the customer profile and click **Predict** to get the churn probability.")

    with st.form("single_pred_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Subscription")
            tenure      = st.slider("Tenure (months)",          0,   72,  24)
            monthly     = st.slider("Monthly Charges ($)",      5.0, 150.0, 65.0, step=0.5)
            total_def   = round(tenure * monthly, 2)
            total       = st.number_input("Total Charges ($)",
                            min_value=0.0, max_value=15000.0,
                            value=total_def, step=10.0)

        with c2:
            st.subheader("Plan & Contract")
            has_internet = st.selectbox("Has Internet / Data Plan", ["No", "Yes"])
            has_phone    = st.selectbox("Has Phone / Voice Plan",   ["Yes", "No"])
            is_monthly   = st.selectbox("Contract Type",
                            ["Month-to-Month", "Annual / Long-term"],
                            help="Month-to-Month maps to is_monthly_contract = 1")
            num_services = st.slider("Number of Add-on Services",  0,   7,  1)

        with c3:
            st.subheader("Support & Demographics")
            csc          = st.slider("Customer Service Calls",     0,   10, 1)
            complaints   = st.selectbox("Formal Complaints",      ["No", "Yes"])
            is_senior    = st.selectbox("Senior Citizen",          ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Predict Churn Probability", type="primary")

    if submitted:
        inp = {
            "tenure":              float(tenure),
            "monthly_charges":     float(monthly),
            "total_charges":       float(total),
            "num_services":        float(num_services),
            "has_internet":        1.0 if has_internet == "Yes" else 0.0,
            "has_phone":           1.0 if has_phone    == "Yes" else 0.0,
            "is_monthly_contract": 1.0 if is_monthly   == "Month-to-Month" else 0.0,
            "cust_service_calls":  float(csc),
            "has_complaints":      1.0 if complaints   == "Yes" else 0.0,
            "is_senior":           1.0 if is_senior    == "Yes" else 0.0,
        }

        # Build DataFrame → scale → predict  (no re-fitting of scaler)
        X_inp = pd.DataFrame([inp])[feature_cols].values
        X_sc  = scaler.transform(X_inp)
        prob  = float(model.predict_proba(X_sc)[0, 1])
        pred  = int(model.predict(X_sc)[0])

        st.divider()
        r1, r2 = st.columns(2)

        with r1:
            if pred == 1:
                st.error("⚠️  This customer is **likely to CHURN**")
            else:
                st.success("✅  This customer is **likely to STAY**")

            st.metric("Churn Probability",     f"{prob:.1%}")
            st.metric("Retention Probability", f"{1 - prob:.1%}")

            risk   = "High" if prob >= 0.60 else "Medium" if prob >= 0.30 else "Low"
            icon   = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"**Risk Level :** {icon[risk]} &nbsp; **{risk}**")

            st.markdown("---")
            st.markdown("**Input Summary**")
            summary = {FEATURE_LABELS.get(k, k): v for k, v in inp.items()}
            st.dataframe(
                pd.DataFrame(summary.items(), columns=["Feature", "Value"]),
                use_container_width=True, hide_index=True,
            )

        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 40}},
                delta={"reference": 27.0, "suffix": "% vs baseline", "valueformat": ".1f"},
                title={"text": "Churn Risk Score", "font": {"size": 22}},
                gauge={
                    "axis":  {"range": [0, 100], "tickwidth": 1},
                    "bar":   {"color": "navy", "thickness": 0.25},
                    "steps": [
                        {"range": [0,  30], "color": "#2dc937"},
                        {"range": [30, 60], "color": "#e7b416"},
                        {"range": [60, 100],"color": "#cc3232"},
                    ],
                    "threshold": {
                        "line":      {"color": "black", "width": 4},
                        "thickness": 0.85,
                        "value":     prob * 100,
                    },
                },
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 – Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════

def page_batch(model, scaler, feature_cols: list) -> None:
    from data_loader import FEATURE_LABELS

    st.header("📋 Batch Churn Prediction")
    st.markdown(
        "Upload a CSV with customer records.  "
        "The model will score each row and return a churn probability."
    )

    with st.expander("Expected CSV columns"):
        st.code(", ".join(feature_cols))
        st.markdown(
            "All columns must be numeric.  "
            "Binary columns use **1 = Yes, 0 = No**.  "
            "Any extra columns (e.g. customerID) are kept but not used."
        )

    # ── Sample download ────────────────────────────────────────────────────
    sample_data = {col: [0.5] for col in feature_cols}
    sample_csv  = pd.DataFrame(sample_data).to_csv(index=False).encode()
    st.download_button("⬇ Download sample CSV template",
                       sample_csv, "sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse the file: {e}")
        return

    # Validate required columns
    missing = [c for c in feature_cols if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    X_batch = raw[feature_cols].astype(float).values
    X_sc    = scaler.transform(X_batch)
    probs   = model.predict_proba(X_sc)[:, 1]
    preds   = model.predict(X_sc)

    result = raw.copy()
    result["churn_probability"]  = probs.round(4)
    result["churn_prediction"]   = ["Yes" if p == 1 else "No" for p in preds]
    result["risk_level"]         = pd.cut(
        probs, bins=[0, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    )

    # ── Summary ────────────────────────────────────────────────────────────
    st.success(f"Predictions generated for **{len(result):,}** customers.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total",         f"{len(result):,}")
    k2.metric("High Risk",     int((result["risk_level"] == "High").sum()))
    k3.metric("Medium Risk",   int((result["risk_level"] == "Medium").sum()))
    k4.metric("Low Risk",      int((result["risk_level"] == "Low").sum()))

    # ── Distribution chart ─────────────────────────────────────────────────
    fig = px.histogram(
        result, x="churn_probability",
        color="risk_level",
        color_discrete_map={"Low": "#2dc937", "Medium": "#e7b416", "High": "#cc3232"},
        nbins=40,
        title="Churn Probability Distribution Across Batch",
        labels={"churn_probability": "Churn Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Results table ──────────────────────────────────────────────────────
    st.subheader("Scored Results")
    st.dataframe(
        result.style.background_gradient(
            subset=["churn_probability"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True, height=400,
    )

    csv_bytes = result.to_csv(index=False).encode()
    st.download_button("⬇ Download Predictions CSV",
                       csv_bytes, "churn_predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# App entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📉 Churn Dashboard")
        st.caption("XGBoost · Multi-Dataset · Streamlit")
        st.divider()
        page = st.radio(
            "Navigate",
            [
                "🏠 Overview & EDA",
                "📈 Model Performance",
                "🔮 Single Prediction",
                "📋 Batch Prediction",
            ],
        )
        st.divider()
        available = [
            os.path.exists(TELCO_PATH),
            os.path.exists(ORANGE_TRAIN_PATH),
            os.path.exists(IRANIAN_PATH),
        ]
        labels = ["IBM Telco", "Orange Telecom", "Iranian Churn"]
        for lab, ok in zip(labels, available):
            st.markdown(f"{'✅' if ok else '❌'}  {lab}")

    # ── Guard: at least one dataset must exist ────────────────────────────
    if not any(os.path.exists(p) for p in [TELCO_PATH, ORANGE_TRAIN_PATH, IRANIAN_PATH]):
        st.error("No dataset files found in `data/`.")
        st.info("Download at least one dataset — see **DATASET.md** for instructions.")
        return

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        df = load_combined()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    model, scaler, metrics, feature_cols = load_artifacts()

    # ── Guard: model must be trained ──────────────────────────────────────
    if model is None:
        st.warning("No trained model found in `models/`.")
        st.code("python train_model.py", language="bash")
        if st.button("🚀 Train Model Now  (runs in app)", type="primary"):
            with st.spinner("Training XGBoost — typically takes 30–60 s …"):
                trigger_training()
            st.success("Training complete!")
            st.rerun()
        return

    # ── Page routing ──────────────────────────────────────────────────────
    if page == "🏠 Overview & EDA":
        page_eda(df)
    elif page == "📈 Model Performance":
        page_performance(metrics)
    elif page == "🔮 Single Prediction":
        page_single(model, scaler, feature_cols)
    elif page == "📋 Batch Prediction":
        page_batch(model, scaler, feature_cols)


if __name__ == "__main__":
    main()
