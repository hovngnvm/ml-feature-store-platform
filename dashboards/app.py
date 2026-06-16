import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Real-Time Feature Store & Fraud Detection AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "ensemble_fraud_model.joblib")
REPORT_JSON_PATH = os.path.join(PROJECT_DIR, "models", "evaluation_report.json")
DRIFT_HTML_PATH = os.path.join(PROJECT_DIR, "dashboards", "feature_drift_report.html")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Bind FraudModelEnsemble for joblib
try:
    from src.ml.train import FraudModelEnsemble
    sys.modules['__main__'].FraudModelEnsemble = FraudModelEnsemble
except Exception:
    pass

@st.cache_resource
def load_ensemble_model():
    """Loads and caches trained Model Ensemble pipeline."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

@st.cache_data
def load_evaluation_report():
    """Loads model evaluation metrics JSON."""
    if os.path.exists(REPORT_JSON_PATH):
        try:
            with open(REPORT_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def get_card_historical_features(card_id_str: str):
    """Lookup real features from batch_features.parquet or compute card-specific deterministic baseline."""
    parquet_path = os.path.join(PROJECT_DIR, "data", "batch_features.parquet")
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            matched = df[df["card_id"].astype(str) == str(card_id_str)]
            if not matched.empty:
                row = matched.iloc[0]
                return {
                    "trans_count_7d": float(row.get("trans_count_7d", 2.0)),
                    "trans_count_30d": float(row.get("trans_count_30d", 8.0)),
                    "avg_amount_30d": float(row.get("avg_amount_30d", 150.0)),
                    "max_amount_30d": float(row.get("max_amount_30d", 500.0)),
                    "distinct_addr_7d": float(row.get("distinct_addr_7d", 1.0)),
                    "days_since_last_trans": float(row.get("days_since_last_trans", 1.5)),
                }
        except Exception:
            pass

    card_hash = abs(hash(str(card_id_str)))
    avg_amt = float(30.0 + (card_hash % 500))
    max_amt = float(avg_amt * (1.5 + (card_hash % 3)))
    return {
        "trans_count_7d": float(1 + (card_hash % 10)),
        "trans_count_30d": float(3 + (card_hash % 30)),
        "avg_amount_30d": avg_amt,
        "max_amount_30d": max_amt,
        "distinct_addr_7d": float(1 + (card_hash % 4)),
        "days_since_last_trans": float(0.5 + (card_hash % 14)),
    }

# Custom Glassmorphic Dark Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .status-approved {
        color: #00e676;
        font-weight: bold;
        font-size: 24px;
    }
    .status-fraud {
        color: #ff1744;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🛡️ Real-Time Feature Store & Fraud Detection AI Platform")
st.caption("Powered by PyFlink, DuckDB, Feast, Redis, XGBoost + LightGBM Model Ensemble & Evidently AI")

model_pipeline = load_ensemble_model()
eval_report = load_evaluation_report()

# Create App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💳 Real-Time Fraud Simulator & SHAP",
    "📊 Model Performance & Ensemble Benchmark",
    "🏪 Redis Feature Store Inspector",
    "🛡️ Evidently AI Data Drift"
])

# =============================================================================
# TAB 1: Real-Time Fraud Simulator & SHAP Explainability
# =============================================================================
with tab1:
    st.subheader("💳 Real-Time Transaction Fraud Prediction Simulator")
    st.write("Simulate a credit card transaction and observe real-time fraud probability scores & SHAP feature attributions.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### 📝 Transaction Payload Input")
        card_id = st.text_input("Card ID", value="11556", help="Card identifier")
        curr_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=250.0, step=10.0)

        st.markdown("#### ⚡ Preset Test Scenarios")
        c_p1, c_p2, c_p3 = st.columns(3)
        if c_p1.button("🟢 Normal ($45)"):
            curr_amount = 45.0
        if c_p2.button("🟡 Medium ($350)"):
            curr_amount = 350.0
        if c_p3.button("🔴 High Risk ($2,800)"):
            curr_amount = 2800.0

        btn_predict = st.button("🚀 Predict Fraud Score", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Prediction & Risk Assessment")
        if btn_predict or curr_amount > 0:
            if model_pipeline is None:
                st.warning("⚠️ Model artifact not loaded. Please run `python src/ml/train.py` first.")
            else:
                start_t = time.perf_counter()

                # Dynamic feature retrieval based on entered Card ID
                online_features = get_card_historical_features(card_id)
                online_features["TransactionAmt"] = float(curr_amount)

                # Compute derived features
                avg_30d = online_features["avg_amount_30d"]
                max_30d = online_features["max_amount_30d"]
                online_features["amount_ratio_30d"] = curr_amount / (avg_30d + 1.0)
                online_features["is_amount_gt_30d_max"] = 1.0 if curr_amount > max_30d else 0.0

                # Prepare DataFrame
                feature_cols = model_pipeline.feature_names
                input_df = pd.DataFrame([online_features])[feature_cols]

                # Run Inference
                xgb_p = float(model_pipeline.xgb_model.predict_proba(input_df)[0, 1])
                lgb_p = float(model_pipeline.lgb_model.predict_proba(input_df)[0, 1])
                fraud_score = float(model_pipeline.predict_proba(input_df)[0, 1])
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                is_fraud = fraud_score >= 0.5
                decision_str = "🚨 ALERT: FRAUD DETECTED" if is_fraud else "✅ APPROVED (LEGITIMATE)"
                status_class = "status-fraud" if is_fraud else "status-approved"

                st.markdown(f'<div class="{status_class}">{decision_str}</div>', unsafe_allow_html=True)
                st.markdown(f"**Fraud Probability Score:** `{fraud_score * 100:.2f}%` | **SLA Latency:** `{latency_ms:.2f} ms`")

                # Gauge Meter Plot
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_score * 100,
                    title={'text': "Fraud Risk Score (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff1744" if is_fraud else "#00e676"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(0, 230, 118, 0.2)"},
                            {'range': [30, 60], 'color': "rgba(255, 235, 59, 0.2)"},
                            {'range': [60, 100], 'color': "rgba(255, 23, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("XGBoost Score", f"{xgb_p * 100:.1f}%")
                m2.metric("LightGBM Score", f"{lgb_p * 100:.1f}%")
                m3.metric("Ensemble Score", f"{fraud_score * 100:.1f}%")

    st.markdown("---")
    st.subheader("🔍 SHAP Feature Explainability Analysis")
    st.write("Understand **WHY** the model assigned this prediction score by inspecting feature contribution values.")

    if model_pipeline and btn_predict:
        try:
            explainer = shap.TreeExplainer(model_pipeline.xgb_model)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP Waterfall plot feature attribution notice: {e}")

# =============================================================================
# TAB 2: Model Performance & Ensemble Benchmark
# =============================================================================
with tab2:
    st.subheader("📊 Model Performance & Ensemble Benchmark")
    st.write("Comparative evaluation metrics (PR-AUC, ROC-AUC, F1-Score) across XGBoost, LightGBM, and Model Ensemble.")

    if eval_report:
        metrics_dict = eval_report.get("metrics", {})
        
        # Format comparison table
        rows = []
        for model_name, m_data in metrics_dict.items():
            rows.append({
                "Model Architecture": model_name.upper(),
                "PR-AUC (Precision-Recall AUC)": m_data.get("pr_auc"),
                "ROC-AUC Score": m_data.get("roc_auc"),
                "F1-Score": m_data.get("f1_score")
            })
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True)

        # Bar chart comparison
        fig_bar = px.bar(
            df_metrics,
            x="Model Architecture",
            y=["PR-AUC (Precision-Recall AUC)", "ROC-AUC Score"],
            barmode="group",
            title="Model Evaluation Metric Comparison",
            color_discrete_sequence=["#00e676", "#29b6f6"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ Evaluation report JSON not found at `models/evaluation_report.json`. Run `python src/ml/train.py` to generate.")

# =============================================================================
# TAB 3: Redis Feature Store Inspector
# =============================================================================
with tab3:
    st.subheader("🏪 Redis Online Feature Store Inspector")
    st.write("Inspect low-latency feature vectors stored in Redis Online Store.")

    lookup_card = st.text_input("Enter Card ID to Inspect", value="11556")
    if st.button("🔍 Search Redis Keys"):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            key_name = f"card:{lookup_card}:stream_features"
            data_hash = r.hgetall(key_name)
            if data_hash:
                st.success(f"Found Redis Key: `{key_name}`")
                st.json(data_hash)
            else:
                st.info(f"Key `{key_name}` not populated yet in Redis. Connect live stream engine or run Feast materialization.")
        except Exception as e:
            st.error(f"Redis Connection Error: {e}")

# =============================================================================
# TAB 4: Evidently AI Data Drift Report
# =============================================================================
with tab4:
    st.subheader("🛡️ Evidently AI Feature Drift & Quality Report")
    st.write("Embedded interactive Data Drift report generated by Evidently AI engine.")

    if os.path.exists(DRIFT_HTML_PATH):
        with open(DRIFT_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning(f"⚠️ Data Drift HTML Report not found at `{DRIFT_HTML_PATH}`. Run `python src/quality/feature_monitoring.py` to generate.")
