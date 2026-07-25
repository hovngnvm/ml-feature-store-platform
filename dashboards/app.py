from pathlib import Path
import json
import time
from typing import Any
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import redis
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Real-Time Feature Store & Fraud Detection AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config.settings import settings
from src.ml.ensemble import FraudModelEnsemble

MODEL_PATH = Path(settings.model_artifact_path)
REPORT_JSON_PATH = Path(settings.report_json_path)
DRIFT_HTML_PATH = Path(settings.dashboard_dir) / "feature_drift_report.html"
REDIS_HOST = settings.redis_host
REDIS_PORT = settings.redis_port


@st.cache_resource
def load_ensemble_model() -> Any:
    """Loads and caches trained Model Ensemble pipeline."""
    if MODEL_PATH.is_file():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


@st.cache_data
def load_evaluation_report() -> dict | None:
    """Loads model evaluation metrics JSON."""
    if REPORT_JSON_PATH.is_file():
        try:
            with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def get_card_historical_features(card_id_str: str) -> tuple[dict[str, float], bool]:
    """Lookup real features from batch_features.parquet or return cold-start baseline."""
    parquet_path = Path(settings.batch_parquet_path)
    if parquet_path.is_file():
        try:
            df = pd.read_parquet(parquet_path)
            matched = df[df["card_id"].astype(str) == str(card_id_str)]
            if not matched.empty:
                row = matched.iloc[0]
                return {
                    "trans_count_7d": float(row.get("trans_count_7d", 1.0)),
                    "trans_count_30d": float(row.get("trans_count_30d", 5.0)),
                    "avg_amount_30d": float(row.get("avg_amount_30d", 150.0)),
                    "max_amount_30d": float(row.get("max_amount_30d", 500.0)),
                    "distinct_addr_7d": float(row.get("distinct_addr_7d", 1.0)),
                    "days_since_last_trans": float(row.get("days_since_last_trans", 2.0)),
                }, True
        except Exception:
            pass

    return {
        "trans_count_7d": 1.0,
        "trans_count_30d": 1.0,
        "avg_amount_30d": 100.0,
        "max_amount_30d": 200.0,
        "distinct_addr_7d": 1.0,
        "days_since_last_trans": 1.0,
    }, False

# Custom Glassmorphic Dark Styling
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .status-fraud {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.2), rgba(213, 0, 0, 0.4));
        border: 1px solid #ff1744;
        border-radius: 8px;
        padding: 12px;
        color: #ff5252;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .status-approved {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.2), rgba(0, 200, 83, 0.4));
        border: 1px solid #00e676;
        border-radius: 8px;
        padding: 12px;
        color: #69f0ae;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("💳 Real-Time Fraud Detection & Feature Store AI Platform")
st.markdown("Enterprise MLOps Platform with Dual-Path Feature Ingestion, Model Ensemble (XGBoost + LightGBM), and Dynamic Decision Threshold Tuning.")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Real-Time Model Serving (< 5ms SLA)",
    "📊 Model Performance & Cost Optimization",
    "🏪 Redis Feature Store Inspector",
    "🛡️ Evidently AI Data Drift Report"
])

model_pipeline = load_ensemble_model()
eval_report = load_evaluation_report()

# Tab 1: Real-Time Model Serving Simulator
with tab1:
    st.subheader("⚡ Low-Latency Fraud Prediction & Dynamic Threshold Inference")
    st.write("Simulate real-time transactions by querying online features from Redis/Feast and evaluating risk against the Cost-Optimal Decision Threshold.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### 📝 Transaction Simulation Parameters")
        card_id = st.text_input("Credit Card ID (card1)", value="11556")
        amount = st.number_input("Transaction Amount ($ USD)", min_value=1.0, max_value=50000.0, value=250.0, step=10.0)
        
        card_features, is_found = get_card_historical_features(card_id)
        
        with st.expander("🔍 Online Historical Features (from Feast/Redis)", expanded=True):
            if not is_found:
                st.caption("ℹ️ Card not found in offline store. Using cold-start baseline features.")
            st.json(card_features)

        btn_predict = st.button("🚀 Score Transaction (Inference SLA < 5ms)", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Decision & Risk Scoring Result")
        if btn_predict:
            if model_pipeline is None:
                st.error("Model is not loaded! Please run `python src/ml/train.py` first.")
            else:
                start_t = time.perf_counter()
                
                curr_amount = float(amount)
                avg_30d = float(card_features["avg_amount_30d"])
                max_30d = float(card_features["max_amount_30d"])
                
                online_features = dict(card_features)
                online_features["TransactionAmt"] = curr_amount
                online_features["amount_ratio_30d"] = curr_amount / (avg_30d + 1.0)
                online_features["is_amount_gt_30d_max"] = 1.0 if curr_amount > max_30d else 0.0

                feature_cols = model_pipeline.feature_names
                input_df = pd.DataFrame([online_features])[feature_cols]

                xgb_p = float(model_pipeline.xgb_model.predict_proba(input_df)[0, 1])
                lgb_p = float(model_pipeline.lgb_model.predict_proba(input_df)[0, 1])
                w_xgb = getattr(model_pipeline, "xgb_weight", 0.5)
                w_lgb = getattr(model_pipeline, "lgb_weight", 0.5)
                fraud_score = float(w_xgb * xgb_p + w_lgb * lgb_p)
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                decision_th = float(getattr(model_pipeline, "optimal_threshold", 0.5))
                is_fraud = fraud_score >= decision_th
                decision_str = "🚨 ALERT: FRAUD DETECTED" if is_fraud else "✅ APPROVED (LEGITIMATE)"
                status_class = "status-fraud" if is_fraud else "status-approved"

                st.markdown(f'<div class="{status_class}">{decision_str}</div>', unsafe_allow_html=True)
                st.markdown(f"**Fraud Risk Score:** `{fraud_score * 100:.2f}%` | **Cost-Optimal Threshold (θ):** `{decision_th * 100:.2f}%` | **SLA Latency:** `{latency_ms:.2f} ms`")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_score * 100,
                    title={'text': "Fraud Risk Score (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff1744" if is_fraud else "#00e676"},
                        'steps': [
                            {'range': [0, decision_th * 100], 'color': "rgba(0, 230, 118, 0.2)"},
                            {'range': [decision_th * 100, 100], 'color': "rgba(255, 23, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': decision_th * 100
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

# Tab 2: Model Performance & Cost Optimization Benchmark
with tab2:
    st.subheader("📊 Model Performance & Financial Cost Matrix Optimization")
    st.write("Comparative evaluation metrics and business cost minimization analysis using Dynamic Decision Threshold Tuning.")

    if eval_report:
        th_tuning = eval_report.get("threshold_tuning", {})
        if th_tuning:
            st.markdown("### 💰 Financial Cost Matrix Optimization Summary")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Optimal Threshold (θ*)", f"{th_tuning.get('optimal_threshold', 0.5):.4f}", "Cost-Tuned")
            kpi2.metric("Min Financial Loss ($)", f"${th_tuning.get('min_cost', 0.0):,.2f}")
            kpi3.metric("Loss @ Fixed 0.50 ($)", f"${th_tuning.get('cost_at_05', 0.0):,.2f}")
            kpi4.metric("Financial Savings", f"${th_tuning.get('savings_amount', 0.0):,.2f}", f"-{th_tuning.get('savings_pct', 0.0):.2f}% Loss")

        st.markdown("### 📈 Model Evaluation Metrics Comparison")
        metrics_dict = eval_report.get("metrics", {})
        rows = []
        for model_name, m_data in metrics_dict.items():
            if isinstance(m_data, dict) and "pr_auc" in m_data:
                rows.append({
                    "Model Setup": model_name.replace("_", " ").upper(),
                    "PR-AUC (Precision-Recall AUC)": m_data.get("pr_auc"),
                    "ROC-AUC Score": m_data.get("roc_auc"),
                    "F1-Score": m_data.get("f1_score"),
                    "Decision Threshold": m_data.get("threshold", 0.5)
                })
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True)

        # Visualizations from models directory
        cost_plot_path = Path(settings.model_dir) / "cost_vs_threshold.png"
        tradeoff_plot_path = Path(settings.model_dir) / "threshold_tradeoffs.png"

        c1, c2 = st.columns(2)
        if cost_plot_path.is_file():
            with c1:
                st.markdown("#### 📉 Financial Cost Curve vs Threshold")
                st.image(str(cost_plot_path), use_container_width=True)
        if tradeoff_plot_path.is_file():
            with c2:
                st.markdown("#### 🎯 Metric Trade-offs vs Threshold")
                st.image(str(tradeoff_plot_path), use_container_width=True)
    else:
        st.warning(f"⚠️ Evaluation report JSON not found at `{REPORT_JSON_PATH}`. Run `python src/ml/train.py` to generate.")

# Tab 3: Redis Feature Store Inspector
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

# Tab 4: Evidently AI Data Drift Report
with tab4:
    st.subheader("🛡️ Evidently AI Feature Drift & Quality Report")
    st.write("Embedded interactive Data Drift report generated by Evidently AI engine.")

    if DRIFT_HTML_PATH.is_file():
        with open(DRIFT_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning(f"⚠️ Data Drift HTML Report not found at `{DRIFT_HTML_PATH}`. Run `python src/quality/feature_monitoring.py` to generate.")
