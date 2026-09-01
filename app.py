import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HR Employee Attrition & Compensation Equity Optimizer",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 HR Employee Attrition & Compensation Equity Optimizer")
st.markdown("**Domain**: `People Analytics / HR Tech ML` | **Tech Stack**: `XGBoost, Scikit-Learn, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    base_unit_cost = st.slider("Base Unit Cost", float(10.0), float(150.0), float(45.0))
    competitor_price_ratio = st.slider("Competitor Price Ratio", float(0.6), float(1.6), float(1.05))
    historical_demand_index = st.slider("Historical Demand Index", float(50.0), float(1200.0), float(500.0))
    promotional_lift_pct = st.slider("Promotional Lift Pct", float(0.0), float(0.5), float(0.12))
    inventory_coverage_days = st.slider("Inventory Coverage Days", int(3), int(60), int(25))
    macro_seasonality_factor = st.slider("Macro Seasonality Factor", float(0.5), float(1.8), float(1.0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"base_unit_cost": base_unit_cost, "competitor_price_ratio": competitor_price_ratio, "historical_demand_index": historical_demand_index, "promotional_lift_pct": promotional_lift_pct, "inventory_coverage_days": inventory_coverage_days, "macro_seasonality_factor": macro_seasonality_factor}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `will_convert_purchase`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
