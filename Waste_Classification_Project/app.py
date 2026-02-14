import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Smart Waste Dashboard", layout="wide")

# ------------------------------
# LOAD MODEL (CLOUD SAFE)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "waste_model.pkl")
model = joblib.load(MODEL_PATH)

# ------------------------------
# MAPS (UI labels)
# ------------------------------
SIZE_MAP = {"Small": 1, "Medium": 2, "Large": 3}
TEXTURE_MAP = {"Smooth": 1, "Rough": 2}
COLOR_MAP = {"Plastic": 1, "Organic": 2, "Metal": 3, "Paper": 4}

# Output guidance
DISPOSAL_GUIDE = {
    "Plastic": {"bin": "Dry / Plastic bin", "decomp": "450+ years", "risk": "High"},
    "Organic": {"bin": "Wet / Compost bin", "decomp": "2–6 months", "risk": "Low"},
    "Metal":   {"bin": "Dry / Metal bin", "decomp": "50–200 years", "risk": "Medium"},
    "Paper":   {"bin": "Dry / Paper bin", "decomp": "2–6 weeks", "risk": "Low"},
}

# Simplified emission factors (kg CO2e per kg waste)
EMISSION_DISPOSE = {"Plastic": 2.7, "Organic": 0.6, "Metal": 1.9, "Paper": 1.1}
EMISSION_RECYCLE = {"Plastic": 1.2, "Organic": 0.2, "Metal": 0.7, "Paper": 0.4}

# ------------------------------
# TITLE
# ------------------------------
st.markdown(
    "## ♻️ Smart Waste Classification & Sustainability Dashboard\n"
    "AI-powered Environmental Decision Support System"
)
st.divider()

# ------------------------------
# INPUTS (Top row layout)
# ------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    weight_g = st.number_input("Weight (grams)", min_value=0.0, value=500.0, step=10.0)

with c2:
    size_label = st.selectbox("Size", list(SIZE_MAP.keys()), index=0)

with c3:
    texture_label = st.selectbox("Texture", list(TEXTURE_MAP.keys()), index=0)

with c4:
    color_label = st.selectbox("Color Category", list(COLOR_MAP.keys()), index=0)

st.write("")
predict_btn = st.button("🔍 Predict Waste Type", use_container_width=False)

# ------------------------------
# PREDICT + DASHBOARD
# ------------------------------
if predict_btn:
    # Build input with correct feature names
    X = pd.DataFrame([{
        "Weight": float(weight_g),
        "Size": SIZE_MAP[size_label],
        "Texture": TEXTURE_MAP[texture_label],
        "Color": COLOR_MAP[color_label]
    }])

    pred = model.predict(X)[0]

    probs = None
    classes = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_

    st.success(f"✅ Predicted Waste Type: **{pred}**")
    st.divider()

    # ------------------------------
    # DISPOSAL GUIDANCE
    # ------------------------------
    guide = DISPOSAL_GUIDE.get(str(pred), {"bin": "General bin", "decomp": "N/A", "risk": "N/A"})
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("### 🗑️ Recommended Bin")
        st.info(guide["bin"])

    with g2:
        st.markdown("### ⏳ Decomposition Time")
        st.info(guide["decomp"])

    with g3:
        st.markdown("### ⚠️ Environmental Risk")
        st.info(guide["risk"])

    st.divider()

    # ------------------------------
    # CARBON FOOTPRINT (Disposed / Recycling / Saved)
    # ------------------------------
    st.markdown("### 🌍 Carbon Footprint Calculator (Estimated CO₂e)")

    kg = float(weight_g) / 1000.0
    dispose_factor = EMISSION_DISPOSE.get(str(pred), 1.0)
    recycle_factor = EMISSION_RECYCLE.get(str(pred), 0.5)

    disposed = kg * dispose_factor
    recycled = kg * recycle_factor
    saved = max(disposed - recycled, 0)

    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        st.metric("Estimated CO₂e (If Disposed)", f"{disposed:.2f} kg")
    with cf2:
        st.metric("Estimated CO₂e (If Recycling)", f"{recycled:.2f} kg")
    with cf3:
        st.metric("Estimated CO₂e Saved", f"{saved:.2f} kg")

    st.caption(
        "*Estimates use simplified emission factors (kg CO₂e per kg waste). "
        "Real values vary by location and processing method.*"
    )

    st.divider()

    # ------------------------------
    # PIE CHART (dynamic using model confidence)
    # ------------------------------
    st.markdown("### 📊 AI Prediction Confidence Distribution")

    if probs is not None and classes is not None:
        conf_df = pd.DataFrame({
            "Waste_Type": [str(c) for c in classes],
            "Confidence": [float(p) for p in probs]
        })

        fig = px.pie(
            conf_df,
            names="Waste_Type",
            values="Confidence",
            title="Model Confidence for Each Waste Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("This model does not provide prediction probabilities, so the pie chart cannot change dynamically.")

# ------------------------------
# FOOTER (Option 3 - Your Name)
# ------------------------------
st.markdown("---")
st.caption("© 2026 Srivarsha M | Smart Waste Classification & Sustainability Dashboard")
