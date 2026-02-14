import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ==============================
# LOAD MODEL SAFELY (CLOUD FIX)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "waste_model.pkl")

model = joblib.load(MODEL_PATH)

# ==============================
# PAGE TITLE
# ==============================

st.title("♻️ Smart Waste Classification Dashboard")

st.write("AI-based waste prediction with visualization & carbon footprint calculator.")

# ==============================
# USER INPUT
# ==============================

st.sidebar.header("Enter Waste Details")

weight = st.sidebar.number_input("Weight", min_value=0)
size = st.sidebar.selectbox("Size", [1, 2, 3])
texture = st.sidebar.selectbox("Texture", [1, 2])
color = st.sidebar.selectbox("Color", [1, 2, 3, 4])

# ==============================
# PREDICTION
# ==============================

if st.sidebar.button("Predict Waste Type"):

    input_data = pd.DataFrame({
        "Weight": [weight],
        "Size": [size],
        "Texture": [texture],
        "Color": [color]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Waste Type: {prediction}")

    # ==============================
    # PIE CHART VISUALIZATION
    # ==============================

    waste_data = pd.DataFrame({
        "Waste_Type": ["Plastic", "Organic", "Metal", "Paper"],
        "Count": [2, 3, 2, 3]
    })

    fig = px.pie(
        waste_data,
        names="Waste_Type",
        values="Count",
        title="Waste Distribution Overview"
    )

    st.plotly_chart(fig)

    # ==============================
    # CARBON FOOTPRINT CALCULATOR
    # ==============================

    st.subheader("🌍 Carbon Footprint Estimation")

    carbon_values = {
        "Plastic": 6,
        "Organic": 2,
        "Metal": 5,
        "Paper": 3
    }

    carbon_output = carbon_values.get(prediction, 0) * weight / 100

    st.info(f"Estimated Carbon Impact Score: {carbon_output:.2f}")
