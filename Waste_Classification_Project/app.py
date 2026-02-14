import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Waste AI System",
    page_icon="♻️",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        h1, h2, h3 { color: white; }
        .stButton>button {
            background-color: #00C897;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .card {
            background-color: #1C1F26;
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
        }
        .small-note {
            color: #B0B3B8;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("waste_model.pkl")

# ---------------- HEADER ----------------
st.title("♻️ Smart Waste Classification & Sustainability Dashboard")
st.markdown("AI-powered Environmental Decision Support System")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    weight_g = st.number_input("Weight (grams)", min_value=0.0, step=10.0)

with col2:
    size = st.selectbox("Size", ["Small", "Medium", "Large"])

with col3:
    texture = st.selectbox("Texture", ["Smooth", "Rough"])

with col4:
    color = st.selectbox("Color Category",
                         ["Plastic", "Organic", "Metal", "Paper"])

size_map = {"Small": 1, "Medium": 2, "Large": 3}
texture_map = {"Smooth": 1, "Rough": 2}
color_map = {"Plastic": 1, "Organic": 2, "Metal": 3, "Paper": 4}

# ---------------- WASTE INFORMATION ----------------
waste_info = {
    "Plastic": {
        "bin": "Blue Bin",
        "decompose": "450+ years",
        "impact": "Major contributor to ocean pollution and microplastics.",
        "tip": "Reduce single-use plastics and recycle properly."
    },
    "Organic": {
        "bin": "Green Bin",
        "decompose": "2-6 weeks",
        "impact": "Biodegradable and improves soil quality.",
        "tip": "Use composting to convert into fertilizer."
    },
    "Metal": {
        "bin": "Yellow Bin",
        "decompose": "100+ years",
        "impact": "Highly recyclable but energy-intensive to produce.",
        "tip": "Clean metal items before recycling."
    },
    "Paper": {
        "bin": "Blue Bin",
        "decompose": "2-6 weeks",
        "impact": "Recyclable but excessive use impacts forests.",
        "tip": "Keep paper dry before recycling."
    }
}

# ---------------- SUSTAINABILITY DATA ----------------
risk_data = {
    "Plastic": ("High", 3, "Reduce usage and recycle responsibly."),
    "Organic": ("Low", 9, "Compost to improve soil health."),
    "Metal": ("Medium", 6, "Recycle to conserve energy."),
    "Paper": ("Medium", 7, "Reuse or recycle dry paper.")
}

# ---------------- CARBON FOOTPRINT FACTORS (ESTIMATES) ----------------
# Units: kg CO2e per kg of waste
# "disposal" ~ landfill/incineration average impact estimate
# "better_option" ~ recycling/composting estimate
carbon_factors = {
    "Plastic": {"disposal": 2.70, "better_option": 1.20, "better_label": "Recycling"},
    "Organic": {"disposal": 1.90, "better_option": 0.20, "better_label": "Composting"},
    "Metal":   {"disposal": 1.60, "better_option": 0.40, "better_label": "Recycling"},
    "Paper":   {"disposal": 1.00, "better_option": 0.30, "better_label": "Recycling"},
}

def kg_from_grams(g: float) -> float:
    return g / 1000.0

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Waste Type"):

    sample = pd.DataFrame([[weight_g,
                            size_map[size],
                            texture_map[texture],
                            color_map[color]]],
                          columns=["Weight", "Size", "Texture", "Color"])

    result = model.predict(sample)[0]

    st.success(f"Predicted Waste Type: {result}")

    info = waste_info[result]
    risk_level, score, advice = risk_data[result]

    st.divider()

    # ---------- GUIDANCE CARDS ----------
    st.subheader("🗑️ Disposal Guidance")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
        <div class="card">
        <h3>Recommended Bin</h3>
        <h2>{info['bin']}</h2>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
        <div class="card">
        <h3>Decomposition Time</h3>
        <h2>{info['decompose']}</h2>
        </div>
        """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="card">
        <h3>Environmental Risk</h3>
        <h2>{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------- IMPACT & SCORE ----------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("🌍 Environmental Impact")
        st.info(info["impact"])

    with colB:
        st.subheader("♻️ Sustainability Score")
        st.metric("Score (Out of 10)", score)
        st.markdown("### Recommended Action")
        st.warning(advice)

    # ---------- CARBON FOOTPRINT CALCULATOR ----------
    st.divider()
    st.subheader("🌫️ Carbon Footprint Calculator (Estimated CO₂e)")

    weight_kg = kg_from_grams(weight_g)
    factors = carbon_factors.get(result)

    if factors is None:
        st.write("Carbon factors not available for this category.")
    else:
        disposal = weight_kg * factors["disposal"]
        better = weight_kg * factors["better_option"]
        savings = max(disposal - better, 0.0)

        cf1, cf2, cf3 = st.columns(3)
        cf1.metric("Estimated CO₂e (If Disposed)", f"{disposal:.2f} kg")
        cf2.metric(f"Estimated CO₂e (If {factors['better_label']})", f"{better:.2f} kg")
        cf3.metric("Estimated CO₂e Saved", f"{savings:.2f} kg")

        st.markdown(
            f"<div class='small-note'>*Estimates based on simplified emission factors "
            f"(kg CO₂e per kg waste). Real values vary by location and processing method.*</div>",
            unsafe_allow_html=True
        )

    # ---------- PIE CHART (AI CONFIDENCE) ----------
    st.divider()
    st.subheader("📊 AI Prediction Confidence Distribution")

    probabilities = model.predict_proba(sample)[0]
    categories = model.classes_

    df_prob = pd.DataFrame({
        "Waste Category": categories,
        "Confidence": probabilities
    })

    fig = px.pie(
        df_prob,
        names="Waste Category",
        values="Confidence",
        title="Model Confidence for Each Waste Category",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)
