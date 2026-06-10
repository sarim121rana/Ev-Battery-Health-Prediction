import streamlit as st
import pickle
import numpy as np

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="EV Battery Health Predictor",
    layout="centered"
)

st.title("🔋 Electric Vehicle Battery Health Predictor")
st.write(
    "NASA Li-ion Sensor Data par trained AI Model jo realtime me battery ki remaining capacity batayega."
)
st.markdown("---")

# =========================
# Load Trained Model
# =========================
@st.cache_resource
def load_model():
    with open("ev_battery_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.success("✅ AI Model successfully loaded!")
except Exception as e:
    st.error(f"❌ Model load karne me dikkat aayi: {e}")
    st.stop()

# =========================
# User Inputs
# =========================
st.markdown("### 📊 Enter Sensor Readings:")

cycle_num = st.slider(
    "🔄 Cycle Number (Kitni baar use hui)",
    min_value=1,
    max_value=1000,
    value=2
)

voltage = st.slider(
    "⚡ Voltage Measured (V)",
    min_value=0.0,
    max_value=5.0,
    value=3.6,
    step=0.1
)

current = st.slider(
    "🔌 Current Measured (A)",
    min_value=-5.0,
    max_value=5.0,
    value=-2.0,
    step=0.1
)

temperature = st.slider(
    "🌡️ Temperature Measured (°C)",
    min_value=10.0,
    max_value=70.0,
    value=44.0,
    step=0.5
)

st.markdown("---")

# =========================
# Prediction Section
# =========================
if st.button("🔮 Predict Battery Health", type="primary"):

    try:
        # Input Features
        input_features = np.array([
            [cycle_num, voltage, current, temperature]
        ])

        # Prediction
        predicted_capacity = float(model.predict(input_features)[0])

        # Original Capacity
        original_capacity = 1.85

        # SoH Calculation
        soh_percentage = (predicted_capacity / original_capacity) * 100

        # Progress Bar ke liye safe value
        progress_value = max(0, min(int(soh_percentage), 100))

        # Results
        st.markdown("### 🏆 Prediction Results:")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Capacity",
                value=f"{predicted_capacity:.4f} Ah"
            )

        with col2:
            st.metric(
                label="State of Health (SoH)",
                value=f"{soh_percentage:.2f}%"
            )

        # Progress Bar
        st.progress(progress_value)

        # Health Status
        if soh_percentage > 80:
            st.success("🍏 Battery ekdum badhiya condition me hai!")

        elif soh_percentage > 60:
            st.warning("⚠️ Battery dheere-dheere degrade ho rahi hai.")

        else:
            st.error("🚨 Danger! Battery badalne ka waqt aa gaya hai.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
