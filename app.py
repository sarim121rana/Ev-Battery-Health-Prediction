 import streamlit as st
import pickle
import numpy as np
import pandas as pd

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
# Session State for History Tracking (Sliding Window Buffer)
# =========================
# Agar history state pehle se nahi bani hai, toh empty list initialize karein
if 'sensor_history' not in st.session_state:
    st.session_state.sensor_history = []

# Clear history button agar user naye siray se test karna chahe
if st.sidebar.button("🔄 Reset Sensor History Memory"):
    st.session_state.sensor_history = []
    st.sidebar.success("Memory cleared successfully!")

# Sidebar me live status dikhane ke liye
st.sidebar.markdown("### 🧠 Live Memory Buffer Status")
st.sidebar.write(f"Stored Readings: {len(st.session_state.sensor_history)} / 50")

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
        # Current slider values ko temporary session state memory me add karna
        st.session_state.sensor_history.append({
            'Voltage': voltage,
            'Temperature': temperature
        })
        
        # Buffer limit set karna: Sirf pichli 50 readings hi list me rahengi (Sliding Window concept)
        if len(st.session_state.sensor_history) > 50:
            st.session_state.sensor_history.pop(0)
            
        # Stored buffer ko DataFrame me convert karna taaki rolling metrics nikal saken
        history_df = pd.DataFrame(st.session_state.sensor_history)
        
        # Live rolling features calculate karna
        v_mean = float(history_df['Voltage'].mean())
        t_mean = float(history_df['Temperature'].mean())
        
        # Agar sirf 1 reading hai memory me, toh standard deviation 0 hoga
        v_std = float(history_df['Voltage'].std()) if len(history_df) > 1 else 0.0
        t_std = float(history_df['Temperature'].std()) if len(history_df) > 1 else 0.0
        
        # Handle NaN values safely agar calculations me koi issue aaye
        if np.isnan(v_std): v_std = 0.0
        if np.isnan(t_std): t_std = 0.0

        # Model pure 8 features dhoondh raha hai, toh input data isi strict order me banaya hai
        input_features = np.array([
            [cycle_num, voltage, current, temperature, v_mean, v_std, t_mean, t_std]
        ])

        # Prediction execution
        predicted_capacity = float(model.predict(input_features)[0])

        # Original new battery capacity baseline
        original_capacity = 1.85

        # SoH Calculation
        raw_soh = (predicted_capacity / original_capacity) * 100
        
        # Fix: SoH capping implementation jo ensure karega ki value strictly 0% se 100% ke beech rahe
        soh_percentage = min(raw_soh, 100.0)
        soh_percentage = max(soh_percentage, 0.0)

        # Progress Bar ke liye integer rendering format
        progress_value = int(soh_percentage)

        # =========================
        # Displaying Results
        # =========================
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

        # Progress Bar render
        st.progress(progress_value)

        # Health Status classification
        if soh_percentage > 80:
            st.success("🍏 Battery ekdum badhiya condition me hai!")
        elif soh_percentage > 60:
            st.warning("⚠️ Battery dheere-dheere degrade ho rahi hai.")
        else:
            st.error("🚨 Danger! Battery badalne ka waqt aa gaya hai.")
            
        # Optional: Model ke mathematical features breakdown interface me show karne ke liye
        with st.expander("🛠️ Engineered Features Summary (Model Input Insights)"):
            st.write(f"Calculated V_mean: {v_mean:.4f}")
            st.write(f"Calculated V_std (Fluctuation): {v_std:.4f}")
            st.write(f"Calculated T_mean: {t_mean:.2f}°C")
            st.write(f"Calculated T_std (Fluctuation): {t_std:.4f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
