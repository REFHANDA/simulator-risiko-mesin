import streamlit as st
import numpy as np
import joblib

# ==========================
# Load model dan scaler
# ==========================
model = joblib.load("model_risiko_v1.joblib")
scaler = joblib.load("scaler_risiko_v1.joblib")

st.set_page_config(page_title="Simulator Risiko Mesin", page_icon="⚙️")

st.title("⚙️ Simulator Risiko Kegagalan Sistem")
st.write("Prediksi risiko berdasarkan suhu dan getaran mesin.")

# ==========================
# Input User
# ==========================
suhu = st.number_input("Masukkan Suhu Mesin (°C)", value=80.0)
getaran = st.number_input("Masukkan Getaran Mesin", value=5.0)

# ==========================
# Monitoring Drift
# ==========================
if suhu > 120 or suhu < 10:
    st.warning("⚠️ Input suhu berada di luar data pelatihan. Hasil prediksi mungkin kurang akurat.")

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi Risiko"):

    data = np.array([[suhu, getaran]])

    data_scaled = scaler.transform(data)

    hasil = model.predict(data_scaled)

    st.success(f"Skor Risiko = {hasil[0]:.2f}")