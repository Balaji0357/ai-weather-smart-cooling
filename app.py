import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

st.title("🌤️ AI-Based Smart Cooling System")
st.write("LSTM Prediction + Smart Cooling Decision")

# -------------------------------
# 🔹 Load Model + Data
# -------------------------------
@st.cache_resource
def load_all():
    model = load_model("model/lstm_weather.h5", compile=False)
    scaler = joblib.load("model/scaler.pkl")
    df = pd.read_csv("data/weather_history.csv").dropna()
    return model, scaler, df

model, scaler, df = load_all()

features = ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"]
scaled = scaler.transform(df[features])

# -------------------------------
# 🔹 Run Prediction
# -------------------------------
if st.button("🔍 Run AI Prediction"):

    predictions = []
    input_seq = scaled[-24:].copy()

    for _ in range(6):
        x = input_seq.reshape(1, 24, len(features))
        p = model.predict(x, verbose=0)[0][0]

        dummy = np.zeros((1, len(features)))
        dummy[0][0] = p
        real_temp = scaler.inverse_transform(dummy)[0][0]

        predictions.append(real_temp)

        new_row = input_seq[-1].copy()
        new_row[0] = p
        input_seq = np.vstack([input_seq[1:], new_row])

    current_temp = df["temperature_2m"].values[-1]

    # -------------------------------
    # 🔹 Cooling Decision
    # -------------------------------
    st.subheader("⚙️ Smart Cooling Decision")

    if predictions[0] > current_temp:
        st.error("Cooling ON 🔴 (Temperature increasing)")
    elif predictions[0] < current_temp:
        st.success("Cooling OFF 🟢 (Temperature decreasing)")
    else:
        st.info("Temperature Stable ⚪")

    # -------------------------------
    # 🔹 Tabs
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Graph", "🌡️ Summary", "📈 Trends"])

    # -------------------------------
    # 📊 Graph Tab
    # -------------------------------
    with tab1:
        actuals = df["temperature_2m"].values[-72:]
        actual_hours = list(range(len(actuals)))
        pred_hours = [len(actuals) + i for i in range(6)]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        fig.suptitle("AI-Based Weather Prediction (LSTM)", fontsize=14)

        # Temperature
        ax1 = axes[0]
        ax1.plot(actual_hours, actuals, color="blue", label="Actual Temp")
        ax1.plot(pred_hours, predictions, color="orange",
                 linestyle="--", marker="o", label="Predicted Temp")
        ax1.axvline(x=len(actuals)-1, color="red", linestyle=":", label="Now")

        ax1.set_title("Temperature Prediction")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("°C")
        ax1.legend()
        ax1.grid()

        # Humidity
        ax2 = axes[1]
        humidity = df["relativehumidity_2m"].values[-72:]
        ax2.plot(actual_hours, humidity, color="green", label="Humidity")
        ax2.set_title("Humidity")
        ax2.legend()
        ax2.grid()

        st.pyplot(fig)

    # -------------------------------
    # 🌡️ Summary Tab
    # -------------------------------
    with tab2:
        st.subheader("🌡️ Detailed Temperature Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Current", f"{current_temp:.1f}°C")
        col2.metric("+1 Hour", f"{predictions[0]:.1f}°C")
        col3.metric("+2 Hours", f"{predictions[1]:.1f}°C")
        col4.metric("+3 Hours", f"{predictions[2]:.1f}°C")

        st.markdown("---")

        col5, col6 = st.columns(2)
        col5.metric("+6 Hours", f"{predictions[5]:.1f}°C")
        col6.metric("Total Change", f"{predictions[5] - current_temp:.1f}°C")

    # -------------------------------
    # 📈 Trends Tab
    # -------------------------------
    with tab3:
        st.subheader("📈 Temperature Trends")

        st.line_chart(predictions)
        st.bar_chart(predictions)

        if predictions[0] > current_temp:
            st.warning("📈 Temperature Increasing → Cooling Needed")
        else:
            st.success("📉 Temperature Decreasing → Less Cooling")

    # -------------------------------
    # 🔹 Extra Insights
    # -------------------------------
    st.subheader("🧠 Smart Insights")

    max_temp = max(predictions)
    min_temp = min(predictions)

    st.write(f"🔥 Highest predicted temperature: {max_temp:.1f}°C")
    st.write(f"❄️ Lowest predicted temperature: {min_temp:.1f}°C")

    # Temperature Level Indicator
    st.subheader("🌡️ Temperature Level Indicator")

    if predictions[0] > 35:
        st.error("🔥 High Temperature Zone")
    elif predictions[0] > 28:
        st.warning("🌤 Medium Temperature Zone")
    else:
        st.success("❄️ Low Temperature Zone")