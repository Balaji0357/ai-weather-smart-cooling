import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
df = pd.read_csv("data/weather_history.csv").dropna()
features = ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"]
print(f"Loaded {len(df)} rows of weather data")

# Scale the features between 0 and 1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# Create sequences: 24 hours of input -> predict next hour temperature
SEQ_LEN = 24
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][0])  # index 0 = temperature

X, y = np.array(X), np.array(y)
print(f"Created {len(X)} training sequences")

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

print("\nTraining LSTM model... (this takes 3-5 minutes)")
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Save model and scaler
model.save("model/lstm_weather.h5")
joblib.dump(scaler, "model/scaler.pkl")

final_loss = history.history['loss'][-1]
print(f"\nTraining complete! Final loss: {final_loss:.4f}")
print("Model saved to model/lstm_weather.h5")
print("Scaler saved to model/scaler.pkl")
