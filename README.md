🌤️ AI-Based Smart Cooling System

An AI-powered weather forecasting and smart cooling decision system that uses a Deep Learning (LSTM) model to predict short-term temperature, humidity, and wind trends for Bengaluru, and automatically recommends whether cooling should be turned ON or OFF.

📊 Project Overview


Goal: Forecast temperature up to 6 hours ahead using historical weather sequences and trigger smart cooling decisions based on predicted trends
Data Source: 6,500+ hourly weather records (temperature, humidity, wind speed, precipitation) fetched live via the Open-Meteo Historical Weather API for Bengaluru
Model: 2-layer Stacked LSTM (Long Short-Term Memory) network with Dropout regularization
Sequence Length: 24-hour rolling window used to predict the next hour's temperature (recursively extended to 6-hour forecasts)
Deployment: Interactive multi-tab Streamlit dashboard


✨ Features


🔍 One-click AI prediction — runs a 6-hour recursive temperature forecast on demand
⚙️ Smart Cooling Decision Engine — automatically flags cooling ON/OFF based on predicted temperature direction
📊 Live Graphs — actual vs. predicted temperature, humidity trends over the last 72 hours
🌡️ Detailed Summary Tab — hour-by-hour predicted temperatures with total change metric
🧠 Smart Insights — automatic detection of highest/lowest predicted temperature and temperature zone classification (Low/Medium/High)


🛠️ Tech Stack

CategoryToolsLanguagePythonDeep LearningTensorFlow / Keras (LSTM)Data ProcessingPandas, NumPy, Scikit-learn (MinMaxScaler)VisualizationMatplotlib, Streamlit native chartsWeb AppStreamlitData SourceOpen-Meteo Historical Weather APIModel PersistenceJoblib, HDF5 (.h5)

🧠 How It Works


fetch_data.py — Pulls hourly historical weather data (temperature, humidity, wind speed, precipitation) for Bengaluru via the Open-Meteo API and saves it as a CSV
train.py — Scales features with MinMaxScaler, builds 24-hour input sequences, and trains a 2-layer LSTM model (64 → 32 units with Dropout) to predict the next hour's temperature
app.py — Loads the trained model and scaler, recursively forecasts the next 6 hours, visualizes results in an interactive Streamlit dashboard, and generates a smart cooling recommendation


