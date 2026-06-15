import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 12.97,
    "longitude": 77.59,
    "start_date": "2024-01-01",
    "end_date": "2024-09-30",
    "hourly": "temperature_2m,relativehumidity_2m,precipitation,windspeed_10m",
    "timezone": "Asia/Kolkata"
}

print("Fetching weather data for Bengaluru...")
r = requests.get(url, params=params)
data = r.json()["hourly"]
df = pd.DataFrame(data)
df.to_csv("data/weather_history.csv", index=False)
print(f"Done! Saved {len(df)} rows to data/weather_history.csv")
