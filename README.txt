# AI Weather Prediction System — LSTM
## Step-by-Step VS Code Setup Guide

---

### STEP 1 — Open VS Code and Create Project Folder

1. Open VS Code
2. Click File → Open Folder → Create a new folder called: weather_ai → Open it
3. Open Terminal in VS Code: press Ctrl + ` (backtick)
4. In the terminal, type:
   mkdir data
   mkdir model

---

### STEP 2 — Copy All Project Files

Copy these 4 files into your weather_ai folder:
- fetch_data.py
- train.py
- app.py
- requirements.txt

Your folder should now look like:
weather_ai/
├── data/             (empty folder)
├── model/            (empty folder)
├── fetch_data.py
├── train.py
├── app.py
└── requirements.txt

---

### STEP 3 — Install Libraries

In the VS Code terminal, run:
   pip install -r requirements.txt

Wait for all libraries to finish installing (2-5 minutes).

---

### STEP 4 — Fetch Weather Data

In the terminal, run:
   python fetch_data.py

Expected output:
   Fetching weather data for Bengaluru...
   Done! Saved 6552 rows to data/weather_history.csv

Check that data/weather_history.csv was created.

---

### STEP 5 — Train the LSTM Model

In the terminal, run:
   python train.py

You will see:
   Epoch 1/20 ...
   Epoch 2/20 ...
   ...
   Epoch 20/20 ...
   Training complete!

This takes 3-5 minutes. Loss value should decrease each epoch.
Model files will be saved in the model/ folder.

---

### STEP 6 — Run the Prediction

In the terminal, run:
   python app.py

You will see predictions printed in the terminal:
   Current Temp  : 27.3°C
   Predicted +1h : 27.8°C
   Predicted +2h : 28.1°C
   ...

A graph window will pop up showing actual vs predicted temperatures.
The graph is also saved as prediction_output.png.

---

### FOR CLASS DEMO

On presentation day:
1. Open VS Code
2. Open Terminal (Ctrl + `)
3. Run: python app.py
4. Show the graph that pops up — that is your live AI prediction!

No need to retrain. The saved model gives instant results.
