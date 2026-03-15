from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(title="Smart Lottery Predictor")

DATA_FILE = "draws.csv"

# -----------------------------
# User input model
# -----------------------------
class DrawInput(BaseModel):
    timestamp: str
    issue: str
    number: int
    result: str  # "Odd" or "Even"

# -----------------------------
# Ensure dataset exists
# -----------------------------
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["timestamp","issue","number","result"])
    df.to_csv(DATA_FILE, index=False)

# -----------------------------
# Add new draw result
# -----------------------------
@app.post("/add-result")
def add_result(draw: DrawInput):
    df = pd.read_csv(DATA_FILE)

    # Prevent duplicate issue entries
    if draw.issue in df['issue'].values:
        return {"message":"Issue already exists, no changes made."}

    new_row = {
        "timestamp": draw.timestamp,
        "issue": draw.issue,
        "number": draw.number,
        "result": draw.result
    }

    df.loc[len(df)] = new_row
    df.to_csv(DATA_FILE, index=False)

    return {"message": "Result added successfully", "latest_number": draw.number}

# -----------------------------
# Prediction engine
# -----------------------------
@app.get("/predict")
def predict():
    df = pd.read_csv(DATA_FILE)
    numbers = df["number"].tolist()
    
    if len(numbers) < 5:
        return {"message": "Not enough data yet for prediction."}

    # -----------------------------
    # Odd/Even probability
    # -----------------------------
    last10 = numbers[-10:]
    odd = sum(1 for n in last10 if n % 2 != 0)
    even = len(last10) - odd
    odd_prob = odd / len(last10)
    even_prob = even / len(last10)
    parity_prediction = "Even" if even_prob >= odd_prob else "Odd"

    # -----------------------------
    # Hot numbers (frequent + recent)
    # -----------------------------
    freq = {i:0 for i in range(1,81)}
    for n in numbers:
        freq[n] += 1

    # Gap (overdue numbers)
    gap = {}
    for i in range(1,81):
        if i in numbers:
            gap[i] = len(numbers) - numbers[::-1].index(i)
        else:
            gap[i] = len(numbers)

    # Scoring hot numbers
    score = {}
    for i in range(1,81):
        score[i] = 0.6*freq[i] + 0.4*gap[i]

    ranked = sorted(score.items(), key=lambda x:x[1], reverse=True)
    top3 = [n[0] for n in ranked[:3]]

    # Cold numbers (least recent / low frequency)
    cold_ranked = sorted(score.items(), key=lambda x:x[1])
    cold3 = [n[0] for n in cold_ranked[:3]]

    # -----------------------------
    # Streak detection
    # -----------------------------
    streak_odd = streak_even = max_streak_odd = max_streak_even = 0
    for res in df['result']:
        if res == "Odd":
            streak_odd +=1
            streak_even = 0
        else:
            streak_even +=1
            streak_odd = 0
        max_streak_odd = max(max_streak_odd, streak_odd)
        max_streak_even = max(max_streak_even, streak_even)

    return {
        "parity_prediction": parity_prediction,
        "odd_probability": round(odd_prob,2),
        "even_probability": round(even_prob,2),
        "top3_numbers": top3,
        "cold3_numbers": cold3,
        "max_odd_streak": max_streak_odd,
        "max_even_streak": max_streak_even
    }

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Smart Lottery Predictor API running. Use /predict or /add-result"}
