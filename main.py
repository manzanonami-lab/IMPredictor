from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lottery Prediction API")

# Allow all apps to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
df = pd.read_csv("draws.csv")
numbers = df["number"].tolist()

# Odd/Even prediction
def predict_parity(nums):
    last10 = nums[-10:] if len(nums) >= 10 else nums
    odd = sum(1 for n in last10 if n % 2 != 0)
    even = len(last10) - odd
    odd_prob = odd / len(last10)
    even_prob = even / len(last10)
    prediction = "Even" if even_prob >= odd_prob else "Odd"
    return prediction, odd_prob, even_prob

# Hot numbers
def hot_numbers(nums):
    freq = {i:0 for i in range(1,81)}
    for n in nums:
        freq[n] += 1

    gap = {}
    for i in range(1,81):
        if i in nums:
            gap[i] = len(nums) - nums[::-1].index(i)
        else:
            gap[i] = len(nums)

    # Transition matrix
    transition = np.zeros((81,81))
    for i in range(len(nums)-1):
        prev = nums[i]
        nxt = nums[i+1]
        transition[prev][nxt] += 1

    last = nums[-1]
    trans_prob = transition[last]

    score = {}
    for i in range(1,81):
        score[i] = 0.4*freq[i] + 0.3*gap[i] + 0.3*trans_prob[i]

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    top3 = [n[0] for n in ranked[:3]]
    return top3

@app.get("/predict")
def predict():
    parity, odd_p, even_p = predict_parity(numbers)
    top3 = hot_numbers(numbers)
    return {
        "parity_prediction": parity,
        "odd_probability": round(odd_p,2),
        "even_probability": round(even_p,2),
        "top3_numbers": top3
    }
