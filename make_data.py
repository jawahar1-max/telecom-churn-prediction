# make_data.py
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 2000
age = np.random.randint(18, 80, size=n)
call_duration = np.round(np.random.exponential(scale=200, size=n), 1)  # minutes per month
internet_usage = np.round(np.random.normal(loc=50, scale=30, size=n), 1)  # GB per month
internet_usage = np.clip(internet_usage, 0, None)
complaints = np.random.poisson(lam=0.3, size=n)
monthly_charges = np.round(20 + 0.05*call_duration + 0.2*internet_usage + np.random.normal(0,5,size=n), 2)

# churn probability model (synthetic)
logit = -3.5 + 0.01*(age-40) + 0.003*call_duration - 0.02*internet_usage + 0.8*complaints + 0.02*(monthly_charges-50)
prob = 1/(1+np.exp(-logit))
churn = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(n)],
    "age": age,
    "call_duration": call_duration,
    "internet_usage": internet_usage,
    "complaints": complaints,
    "monthly_charges": monthly_charges,
    "churn": churn
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/telecom_churn_sample.csv", index=False)
print("Saved data/telecom_churn_sample.csv — rows:", len(df))