import pandas as pd
from prophet import Prophet

# Sample dataset
df = pd.DataFrame({
    "ds": pd.date_range(
        start="2024-01-01",
        periods=30
    ),
    "y": [
        10, 12, 13, 15, 16,
        18, 20, 21, 22, 24,
        23, 25, 27, 28, 30,
        32, 31, 33, 35, 36,
        38, 39, 40, 42, 44,
        43, 45, 47, 48, 50
    ]
})

# Create model
model = Prophet()

# Train model
model.fit(df)

# Future dates
future = model.make_future_dataframe(
    periods=7
)

# Forecast
forecast = model.predict(future)

print(forecast[["ds", "yhat"]].tail())
