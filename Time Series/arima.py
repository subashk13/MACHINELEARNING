import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Sample time series
data = pd.Series([
    112, 118, 132, 129, 121,
    135, 148, 148, 136, 119,
    104, 118
])

# Create model
model = ARIMA(
    data,
    order=(2, 1, 2)
)

# Train model
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=5)

print(forecast)
