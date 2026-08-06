import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Sample time series
data = pd.Series([
    112, 118, 132, 129, 121,
    135, 148, 148, 136, 119,
    104, 118
])

# Create model
model = SARIMAX(
    data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

# Train model
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=5)

print(forecast)
