import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X_train)

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X_train.flatten() + b

    dm = (-2 / n) * np.sum(X_train.flatten() * (y_train - y_pred))
    db = (-2 / n) * np.sum(y_train - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

# Predictions
y_pred = m * X_test.flatten() + b

# Evaluation
print("Slope (m):", m)
print("Intercept (b):", b)
print("MSE:", mean_squared_error(y_test, y_pred))
