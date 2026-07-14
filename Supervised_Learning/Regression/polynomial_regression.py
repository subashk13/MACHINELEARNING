from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=15,
    random_state=42
)

# Transform features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
