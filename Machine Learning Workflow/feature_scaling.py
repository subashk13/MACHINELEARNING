from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create dataset
X, _ = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Create scaler
scaler = StandardScaler()

# Scale features
X_scaled = scaler.fit_transform(X)

print(X_scaled)
