from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=1,
    cluster_std=0.60,
    random_state=42
)

# Create model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

# Train model
model.fit(X)

# Predict
labels = model.predict(X)

# Results
print("Normal Points:", (labels == 1).sum())
print("Outliers:", (labels == -1).sum())
