from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=1,
    cluster_std=0.60,
    random_state=42
)

# Create model
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

# Predict
labels = model.fit_predict(X)

# Results
print("Normal Points:", (labels == 1).sum())
print("Outliers:", (labels == -1).sum())
