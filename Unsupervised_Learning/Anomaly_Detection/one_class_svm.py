from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=1,
    cluster_std=0.60,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create model
model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

# Train model
model.fit(X)

# Predict
labels = model.predict(X)

# Results
print("Normal Points:", (labels == 1).sum())
print("Outliers:", (labels == -1).sum())
