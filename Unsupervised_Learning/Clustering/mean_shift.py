from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Create model
model = MeanShift()

# Train model
labels = model.fit_predict(X)

# Evaluation
print("Silhouette Score:", silhouette_score(X, labels))
