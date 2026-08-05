from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Create model
model = OPTICS(min_samples=5)

# Train model
labels = model.fit_predict(X)

# Evaluation
if len(set(labels)) > 1 and -1 not in set(labels):
    print("Silhouette Score:", silhouette_score(X, labels))
else:
    print("Silhouette Score cannot be calculated.")
    