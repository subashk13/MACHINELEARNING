from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create dataset
X, _ = make_blobs(
    n_samples=100,
    centers=3,
    n_features=2,
    random_state=42
)

# Create model
model = KMeans(
    n_clusters=3,
    random_state=42
)

# Train model
model.fit(X)

# Predict cluster labels
labels = model.labels_

# Evaluation
print("Cluster Centers:")
print(model.cluster_centers_)
print("Silhouette Score:", silhouette_score(X, labels))
