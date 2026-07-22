from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Create dataset
X, _ = make_blobs(
    n_samples=100,
    centers=3,
    n_features=2,
    random_state=42
)

# Create model
model = AgglomerativeClustering(n_clusters=3)

# Fit and predict cluster labels
labels = model.fit_predict(X)

# Evaluation
print("Cluster Labels:")
print(labels)
print("Silhouette Score:", silhouette_score(X, labels))
