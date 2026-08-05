from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Create dataset
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Create model
model = GaussianMixture(
    n_components=4,
    random_state=42
)

# Train model
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

# Evaluation
print("Silhouette Score:", silhouette_score(X, labels))
