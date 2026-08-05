from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Load dataset
X, _ = load_iris(return_X_y=True)

# Create model
model = TSNE(
    n_components=2,
    random_state=42
)

# Transform data
X_tsne = model.fit_transform(X)

# Results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_tsne.shape)
