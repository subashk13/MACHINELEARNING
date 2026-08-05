#Requires: pip install umap-learn
from sklearn.datasets import load_iris
import umap

# Load dataset
X, _ = load_iris(return_X_y=True)

# Create model
model = umap.UMAP(
    n_components=2,
    random_state=42
)

# Transform data
X_umap = model.fit_transform(X)

# Results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_umap.shape)