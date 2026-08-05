from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = PCA(n_components=2)

# Transform data
X_pca = model.fit_transform(X)

# Results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", model.explained_variance_ratio_)
