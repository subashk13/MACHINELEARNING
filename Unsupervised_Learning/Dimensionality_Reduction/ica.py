from sklearn.datasets import load_iris
from sklearn.decomposition import FastICA

# Load dataset
X, _ = load_iris(return_X_y=True)

# Create model
model = FastICA(
    n_components=2,
    random_state=42
)

# Transform data
X_ica = model.fit_transform(X)

# Results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_ica.shape)
