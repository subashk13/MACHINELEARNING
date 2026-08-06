from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
selector = VarianceThreshold(threshold=0.2)

# Transform data
X_selected = selector.fit_transform(X)

# Results
print("Original Shape:", X.shape)
print("Selected Shape:", X_selected.shape)
