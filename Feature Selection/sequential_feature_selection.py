from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Base model
model = LogisticRegression(max_iter=1000)

# Create selector
selector = SequentialFeatureSelector(
    model,
    n_features_to_select=2
)

# Transform data
X_selected = selector.fit_transform(X, y)

# Results
print("Original Shape:", X.shape)
print("Selected Shape:", X_selected.shape)
