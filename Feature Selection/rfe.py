from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Base model
estimator = LogisticRegression(max_iter=1000)

# Create model
selector = RFE(
    estimator=estimator,
    n_features_to_select=2
)

# Transform data
X_selected = selector.fit_transform(X, y)

# Results
print("Original Shape:", X.shape)
print("Selected Shape:", X_selected.shape)
