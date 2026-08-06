from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
selector = SelectKBest(
    score_func=f_classif,
    k=2
)

# Transform data
X_selected = selector.fit_transform(X, y)

# Results
print("Original Shape:", X.shape)
print("Selected Shape:", X_selected.shape)
