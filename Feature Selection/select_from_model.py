from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Base model
model = RandomForestClassifier(random_state=42)

# Create selector
selector = SelectFromModel(model)

# Transform data
X_selected = selector.fit_transform(X, y)

# Results
print("Original Shape:", X.shape)
print("Selected Shape:", X_selected.shape)
