from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature Importance
importances = model.feature_importances_

print(importances)
