from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Permutation Importance
result = permutation_importance(
    model,
    X,
    y,
    random_state=42
)

print(result.importances_mean)
