from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Validation Curve
param_range = [0.1, 1, 10, 100]

train_scores, test_scores = validation_curve(
    SVC(),
    X,
    y,
    param_name="C",
    param_range=param_range,
    cv=5
)

print("Training Scores:\n", train_scores)
print("Validation Scores:\n", test_scores)
