from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Create model
model = LogisticRegression()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=5
)

print("Train Sizes:\n", train_sizes)
print("Training Scores:\n", train_scores)
print("Validation Scores:\n", test_scores)
