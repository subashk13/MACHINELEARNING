from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Create model
model = LogisticRegression()

# Cross Validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores)
print("Average Accuracy:", scores.mean())
