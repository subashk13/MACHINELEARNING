from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Parameter grid
params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

# Create model
model = SVC()

# Grid Search
grid = GridSearchCV(
    model,
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)
