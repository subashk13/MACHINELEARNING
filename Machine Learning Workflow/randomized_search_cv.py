from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Parameter distribution
params = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, None]
}

# Create model
model = RandomForestClassifier(random_state=42)

# Random Search
search = RandomizedSearchCV(
    model,
    params,
    n_iter=4,
    cv=5,
    random_state=42
)

search.fit(X, y)

print(search.best_params_)
print(search.best_score_)
