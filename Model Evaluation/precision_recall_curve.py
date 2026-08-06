from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# Create dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probability
y_prob = model.predict_proba(X_test)[:, 1]

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

print("Precision:", precision)
print("Recall:", recall)
