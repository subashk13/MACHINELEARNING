#Requires: pip install shap
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# SHAP Values
shap_values = explainer.shap_values(X)

print(shap_values)
