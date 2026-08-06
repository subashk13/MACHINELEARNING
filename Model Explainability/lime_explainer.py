#Requires: pip install lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# LIME Explainer
explainer = LimeTabularExplainer(
    X,
    feature_names=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ],
    class_names=["Setosa", "Versicolor", "Virginica"],
    mode="classification"
)

# Explain one instance
explanation = explainer.explain_instance(
    X[0],
    model.predict_proba
)

print(explanation.as_list())