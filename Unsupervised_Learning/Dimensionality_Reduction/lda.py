from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = LinearDiscriminantAnalysis(n_components=2)

# Transform data
X_lda = model.fit_transform(X, y)

# Results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_lda.shape)
