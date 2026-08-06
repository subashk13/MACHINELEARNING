from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import ADASYN

# Create imbalanced dataset
X, y = make_classification(
    n_samples=500,
    n_features=5,
    weights=[0.9, 0.1],
    random_state=42
)

print("Before:", Counter(y))

# Create sampler
sampler = ADASYN(random_state=42)

# Balance dataset
X_resampled, y_resampled = sampler.fit_resample(X, y)

print("After:", Counter(y_resampled))
