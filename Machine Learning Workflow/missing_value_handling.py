import pandas as pd
from sklearn.impute import SimpleImputer

# Sample dataset
df = pd.DataFrame({
    "Age": [25, 30, None, 40, None],
    "Salary": [50000, None, 60000, 70000, 65000]
})

# Create imputer
imputer = SimpleImputer(strategy="mean")

# Handle missing values
df[:] = imputer.fit_transform(df)

print(df)
