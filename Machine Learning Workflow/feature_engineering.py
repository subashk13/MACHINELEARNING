import pandas as pd

# Sample dataset
df = pd.DataFrame({
    "Length": [5, 8, 10],
    "Width": [2, 4, 5]
})

# Create new feature
df["Area"] = df["Length"] * df["Width"]

print(df)
