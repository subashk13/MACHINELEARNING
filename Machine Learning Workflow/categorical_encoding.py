import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample dataset
df = pd.DataFrame({
    "City": ["Chennai", "Coimbatore", "Madurai", "Chennai"]
})

# Create encoder
encoder = LabelEncoder()

# Encode categorical data
df["City"] = encoder.fit_transform(df["City"])

print(df)
