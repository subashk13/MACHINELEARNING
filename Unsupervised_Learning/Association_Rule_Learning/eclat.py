#Scikit-learn does not provide an ECLAT implementation. A commonly used package is pyECLAT.
#Requires: pip install pyECLAT
import pandas as pd
from pyECLAT import ECLAT

# Sample transactions
data = pd.DataFrame({
    "Item1": ["Milk", "Milk", "Bread", "Milk"],
    "Item2": ["Bread", "Butter", "Butter", "Eggs"],
    "Item3": ["Eggs", None, "Milk", None]
})

# Create model
model = ECLAT(data=data)

# Find frequent itemsets
itemsets, support = model.fit(
    min_support=0.5,
    min_combination=1,
    max_combination=3
)

# Results
print(itemsets)
print(support)
