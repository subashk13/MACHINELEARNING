#Requires: pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample dataset
data = {
    "Milk": [1, 1, 0, 1],
    "Bread": [1, 0, 1, 1],
    "Butter": [0, 1, 1, 1],
    "Eggs": [1, 1, 1, 0]
}

df = pd.DataFrame(data).astype(bool)

# Frequent Itemsets
itemsets = apriori(
    df,
    min_support=0.5,
    use_colnames=True
)

# Association Rules
rules = association_rules(
    itemsets,
    metric="confidence",
    min_threshold=0.7
)

print(itemsets)
print(rules)
