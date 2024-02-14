from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

# Sample transaction data (list of lists)
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'diapers', 'beer', 'nuts'],
    ['bread', 'butter', 'diapers'],
    ['bread', 'diapers', 'beer', 'nuts']
]

# Convert transaction data to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply FP-Growth algorithm to find frequent itemsets
frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)

# Display frequent itemsets
print(frequent_itemsets)
