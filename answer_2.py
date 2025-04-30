import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset.csv', sep=';', encoding='ISO-8859-1', on_bad_lines='skip')

colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_selected = df[colunas].copy()

df_selected = df_selected.dropna()

label_encoders = {}
for col in ['Q21Gender', 'Q22Income', 'Q23FLY']:
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    label_encoders[col] = le

df_bin = df_selected.copy()

df_bin['NETPRO  '] = df_bin['NETPRO  '].apply(lambda x: 1 if x >= 9 else 0)  
df_bin['Q20Age'] = df_bin['Q20Age'].apply(lambda x: 1 if x > 40 else 0)  
df_bin['Q5TIMESFLOWN'] = df_bin['Q5TIMESFLOWN'].apply(lambda x: 1 if x > 3 else 0)  
df_bin['Q6LONGUSE'] = df_bin['Q6LONGUSE'].apply(lambda x: 1 if x >= 5 else 0)  

df_bin = df_bin.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = fpgrowth(df_bin, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print("Regras de Associação Geradas:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", size="lift", sizes=(20, 200))
plt.title("Regras de Associação: Suporte vs Confiança")
plt.show()
