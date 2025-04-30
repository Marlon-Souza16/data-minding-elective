import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('dataset.csv', sep=';', encoding='latin1', on_bad_lines='skip')

print("Colunas disponíveis no dataset:")
print(df.columns)

colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_selected = df[colunas].copy()

df_selected = df_selected.dropna()

df_selected['Q21Gender'] = df_selected['Q21Gender'].astype('category').cat.codes
df_selected['Q22Income'] = df_selected['Q22Income'].astype('category').cat.codes

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

inertia = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.grid()
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df_selected['Cluster'] = clusters

cluster_counts = df_selected['Cluster'].value_counts(normalize=True) * 100
print("\nTamanho dos clusters (%):\n", cluster_counts)

menor_cluster = cluster_counts.idxmin()
print(f"\nCluster incomum identificado: {menor_cluster}")
perfil = df_selected[df_selected['Cluster'] == menor_cluster].describe()
print("\nPerfil do cluster incomum:\n", perfil)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_scaled)
df_selected['PCA1'] = reduced_data[:, 0]
df_selected['PCA2'] = reduced_data[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_selected, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('Clusters visualizados com PCA')
plt.show()
