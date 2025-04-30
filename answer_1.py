import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Carregar o dataset
df = pd.read_csv('dataset.csv', sep=';', encoding='latin1', on_bad_lines='skip')

# 2. Exibir as colunas do dataframe
print("Colunas disponíveis no dataset:")
print(df.columns)

# 3. Selecionar as variáveis relevantes (ajustado com base nas colunas corretas)
colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_selected = df[colunas].copy()

# 4. Tratar dados faltantes
df_selected = df_selected.dropna()

# 5. Converter variáveis categóricas em numéricas (ex: Q21Gender, Q22Income)
df_selected['Q21Gender'] = df_selected['Q21Gender'].astype('category').cat.codes
df_selected['Q22Income'] = df_selected['Q22Income'].astype('category').cat.codes

# 6. Normalizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# 7. Encontrar o número ideal de clusters com método do cotovelo
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

# 8. Aplicar K-Means com número ideal (suponha k=4, você pode ajustar)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df_selected['Cluster'] = clusters

# 9. Analisar tamanho dos clusters
cluster_counts = df_selected['Cluster'].value_counts(normalize=True) * 100
print("\nTamanho dos clusters (%):\n", cluster_counts)

# 10. Perfil do grupo incomum (menor cluster)
menor_cluster = cluster_counts.idxmin()
print(f"\nCluster incomum identificado: {menor_cluster}")
perfil = df_selected[df_selected['Cluster'] == menor_cluster].describe()
print("\nPerfil do cluster incomum:\n", perfil)

# 11. Visualização com PCA (opcional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_scaled)
df_selected['PCA1'] = reduced_data[:, 0]
df_selected['PCA2'] = reduced_data[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_selected, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('Clusters visualizados com PCA')
plt.show()
