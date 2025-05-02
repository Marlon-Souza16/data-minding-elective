from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import fpgrowth, association_rules

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def _preprocess_data(df, cols, encode_cols=None):
    df_selected = df[cols].copy().dropna()
    if encode_cols:
        for col in encode_cols:
            df_selected[col] = df_selected[col].astype('category').cat.codes
    return df_selected


def answer_1(df, n_clusters=4):
    cols = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_preprocessed = _preprocess_data(df, cols, encode_cols=['Q21Gender', 'Q22Income'])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_preprocessed)

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

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    df_preprocessed['Cluster'] = clusters

    cluster_counts = df_preprocessed['Cluster'].value_counts(normalize=True) * 100
    print("\nTamanho dos clusters (%):\n", cluster_counts)

    menor_cluster = cluster_counts.idxmin()
    print(f"\nCluster incomum identificado: {menor_cluster}")
    perfil = df_preprocessed[df_preprocessed['Cluster'] == menor_cluster].describe()
    print("\nPerfil do cluster incomum:\n", perfil)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled)
    df_preprocessed['PCA1'] = reduced_data[:, 0]
    df_preprocessed['PCA2'] = reduced_data[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_preprocessed, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
    plt.title('Clusters visualizados com PCA')
    plt.show()


def answer_2(df, min_support=0.1, lift_threshold=1.0):
    cols = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_preprocessed = _preprocess_data(df, cols)

    # Encode categorical columns
    label_encoders = {}
    for col in ['Q21Gender', 'Q22Income', 'Q23FLY']:
        le = LabelEncoder()
        df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
        label_encoders[col] = le

    df_bin = df_preprocessed.copy()
    df_bin['NETPRO  '] = df_bin['NETPRO  '].apply(lambda x: 1 if x >= 9 else 0)
    df_bin['Q20Age'] = df_bin['Q20Age'].apply(lambda x: 1 if x > 40 else 0)
    df_bin['Q5TIMESFLOWN'] = df_bin['Q5TIMESFLOWN'].apply(lambda x: 1 if x > 3 else 0)
    df_bin['Q6LONGUSE'] = df_bin['Q6LONGUSE'].apply(lambda x: 1 if x >= 5 else 0)

    df_bin = df_bin.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = fpgrowth(df_bin, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)

    print("Regras de Associação Geradas:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", size="lift", sizes=(20, 200))
    plt.title("Regras de Associação: Suporte vs Confiança")
    plt.show()


def answer_3(df):
    cols = ['Q5FIRSTTIME', 'Q21Gender', 'Q22Income', 'Q20Age', 'Q23FLY', 'Q5TIMESFLOWN']
    df_preprocessed = _preprocess_data(df, cols)

    X = df_preprocessed[['Q21Gender', 'Q22Income', 'Q20Age', 'Q23FLY', 'Q5TIMESFLOWN']]
    y = df_preprocessed['Q5FIRSTTIME']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.4f}")
