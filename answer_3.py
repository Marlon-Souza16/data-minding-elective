import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('dataset.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

cols = ['Q5FIRSTTIME', 'Q21Gender', 'Q22Income', 'Q20Age', 'Q23FLY', 'Q5TIMESFLOWN']
df = df[cols].dropna()

X = df[['Q21Gender', 'Q22Income', 'Q20Age', 'Q23FLY', 'Q5TIMESFLOWN']]
y = df['Q5FIRSTTIME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.4f}")
