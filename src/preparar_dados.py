from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_CSV = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "customer_churn_sintetico.csv"
)

df = pd.read_csv(CAMINHO_CSV)

# Variáveis explicativas
COLUNAS_FEATURES = [
    "total_atendimentos",
    "atendimentos_abertos",
    "media_satisfacao"
]

X = df[COLUNAS_FEATURES]
y = df["churn_30_dias"]

# Divisão estratificada
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=== DIMENSÕES ===")
print("X_treino:", X_treino.shape)
print("X_teste:", X_teste.shape)
print("y_treino:", y_treino.shape)
print("y_teste:", y_teste.shape)

print("\n=== DISTRIBUIÇÃO DO TARGET ===")
print("Base completa:")
print(y.value_counts().sort_index())

print("\nTreino:")
print(y_treino.value_counts().sort_index())

print("\nTeste:")
print(y_teste.value_counts().sort_index())

print("\n=== TAXAS DE CHURN ===")
print(f"Base completa: {y.mean() * 100:.2f}%")
print(f"Treino: {y_treino.mean() * 100:.2f}%")
print(f"Teste: {y_teste.mean() * 100:.2f}%")

print("\n=== VERIFICAÇÃO DE SOBREPOSIÇÃO ===")
indices_comuns = X_treino.index.intersection(X_teste.index)
print("Registros presentes nos dois conjuntos:", len(indices_comuns))