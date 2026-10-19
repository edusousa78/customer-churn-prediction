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

features = [
    "total_atendimentos",
    "atendimentos_abertos",
    "media_satisfacao",
]

X = df[features]
y = df["churn_30_dias"]

# Primeira divisão: 80% desenvolvimento e 20% teste
X_dev, X_teste, y_dev, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Segunda divisão: 60% treinamento e 20% validação
# 25% de 800 = 200 clientes
X_treino, X_validacao, y_treino, y_validacao = (
    train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )
)

print("=== DIVISÃO DOS DADOS ===")
print("Total de clientes:", len(df))
print("Treinamento:", len(X_treino))
print("Validação:", len(X_validacao))
print("Teste:", len(X_teste))

print("\n=== DISTRIBUIÇÃO DE CHURN ===")

for nome, alvo in [
    ("Treinamento", y_treino),
    ("Validação", y_validacao),
    ("Teste", y_teste),
]:
    quantidade_churn = int(alvo.sum())
    percentual = alvo.mean() * 100

    print(
        f"{nome}: "
        f"{quantidade_churn} churn "
        f"({percentual:.2f}%)"
    )

print("\n=== VERIFICAÇÃO DE SOBREPOSIÇÃO ===")

sobreposicao_treino_validacao = len(
    set(X_treino.index) & set(X_validacao.index)
)

sobreposicao_treino_teste = len(
    set(X_treino.index) & set(X_teste.index)
)

sobreposicao_validacao_teste = len(
    set(X_validacao.index) & set(X_teste.index)
)

print(
    "Treinamento x Validação:",
    sobreposicao_treino_validacao,
)

print(
    "Treinamento x Teste:",
    sobreposicao_treino_teste,
)

print(
    "Validação x Teste:",
    sobreposicao_validacao_teste,
)