from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

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

# Reservar 20% para o teste final
X_dev, X_teste, y_dev, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Dividir os 80% restantes em treinamento e validação
X_treino, X_validacao, y_treino, y_validacao = (
    train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )
)

# Treinar somente com os dados de treinamento
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_treino, y_treino)

# Calcular probabilidades somente na validação
probabilidades = modelo.predict_proba(X_validacao)[:, 1]

thresholds = [0.50, 0.40, 0.30, 0.20]

melhor_threshold = None
melhor_f1 = -1.0

print("=== SELEÇÃO DE THRESHOLD ===")
print("Treinamento:", len(X_treino))
print("Validação:", len(X_validacao))
print("Teste reservado:", len(X_teste))

for threshold in thresholds:
    previsoes = (probabilidades >= threshold).astype(int)

    precision = precision_score(
        y_validacao,
        previsoes,
        zero_division=0,
    )

    recall = recall_score(
        y_validacao,
        previsoes,
        zero_division=0,
    )

    f1 = f1_score(
        y_validacao,
        previsoes,
        zero_division=0,
    )

    print(f"\nThreshold: {threshold:.2f}")
    print("Previsões de churn:", int(previsoes.sum()))
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_threshold = threshold

print("\n=== MELHOR CONFIGURAÇÃO NA VALIDAÇÃO ===")
print(f"Threshold selecionado: {melhor_threshold:.2f}")
print(f"F1-score de validação: {melhor_f1 * 100:.2f}%")

print("\nConjunto de teste preservado para avaliação final.")