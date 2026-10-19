from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
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

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Baseline: sempre prever ausência de churn
y_pred_baseline = np.zeros(len(y_teste), dtype=int)

# Modelo de Regressão Logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_treino, y_treino)

probabilidades = modelo.predict_proba(X_teste)[:, 1]

# Threshold escolhido para comparação didática
y_pred_modelo = (probabilidades >= 0.30).astype(int)


def avaliar_modelo(nome, y_real, y_previsto):
    print(f"\n=== {nome} ===")

    print(
        f"Acurácia: "
        f"{accuracy_score(y_real, y_previsto) * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision_score(y_real, y_previsto, zero_division=0) * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall_score(y_real, y_previsto, zero_division=0) * 100:.2f}%"
    )

    print(
        f"F1-score: "
        f"{f1_score(y_real, y_previsto, zero_division=0) * 100:.2f}%"
    )

    print("Clientes previstos com churn:", int(y_previsto.sum()))


print("=== COMPARAÇÃO DE MODELOS ===")
print("Total de clientes no teste:", len(y_teste))
print("Clientes com churn real:", int(y_teste.sum()))

avaliar_modelo(
    "BASELINE - SEMPRE PREVER 0",
    y_teste,
    y_pred_baseline,
)

avaliar_modelo(
    "REGRESSÃO LOGÍSTICA - THRESHOLD 0.30",
    y_teste,
    y_pred_modelo,
)