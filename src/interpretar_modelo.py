from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

# Reservar os dados de teste
X_dev, X_teste, y_dev, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Separar treinamento e validação
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

# Extrair coeficientes e calcular odds ratios
coeficientes = modelo.coef_[0]

df_coeficientes = pd.DataFrame(
    {
        "variavel": features,
        "coeficiente": coeficientes,
        "odds_ratio": np.exp(coeficientes),
    }
)

print("=== INTERPRETAÇÃO DO MODELO ===")
print("Algoritmo: Regressão Logística")
print("Clientes no treinamento:", len(X_treino))

print("\n=== COEFICIENTES ===")

for _, linha in df_coeficientes.iterrows():
    print(f"\nVariável: {linha['variavel']}")
    print(f"Coeficiente: {linha['coeficiente']:.4f}")
    print(f"Odds ratio: {linha['odds_ratio']:.4f}")

    if linha["coeficiente"] > 0:
        print("Associação: aumento das chances de churn")
    elif linha["coeficiente"] < 0:
        print("Associação: redução das chances de churn")
    else:
        print("Associação: coeficiente igual a zero")

print("\n=== INTERCEPTO ===")
print(f"Intercepto: {modelo.intercept_[0]:.4f}")

print("\nConjunto de teste não utilizado nesta análise.")
