from pathlib import Path

import joblib
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_MODELO = (
    RAIZ_PROJETO
    / "models"
    / "modelo_churn_logistico.joblib"
)

# Carregar apenas o artefato confiável criado na Aula 127
artefato = joblib.load(CAMINHO_MODELO)

modelo = artefato["modelo"]
features = artefato["features"]
threshold = artefato["threshold"]

# Dois clientes fictícios para teste
clientes_sinteticos = pd.DataFrame(
    [
        {
            "total_atendimentos": 1,
            "atendimentos_abertos": 0,
            "media_satisfacao": 4.8,
        },
        {
            "total_atendimentos": 7,
            "atendimentos_abertos": 4,
            "media_satisfacao": 1.5,
        },
    ]
)

# Garantir a mesma ordem das variáveis do treinamento
X_novos = clientes_sinteticos[features]

# O modelo já está treinado: usamos predict_proba
probabilidades = modelo.predict_proba(X_novos)[:, 1]

previsoes = (
    probabilidades >= threshold
).astype(int)

print("=== PREVISÃO DE CHURN ===")
print("Modelo carregado:", type(modelo).__name__)
print(f"Threshold: {threshold:.2f}")

for indice, (probabilidade, previsao) in enumerate(
    zip(probabilidades, previsoes),
    start=1,
):
    print(f"\n=== CLIENTE SINTÉTICO {indice} ===")

    cliente = clientes_sinteticos.iloc[indice - 1]

    print(
        "Total de atendimentos:",
        cliente["total_atendimentos"],
    )

    print(
        "Atendimentos abertos:",
        cliente["atendimentos_abertos"],
    )

    print(
        "Média de satisfação:",
        cliente["media_satisfacao"],
    )

    print(
        f"Probabilidade estimada de churn: "
        f"{probabilidade * 100:.2f}%"
    )

    print("Classificação:", int(previsao))

    if previsao == 1:
        print("Resultado: POSSÍVEL CHURN")
    else:
        print("Resultado: SEM ALERTA DE CHURN")

print("\nPrevisões realizadas sem novo treinamento.")
