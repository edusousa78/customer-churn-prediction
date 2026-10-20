from pathlib import Path

import joblib
import pandas as pd

from validar_entrada import validar_cliente

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_MODELO = (
    RAIZ_PROJETO
    / "models"
    / "modelo_churn_logistico.joblib"
)


def carregar_modelo():
    """Carrega o modelo, as features e o threshold."""

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    artefato = joblib.load(CAMINHO_MODELO)

    modelo = artefato["modelo"]
    features = artefato["features"]
    threshold = artefato["threshold"]

    return modelo, features, threshold


def prever_cliente(cliente, modelo, features, threshold):
    """Valida um cliente e retorna sua previsão de churn."""

    validar_cliente(cliente)

    dados = pd.DataFrame([cliente])
    X_novo = dados[features]

    probabilidade = float(
        modelo.predict_proba(X_novo)[0, 1]
    )

    classificacao = int(
        probabilidade >= threshold
    )

    return probabilidade, classificacao


def main():
    modelo, features, threshold = carregar_modelo()

    clientes_teste = [
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
        {
            "total_atendimentos": 2,
            "atendimentos_abertos": 5,
            "media_satisfacao": 3.0,
        },
    ]

    print("=== SISTEMA DE PREVISÃO COM VALIDAÇÃO ===")
    print(f"Threshold: {threshold:.2f}")

    for indice, cliente in enumerate(
        clientes_teste,
        start=1,
    ):
        print(f"\n=== CLIENTE {indice} ===")

        try:
            probabilidade, classificacao = prever_cliente(
                cliente,
                modelo,
                features,
                threshold,
            )

            print("Dados: VÁLIDOS")
            print(
                f"Probabilidade de churn: "
                f"{probabilidade * 100:.2f}%"
            )
            print(f"Classificação: {classificacao}")

            if classificacao == 1:
                print("Resultado: POSSÍVEL CHURN")
            else:
                print("Resultado: SEM ALERTA DE CHURN")

        except ValueError as erro:
            print("Dados: INVÁLIDOS")
            print(f"Motivo: {erro}")
            print("Previsão não realizada.")


if __name__ == "__main__":
    main()