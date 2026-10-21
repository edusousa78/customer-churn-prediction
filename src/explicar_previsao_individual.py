
import math
from pathlib import Path

import joblib
import pandas as pd

from validar_entrada import validar_cliente

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = (
    RAIZ_PROJETO / "models" / "modelo_churn_logistico.joblib"
)


def explicar_previsao(cliente):
    """Explica uma previsão usando os coeficientes do modelo."""

    validar_cliente(cliente)

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    artefato = joblib.load(CAMINHO_MODELO)

    modelo = artefato["modelo"]
    features = artefato["features"]
    threshold = float(artefato["threshold"])

    dados = pd.DataFrame([cliente])[features]

    intercepto = float(modelo.intercept_[0])
    coeficientes = modelo.coef_[0]

    logit = intercepto
    contribuicoes = []

    for nome, coeficiente in zip(features, coeficientes):
        valor = float(dados.iloc[0][nome])
        contribuicao = float(coeficiente) * valor

        logit += contribuicao

        contribuicoes.append(
            {
                "variavel": nome,
                "valor": valor,
                "coeficiente": float(coeficiente),
                "contribuicao": contribuicao,
            }
        )

    # Forma numericamente estável da função sigmoide.
    if logit >= 0:
        probabilidade_calculada = 1 / (1 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probabilidade_calculada = exp_logit / (1 + exp_logit)

    probabilidade_modelo = float(
        modelo.predict_proba(dados)[0, 1]
    )

    if not math.isclose(
        probabilidade_calculada,
        probabilidade_modelo,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "A decomposição não corresponde à previsão do modelo."
        )

    classificacao = int(probabilidade_modelo >= threshold)

    return {
        "intercepto": intercepto,
        "contribuicoes": contribuicoes,
        "logit": logit,
        "probabilidade": probabilidade_modelo,
        "threshold": threshold,
        "classificacao": classificacao,
    }


def main():
    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    resultado = explicar_previsao(cliente)

    print("=== EXPLICACAO INDIVIDUAL DE CHURN ===")

    print("\nDados do cliente:")
    for nome, valor in cliente.items():
        print(f"  {nome}: {valor}")

    print(f"\nIntercepto: {resultado['intercepto']:+.4f}")

    print("\nContribuicoes para o logit:")

    for item in resultado["contribuicoes"]:
        print(
            f"  {item['variavel']}: "
            f"{item['valor']:.2f} x "
            f"{item['coeficiente']:+.4f} = "
            f"{item['contribuicao']:+.4f}"
        )

    print(f"\nLogit total: {resultado['logit']:+.4f}")

    print(
        "Probabilidade estimada de churn: "
        f"{resultado['probabilidade'] * 100:.2f}%"
    )

    print(f"Threshold: {resultado['threshold']:.2f}")
    print(f"Classificacao: {resultado['classificacao']}")

    if resultado["classificacao"] == 1:
        print("Resultado: POSSIVEL CHURN")
    else:
        print("Resultado: SEM ALERTA DE CHURN")

    print(
        "\nNota: as contribuicoes explicam o calculo "
        "do modelo, nao relacoes causais."
    )


if __name__ == "__main__":
    main()
