
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def gerar_grafico_coeficientes():
    """Visualiza os coeficientes do modelo de churn salvo."""

    raiz_projeto = Path(__file__).resolve().parents[1]

    caminho_modelo = (
        raiz_projeto
        / "models"
        / "modelo_churn_logistico.joblib"
    )

    pasta_saida = raiz_projeto / "reports" / "figures"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivo_saida = pasta_saida / "coeficientes_modelo.png"

    # Carregar somente um artefato confiável do próprio projeto.
    artefato = joblib.load(caminho_modelo)

    modelo = artefato["modelo"]
    features = artefato["features"]

    coeficientes = modelo.coef_[0]

    if len(features) != len(coeficientes):
        raise ValueError(
            "Quantidade de features diferente da quantidade "
            "de coeficientes."
        )

    odds_ratios = np.exp(coeficientes)

    # Organizar os resultados para facilitar a leitura.
    resultados = sorted(
        zip(features, coeficientes, odds_ratios),
        key=lambda item: item[1],
    )

    nomes = [item[0] for item in resultados]
    valores = [item[1] for item in resultados]

    cores = [
        "#2563EB" if valor >= 0 else "#D97706"
        for valor in valores
    ]

    figura, eixo = plt.subplots(figsize=(10, 5.5))

    barras = eixo.barh(
        nomes,
        valores,
        color=cores,
        height=0.55,
    )

    eixo.axvline(
        0,
        color="gray",
        linewidth=1,
    )

    eixo.set_title(
        "Coeficientes da Regressão Logística — Churn",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    eixo.set_xlabel(
        "Coeficiente na escala logit (por unidade da variável)"
    )

    eixo.bar_label(
        barras,
        fmt="%.4f",
        padding=4,
    )

    eixo.grid(axis="x", alpha=0.2)
    eixo.set_axisbelow(True)

    figura.tight_layout()

    figura.savefig(
        arquivo_saida,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print("=== COEFICIENTES DO MODELO ===")

    for nome, coeficiente, odds_ratio in resultados:
        print(f"\nVariável: {nome}")
        print(f"Coeficiente: {coeficiente:.4f}")
        print(f"Odds ratio: {odds_ratio:.4f}")

    print(f"\nGráfico gerado: {arquivo_saida}")
    print("Observação: coeficientes não demonstram causalidade.")


if __name__ == "__main__":
    gerar_grafico_coeficientes()

