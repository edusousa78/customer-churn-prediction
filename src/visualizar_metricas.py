
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def gerar_grafico_metricas():
    """Gera um gráfico comparativo das métricas documentadas do modelo."""

    metricas = ["Acurácia", "Precisão", "Recall", "F1-score"]
    valores = [76.00, 28.00, 53.85, 36.84]

    raiz_projeto = Path(__file__).resolve().parents[1]
    pasta_saida = raiz_projeto / "reports" / "figures"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivo_saida = pasta_saida / "metricas_classificacao.png"

    figura, eixo = plt.subplots(figsize=(9, 5.5))

    barras = eixo.bar(
        metricas,
        valores,
        color=["#2563EB", "#D97706", "#059669", "#7C3AED"],
        width=0.58,
    )

    eixo.set_title(
        "Desempenho do Modelo — Customer Churn Prediction",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    eixo.set_ylabel("Resultado (%)")
    eixo.set_ylim(0, 100)

    eixo.bar_label(
        barras,
        labels=[f"{valor:.2f}%" for valor in valores],
        padding=5,
        fontsize=10,
    )

    eixo.grid(axis="y", alpha=0.2)
    eixo.set_axisbelow(True)

    figura.tight_layout()

    figura.savefig(
        arquivo_saida,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print(f"Gráfico gerado com sucesso: {arquivo_saida}")


if __name__ == "__main__":
    gerar_grafico_metricas()
