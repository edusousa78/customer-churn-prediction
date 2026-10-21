from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def gerar_matriz_confusao():
    """Gera a matriz de confusão dos resultados documentados do teste."""

    # Linhas: classes reais.
    # Colunas: classes previstas.
    matriz = np.array(
        [
            [138, 36],
            [12, 14],
        ]
    )

    raiz_projeto = Path(__file__).resolve().parents[1]
    pasta_saida = raiz_projeto / "reports" / "figures"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivo_saida = pasta_saida / "matriz_confusao.png"

    sns.set_theme(style="white")

    figura, eixo = plt.subplots(figsize=(8, 6))

    visualizacao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Não cancelou", "Cancelou"],
    )

    visualizacao.plot(
        ax=eixo,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    eixo.set_title(
        "Matriz de Confusão — Customer Churn Prediction",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    eixo.set_xlabel("Classificação prevista")
    eixo.set_ylabel("Classificação real")

    figura.tight_layout()

    figura.savefig(
        arquivo_saida,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print(f"Gráfico gerado com sucesso: {arquivo_saida}")


if __name__ == "__main__":
    gerar_matriz_confusao()