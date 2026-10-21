
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


def gerar_curva_precision_recall():
    """Gera a curva Precision-Recall no conjunto de teste sintético."""

    raiz_projeto = Path(__file__).resolve().parents[1]

    caminho_csv = (
        raiz_projeto
        / "data"
        / "processed"
        / "customer_churn_sintetico.csv"
    )

    pasta_saida = raiz_projeto / "reports" / "figures"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivo_saida = pasta_saida / "curva_precision_recall.png"

    # Carregar os dados sintéticos.
    df = pd.read_csv(caminho_csv)

    features = [
        "total_atendimentos",
        "atendimentos_abertos",
        "media_satisfacao",
    ]

    X = df[features]
    y = df["churn_30_dias"]

    # Reproduzir a divisão da avaliação final.
    X_dev, X_teste, y_dev, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_treino, _, y_treino, _ = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )

    # Treinar somente com os dados de treinamento.
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_treino, y_treino)

    # Calcular as probabilidades de churn.
    probabilidades = modelo.predict_proba(X_teste)[:, 1]

    # Calcular a curva Precision-Recall.
    precisao, recall, _ = precision_recall_curve(
        y_teste,
        probabilidades,
    )

    average_precision = average_precision_score(
        y_teste,
        probabilidades,
    )

    prevalencia = y_teste.mean()

    # Criar o gráfico.
    figura, eixo = plt.subplots(figsize=(8, 6))

    eixo.plot(
        recall,
        precisao,
        color="#2563EB",
        linewidth=2,
        label=f"Regressão Logística (AP = {average_precision:.4f})",
    )

    eixo.axhline(
        y=prevalencia,
        color="gray",
        linestyle="--",
        label=f"Referência da prevalência ({prevalencia:.2%})",
    )

    eixo.set_title(
        "Curva Precision-Recall — Customer Churn Prediction",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    eixo.set_xlabel("Recall (sensibilidade)")
    eixo.set_ylabel("Precision (precisão)")

    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)

    eixo.legend(loc="upper right")
    eixo.grid(alpha=0.2)

    figura.tight_layout()

    figura.savefig(
        arquivo_saida,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print("=== CURVA PRECISION-RECALL ===")
    print(f"Clientes no teste: {len(X_teste)}")
    print(f"Average Precision (AP): {average_precision:.4f}")
    print(f"Prevalência de churn: {prevalencia:.2%}")
    print(f"Gráfico gerado: {arquivo_saida}")


if __name__ == "__main__":
    gerar_curva_precision_recall()
