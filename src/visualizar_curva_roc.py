
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def gerar_curva_roc():
    """Gera a curva ROC usando o conjunto de teste sintético."""

    raiz_projeto = Path(__file__).resolve().parents[1]

    caminho_csv = (
        raiz_projeto
        / "data"
        / "processed"
        / "customer_churn_sintetico.csv"
    )

    pasta_saida = raiz_projeto / "reports" / "figures"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivo_saida = pasta_saida / "curva_roc.png"

    # Carregar os dados sintéticos.
    df = pd.read_csv(caminho_csv)

    features = [
        "total_atendimentos",
        "atendimentos_abertos",
        "media_satisfacao",
    ]

    X = df[features]
    y = df["churn_30_dias"]

    # Reproduzir a divisão original dos dados.
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

    # Reproduzir o treinamento da avaliação final.
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_treino, y_treino)

    # Obter probabilidades da classe positiva (churn).
    probabilidades = modelo.predict_proba(X_teste)[:, 1]

    # Calcular a curva ROC e a área sob a curva.
    fpr, tpr, _ = roc_curve(
        y_teste,
        probabilidades,
    )

    roc_auc = roc_auc_score(
        y_teste,
        probabilidades,
    )

    # Construir o gráfico.
    figura, eixo = plt.subplots(figsize=(8, 6))

    eixo.plot(
        fpr,
        tpr,
        color="#2563EB",
        linewidth=2,
        label=f"Regressão Logística (AUC = {roc_auc:.4f})",
    )

    eixo.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Referência aleatória (AUC = 0,50)",
    )

    eixo.set_title(
        "Curva ROC — Customer Churn Prediction",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    eixo.set_xlabel("Taxa de falsos positivos (FPR)")
    eixo.set_ylabel("Taxa de verdadeiros positivos (TPR)")

    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)

    eixo.legend(loc="lower right")
    eixo.grid(alpha=0.2)

    figura.tight_layout()

    figura.savefig(
        arquivo_saida,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print("=== CURVA ROC ===")
    print(f"Clientes no teste: {len(X_teste)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Gráfico gerado: {arquivo_saida}")


if __name__ == "__main__":
    gerar_curva_roc()


