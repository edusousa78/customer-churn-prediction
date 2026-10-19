from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
    "media_satisfacao"
]

X = df[features]
y = df["churn_30_dias"]

# Separação estratificada
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Criação do modelo
modelo = LogisticRegression(max_iter=1000)

# Treinamento
modelo.fit(X_treino, y_treino)

# Previsões no conjunto de teste
y_pred = modelo.predict(X_teste)

print("=== PRIMEIRO MODELO DE CHURN ===")
print("Algoritmo: Regressão Logística")
print("Treinamento concluído:", hasattr(modelo, "coef_"))

print("\n=== PREVISÕES ===")
print("Total de previsões:", len(y_pred))
print("Previsões sem churn:", (y_pred == 0).sum())
print("Previsões com churn:", (y_pred == 1).sum())

print("\n=== MÉTRICAS NO TESTE ===")
print(f"Acurácia: {accuracy_score(y_teste, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_teste, y_pred, zero_division=0) * 100:.2f}%")
print(f"Recall: {recall_score(y_teste, y_pred, zero_division=0) * 100:.2f}%")
print(f"F1-score: {f1_score(y_teste, y_pred, zero_division=0) * 100:.2f}%")

from sklearn.metrics import confusion_matrix

# Matriz de confusão do modelo treinado
matriz = confusion_matrix(
    y_teste,
    y_pred,
    labels=[0, 1]
)

vn, fp, fn, vp = matriz.ravel()

print("\n=== MATRIZ DE CONFUSÃO ===")
print(matriz)

print("\n=== INTERPRETAÇÃO ===")
print("Verdadeiros negativos:", vn)
print("Falsos positivos:", fp)
print("Falsos negativos:", fn)
print("Verdadeiros positivos:", vp)

print("\n=== CLIENTES COM CHURN ===")
print("Total real de churn:", fn + vp)
print("Churn identificado:", vp)
print("Churn não identificado:", fn)

# Probabilidades previstas para a classe churn (1)
probabilidades = modelo.predict_proba(X_teste)[:, 1]

print("\n=== COMPARAÇÃO DE THRESHOLDS ===")

for threshold in [0.50, 0.30]:

    # Converter probabilidades em classes
    y_pred_threshold = (
        probabilidades >= threshold
    ).astype(int)

    precision = precision_score(
        y_teste, y_pred_threshold, zero_division=0
    )

    recall = recall_score(
        y_teste, y_pred_threshold, zero_division=0
    )

    f1 = f1_score(
        y_teste, y_pred_threshold, zero_division=0
    )

    matriz_threshold = confusion_matrix(
        y_teste, y_pred_threshold, labels=[0, 1]
    )

    vn, fp, fn, vp = matriz_threshold.ravel()

    print(f"\nThreshold: {threshold:.2f}")
    print("Previsões de churn:", int(y_pred_threshold.sum()))
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Falsos positivos: {fp}")
    print(f"Falsos negativos: {fn}")
    print(f"Verdadeiros positivos: {vp}")
    