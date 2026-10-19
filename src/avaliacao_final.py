from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
    "media_satisfacao",
]

X = df[features]
y = df["churn_30_dias"]

# Primeira divisão: desenvolvimento e teste
X_dev, X_teste, y_dev, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Segunda divisão: treinamento e validação
X_treino, X_validacao, y_treino, y_validacao = (
    train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )
)

# Treinar somente com o conjunto de treinamento
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_treino, y_treino)

# Threshold escolhido na Aula 123
threshold_selecionado = 0.20

# Previsões no conjunto de teste
probabilidades = modelo.predict_proba(X_teste)[:, 1]

y_pred = (
    probabilidades >= threshold_selecionado
).astype(int)

# Calcular métricas
acuracia = accuracy_score(y_teste, y_pred)
precision = precision_score(
    y_teste, y_pred, zero_division=0
)
recall = recall_score(
    y_teste, y_pred, zero_division=0
)
f1 = f1_score(
    y_teste, y_pred, zero_division=0
)

# Matriz de confusão
matriz = confusion_matrix(
    y_teste, y_pred, labels=[0, 1]
)

vn, fp, fn, vp = matriz.ravel()

print("=== AVALIAÇÃO FINAL DO MODELO ===")
print("Algoritmo: Regressão Logística")
print(f"Threshold selecionado: {threshold_selecionado:.2f}")
print("Clientes no treinamento:", len(X_treino))
print("Clientes no teste:", len(X_teste))

print("\n=== MÉTRICAS NO TESTE ===")
print(f"Acurácia: {acuracia * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")

print("\n=== MATRIZ DE CONFUSÃO ===")
print(matriz)

print("\n=== INTERPRETAÇÃO ===")
print("Verdadeiros negativos:", vn)
print("Falsos positivos:", fp)
print("Falsos negativos:", fn)
print("Verdadeiros positivos:", vp)

print("\n=== RESUMO DE CHURN ===")
print("Cancelamentos reais:", int(y_teste.sum()))
print("Cancelamentos identificados:", vp)
print("Cancelamentos não identificados:", fn)
print("Clientes previstos com churn:", int(y_pred.sum()))

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# As métricas usam probabilidades, não classes 0 e 1
roc_auc = roc_auc_score(
    y_teste,
    probabilidades,
)

average_precision = average_precision_score(
    y_teste,
    probabilidades,
)

# Referência de Average Precision para um classificador
# sem capacidade de discriminação
prevalencia_churn = y_teste.mean()

print("\n=== MÉTRICAS BASEADAS EM PROBABILIDADES ===")

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision (AP): {average_precision:.4f}")
print(f"Prevalência de churn: {prevalencia_churn:.4f}")

print("\n=== INTERPRETAÇÃO ===")

print(
    "ROC-AUC acima de 0.50:",
    roc_auc > 0.50,
)

print(
    "Average Precision acima da prevalência:",
    average_precision > prevalencia_churn,
)