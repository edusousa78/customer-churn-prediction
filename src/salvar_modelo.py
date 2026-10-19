from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_CSV = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "customer_churn_sintetico.csv"
)

PASTA_MODELOS = RAIZ_PROJETO / "models"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

CAMINHO_MODELO = (
    PASTA_MODELOS / "modelo_churn_logistico.joblib"
)

df = pd.read_csv(CAMINHO_CSV)

features = [
    "total_atendimentos",
    "atendimentos_abertos",
    "media_satisfacao",
]

X = df[features]
y = df["churn_30_dias"]

# Manter a divisão estabelecida nas aulas anteriores
X_dev, X_teste, y_dev, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_treino, X_validacao, y_treino, y_validacao = (
    train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev,
    )
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_treino, y_treino)

# Threshold selecionado na Aula 123
threshold_selecionado = 0.20

# Guardar modelo e configurações em um único artefato
artefato = {
    "modelo": modelo,
    "features": features,
    "threshold": threshold_selecionado,
}

joblib.dump(artefato, CAMINHO_MODELO)

# Verificar se o arquivo foi criado
print("=== PERSISTÊNCIA DO MODELO ===")
print("Modelo salvo:", CAMINHO_MODELO.exists())
print("Arquivo:", CAMINHO_MODELO.name)
print("Tamanho (bytes):", CAMINHO_MODELO.stat().st_size)

# Carregar o arquivo salvo
artefato_carregado = joblib.load(CAMINHO_MODELO)

print("\n=== VERIFICAÇÃO DO ARTEFATO ===")
print(
    "Algoritmo:",
    type(artefato_carregado["modelo"]).__name__,
)
print(
    "Features:",
    artefato_carregado["features"],
)
print(
    "Threshold:",
    artefato_carregado["threshold"],
)

print(
    "Modelo treinado:",
    hasattr(artefato_carregado["modelo"], "coef_"),
)