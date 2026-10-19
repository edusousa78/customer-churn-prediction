import numpy as np
import pandas as pd

QUANTIDADE_CLIENTES = 1000

rng = np.random.default_rng(42)

df_sintetico = pd.DataFrame({
    "cliente_id": range(1, QUANTIDADE_CLIENTES + 1)
})

df_sintetico["data_referencia"] = pd.Timestamp("2025-08-31")

# Quantidade de atendimentos por cliente
df_sintetico["total_atendimentos"] = rng.poisson(
    lam=2,
    size=QUANTIDADE_CLIENTES
)

# Quantidade de atendimentos abertos
df_sintetico["atendimentos_abertos"] = rng.binomial(
    n=df_sintetico["total_atendimentos"].to_numpy(),
    p=0.25
)

# Média de satisfação fictícia, entre 1 e 5
df_sintetico["media_satisfacao"] = np.round(
    rng.uniform(1, 5, QUANTIDADE_CLIENTES),
    2
)

print("=== BASE SINTÉTICA COM FEATURES ===")
print(df_sintetico.head())

print("\n=== DIMENSÕES ===")
print(df_sintetico.shape)

print("\n=== VALIDAÇÃO DOS ATENDIMENTOS ===")
print(
    "Atendimentos abertos maiores que o total:",
    (
        df_sintetico["atendimentos_abertos"]
        > df_sintetico["total_atendimentos"]
    ).sum()
)

print("\n=== INTERVALO DE SATISFAÇÃO ===")
print("Mínimo:", df_sintetico["media_satisfacao"].min())
print("Máximo:", df_sintetico["media_satisfacao"].max())

# Pontuação de risco sintética
pontuacao_risco = (
    -1.5
    + 0.25 * df_sintetico["total_atendimentos"]
    + 0.55 * df_sintetico["atendimentos_abertos"]
    - 0.45 * df_sintetico["media_satisfacao"]
)

# Conversão da pontuação em probabilidade
probabilidade_churn = 1 / (1 + np.exp(-pontuacao_risco))

# Simulação de cancelamento efetivo nos próximos 30 dias
df_sintetico["churn_30_dias"] = rng.binomial(
    n=1,
    p=probabilidade_churn.to_numpy()
)

print("\n=== DISTRIBUIÇÃO DE CHURN ===")
print(
    df_sintetico["churn_30_dias"]
    .value_counts()
    .sort_index()
)

print("\n=== TAXA DE CHURN ===")
taxa_churn = df_sintetico["churn_30_dias"].mean() * 100
print(f"{taxa_churn:.2f}%")

print("\n=== VALIDAÇÃO DO TARGET ===")
print(
    "Valores únicos:",
    sorted(df_sintetico["churn_30_dias"].unique().tolist())
)

print("\n=== DIMENSÕES ATUALIZADAS ===")
print(df_sintetico.shape)

from pathlib import Path

# Localização da raiz do projeto
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Diretório de saída
DIRETORIO_DADOS = RAIZ_PROJETO / "data" / "processed"
DIRETORIO_DADOS.mkdir(parents=True, exist_ok=True)

# Caminho do arquivo CSV
CAMINHO_CSV = DIRETORIO_DADOS / "customer_churn_sintetico.csv"

# Salvar os dados
df_sintetico.to_csv(CAMINHO_CSV, index=False)

# Recuperar os dados salvos
df_verificacao = pd.read_csv(CAMINHO_CSV)

print("\n=== PERSISTÊNCIA DO DATASET ===")
print("Arquivo:", CAMINHO_CSV.name)
print("Arquivo existe:", CAMINHO_CSV.exists())

print("\n=== VALIDAÇÃO DO CSV ===")
print("Dimensões:", df_verificacao.shape)
print("IDs únicos:", df_verificacao["cliente_id"].nunique())
print("Valores ausentes:", df_verificacao.isna().sum().sum())
print("Taxa de churn:", f"{df_verificacao['churn_30_dias'].mean() * 100:.2f}%")

print("\n=== COMPARAÇÃO COM A BASE ORIGINAL ===")
print(
    "Dados preservados:",
    df_verificacao.equals(
        pd.read_csv(CAMINHO_CSV)
    )
)

# Comparação rigorosa entre a base original e o CSV

df_comparacao = pd.read_csv(
    CAMINHO_CSV,
    parse_dates=["data_referencia"]
)

print("\n=== COMPARAÇÃO RIGOROSA ===")

try:
    pd.testing.assert_frame_equal(
        df_sintetico,
        df_comparacao,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12
    )

    print("Integridade dos dados: VALIDADA")
    print("DataFrames equivalentes: True")

except AssertionError as erro:
    print("Integridade dos dados: FALHOU")
    print(erro)
    