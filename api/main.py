

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from prever_churn_validado import carregar_modelo, prever_cliente

app = FastAPI(
    title="Conecta Brasil Telecom - Churn API",
    description="API educacional para previsão de cancelamento de clientes.",
    version="1.0.0",
)


class ClienteEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_atendimentos: int = Field(ge=0, strict=True)
    atendimentos_abertos: int = Field(ge=0, strict=True)
    media_satisfacao: float = Field(ge=1, le=5, strict=True)


@app.get("/")
def pagina_inicial():
    return {
        "projeto": "Conecta Brasil Telecom",
        "servico": "Customer Churn Prediction",
        "status": "online",
    }


@app.get("/health")
def verificar_saude():
    return {
        "status": "ok",
        "mensagem": "API funcionando corretamente",
    }


@app.post("/prever")
def prever_churn(cliente: ClienteEntrada):
    dados_cliente = cliente.model_dump()

    try:
        modelo, features, threshold = carregar_modelo()

        probabilidade, classificacao = prever_cliente(
            dados_cliente,
            modelo,
            features,
            threshold,
        )

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro

    return {
        "probabilidade_churn": round(probabilidade, 4),
        "classificacao": classificacao,
        "alerta": classificacao == 1,
        "threshold": threshold,
    }

