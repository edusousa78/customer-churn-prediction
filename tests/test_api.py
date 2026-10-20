
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_pagina_inicial():
    resposta = client.get("/")

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["projeto"] == "Conecta Brasil Telecom"
    assert dados["status"] == "online"


def test_health():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_prever_cliente_valido():
    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 200

    dados = resposta.json()

    assert 0 <= dados["probabilidade_churn"] <= 1
    assert dados["classificacao"] == 1
    assert dados["alerta"] is True
    assert dados["threshold"] == 0.2


def test_prever_cliente_invalido():
    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.0,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 422

    dados = resposta.json()

    assert "Atendimentos abertos" in dados["detail"]


def test_prever_campo_obrigatorio_ausente():
    cliente = {
        "total_atendimentos": 3,
        "atendimentos_abertos": 1,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 422


def test_prever_satisfacao_fora_limite():
    cliente = {
        "total_atendimentos": 3,
        "atendimentos_abertos": 1,
        "media_satisfacao": 6.0,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 422
