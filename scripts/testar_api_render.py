
import json
import urllib.error
import urllib.request

URL = "https://conecta-brasil-churn-api.onrender.com/prever"


def enviar_cliente(cliente):
    dados = json.dumps(cliente).encode("utf-8")

    requisicao = urllib.request.Request(
        URL,
        data=dados,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=120) as resposta:
            return resposta.status, json.load(resposta)

    except urllib.error.HTTPError as erro:
        return erro.code, json.load(erro)


def testar_previsao_valida():
    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    status, resultado = enviar_cliente(cliente)

    assert status == 200, f"HTTP inesperado: {status}"
    assert 0 <= resultado["probabilidade_churn"] <= 1
    assert resultado["classificacao"] == 1
    assert resultado["alerta"] is True
    assert resultado["threshold"] == 0.2

    print("APROVADO: Previsão válida — HTTP 200")
    print(
        "Probabilidade de churn:",
        resultado["probabilidade_churn"],
    )


def testar_dados_inconsistentes():
    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.5,
    }

    status, resultado = enviar_cliente(cliente)

    assert status == 422, f"HTTP inesperado: {status}"
    assert "Atendimentos abertos" in resultado["detail"]

    print("APROVADO: Dados inconsistentes — HTTP 422")


if __name__ == "__main__":
    testar_previsao_valida()
    testar_dados_inconsistentes()

    print("\nTODOS OS TESTES DA API PÚBLICA PASSARAM.")
