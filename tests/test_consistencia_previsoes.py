
import math
import sys
from pathlib import Path

import pytest

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from explicar_previsao_individual import explicar_previsao


def criar_cliente(satisfacao=3.0, abertos=2):
    """Cria um cliente sintético válido para os testes."""
    return {
        "total_atendimentos": 7,
        "atendimentos_abertos": abertos,
        "media_satisfacao": satisfacao,
    }


@pytest.mark.parametrize(
    "satisfacao_inicial,satisfacao_final",
    [
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
    ],
)
def test_aumento_satisfacao_reduz_risco(
    satisfacao_inicial,
    satisfacao_final,
):
    cliente_inicial = criar_cliente(
        satisfacao=satisfacao_inicial
    )
    cliente_final = criar_cliente(
        satisfacao=satisfacao_final
    )

    resultado_inicial = explicar_previsao(cliente_inicial)
    resultado_final = explicar_previsao(cliente_final)

    assert (
        resultado_final["probabilidade"]
        < resultado_inicial["probabilidade"]
    )


def test_mesmo_cliente_mesma_probabilidade():
    cliente = criar_cliente(satisfacao=2.5, abertos=3)

    primeira_previsao = explicar_previsao(cliente)
    segunda_previsao = explicar_previsao(cliente)

    assert math.isclose(
        primeira_previsao["probabilidade"],
        segunda_previsao["probabilidade"],
        rel_tol=0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "satisfacao,abertos",
    [
        (1.0, 0),
        (2.0, 1),
        (3.0, 2),
        (4.0, 3),
        (5.0, 4),
    ],
)
def test_classificacao_respeita_threshold(
    satisfacao,
    abertos,
):
    cliente = criar_cliente(
        satisfacao=satisfacao,
        abertos=abertos,
    )

    resultado = explicar_previsao(cliente)

    classificacao_esperada = int(
        resultado["probabilidade"] >= resultado["threshold"]
    )

    assert resultado["threshold"] == 0.20
    assert resultado["classificacao"] == classificacao_esperada


def test_probabilidade_esta_entre_zero_e_um():
    cliente = criar_cliente(satisfacao=3.0, abertos=2)

    resultado = explicar_previsao(cliente)

    assert 0.0 <= resultado["probabilidade"] <= 1.0
