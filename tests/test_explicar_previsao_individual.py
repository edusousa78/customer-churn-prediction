
import math
import sys
from pathlib import Path

import pytest

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from explicar_previsao_individual import explicar_previsao


def test_explicacao_reproduz_probabilidade_do_modelo():
    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    resultado = explicar_previsao(cliente)

    assert math.isclose(
        resultado["probabilidade"],
        0.9307,
        abs_tol=0.001,
    )

    assert resultado["classificacao"] == 1
    assert resultado["threshold"] == 0.20

    soma_contribuicoes = sum(
        item["contribuicao"]
        for item in resultado["contribuicoes"]
    )

    assert math.isclose(
        resultado["logit"],
        resultado["intercepto"] + soma_contribuicoes,
        abs_tol=1e-10,
    )


def test_explicacao_rejeita_cliente_invalido():
    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.0,
    }

    with pytest.raises(
        ValueError,
        match="Atendimentos abertos não podem superar o total",
    ):
        explicar_previsao(cliente)
