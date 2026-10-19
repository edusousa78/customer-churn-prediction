import sys
from pathlib import Path

import pytest


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_SRC = RAIZ_PROJETO / "src"

sys.path.insert(0, str(PASTA_SRC))

from prever_churn_validado import (
    carregar_modelo,
    prever_cliente,
)


@pytest.fixture(scope="module")
def configuracao_modelo():
    # Carregar o modelo uma única vez para estes testes
    modelo, features, threshold = carregar_modelo()

    return modelo, features, threshold


def test_carregamento_modelo(configuracao_modelo):
    modelo, features, threshold = configuracao_modelo

    assert hasattr(modelo, "predict_proba")
    assert len(features) == 3
    assert threshold == pytest.approx(0.20)


def test_probabilidade_entre_zero_e_um(configuracao_modelo):
    modelo, features, threshold = configuracao_modelo

    cliente = {
        "total_atendimentos": 3,
        "atendimentos_abertos": 1,
        "media_satisfacao": 3.5,
    }

    probabilidade, classificacao = prever_cliente(
        cliente,
        modelo,
        features,
        threshold,
    )

    assert 0 <= probabilidade <= 1
    assert classificacao in (0, 1)


def test_cliente_baixo_risco(configuracao_modelo):
    modelo, features, threshold = configuracao_modelo

    cliente = {
        "total_atendimentos": 1,
        "atendimentos_abertos": 0,
        "media_satisfacao": 4.8,
    }

    probabilidade, classificacao = prever_cliente(
        cliente,
        modelo,
        features,
        threshold,
    )

    assert probabilidade < threshold
    assert classificacao == 0


def test_cliente_alto_risco(configuracao_modelo):
    modelo, features, threshold = configuracao_modelo

    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    probabilidade, classificacao = prever_cliente(
        cliente,
        modelo,
        features,
        threshold,
    )

    assert probabilidade >= threshold
    assert classificacao == 1


def test_cliente_invalido_nao_recebe_previsao(
    configuracao_modelo,
):
    modelo, features, threshold = configuracao_modelo

    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.0,
    }

    with pytest.raises(
        ValueError,
        match="Atendimentos abertos não podem superar o total",
    ):
        prever_cliente(
            cliente,
            modelo,
            features,
            threshold,
        )
        