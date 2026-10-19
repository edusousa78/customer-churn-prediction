import sys
from pathlib import Path

import pytest

# Permitir importar os módulos da pasta src
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_SRC = RAIZ_PROJETO / "src"

sys.path.insert(0, str(PASTA_SRC))

from validar_entrada import validar_cliente


def test_cliente_valido():
    cliente = {
        "total_atendimentos": 3,
        "atendimentos_abertos": 1,
        "media_satisfacao": 4.2,
    }

    assert validar_cliente(cliente) is True


def test_atendimentos_abertos_maior_que_total():
    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.0,
    }

    with pytest.raises(
        ValueError,
        match="Atendimentos abertos não podem superar o total",
    ):
        validar_cliente(cliente)


def test_satisfacao_fora_da_escala():
    cliente = {
        "total_atendimentos": 4,
        "atendimentos_abertos": 1,
        "media_satisfacao": 7.5,
    }

    with pytest.raises(
        ValueError,
        match="Média de satisfação deve estar entre 1 e 5",
    ):
        validar_cliente(cliente)


def test_total_atendimentos_negativo():
    cliente = {
        "total_atendimentos": -1,
        "atendimentos_abertos": 0,
        "media_satisfacao": 2.5,
    }

    with pytest.raises(
        ValueError,
        match="Total de atendimentos deve ser inteiro e não negativo",
    ):
        validar_cliente(cliente)


def test_campo_obrigatorio_ausente():
    cliente = {
        "total_atendimentos": 2,
        "media_satisfacao": 4.0,
    }

    with pytest.raises(
        ValueError,
        match="Campo obrigatório ausente: atendimentos_abertos",
    ):
        validar_cliente(cliente)