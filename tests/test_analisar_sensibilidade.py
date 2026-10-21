
import sys
from pathlib import Path

import pytest

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from analisar_sensibilidade import analisar_cenario


def test_aumento_satisfacao_reduz_probabilidade():
    cliente_original = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    cliente_melhorado = {
        **cliente_original,
        "media_satisfacao": 4.0,
    }

    original = analisar_cenario("Original", cliente_original)
    melhorado = analisar_cenario("Satisfacao maior", cliente_melhorado)

    assert melhorado["probabilidade"] < original["probabilidade"]


def test_reducao_pendencias_reduz_probabilidade():
    cliente_original = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    cliente_melhorado = {
        **cliente_original,
        "atendimentos_abertos": 1,
    }

    original = analisar_cenario("Original", cliente_original)
    melhorado = analisar_cenario("Menos pendencias", cliente_melhorado)

    assert melhorado["probabilidade"] < original["probabilidade"]


def test_cenario_combinado_remove_alerta():
    cliente = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 1,
        "media_satisfacao": 4.0,
    }

    resultado = analisar_cenario("Cenario combinado", cliente)

    assert resultado["probabilidade"] == pytest.approx(
        0.1717,
        abs=0.001,
    )
    assert resultado["classificacao"] == 0


def test_cliente_invalido_e_rejeitado():
    cliente = {
        "total_atendimentos": 2,
        "atendimentos_abertos": 5,
        "media_satisfacao": 3.0,
    }

    with pytest.raises(ValueError):
        analisar_cenario("Cliente invalido", cliente)
