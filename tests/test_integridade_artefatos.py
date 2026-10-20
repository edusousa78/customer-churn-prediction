
import hashlib
import json
from pathlib import Path

import pytest

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_METADATA = RAIZ_PROJETO / "metadata" / "modelo_churn_v1.json"


def calcular_sha256(caminho):
    """Calcula o SHA-256 de um arquivo."""
    sha256 = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(65536), b""):
            sha256.update(bloco)

    return sha256.hexdigest().upper()


@pytest.fixture(scope="module")
def metadados():
    """Carrega os metadados registrados no projeto."""
    with ARQUIVO_METADATA.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


@pytest.mark.parametrize("artefato", ["dataset", "modelo"])
def test_integridade_artefato(metadados, artefato):
    """Confirma que o arquivo corresponde ao hash registrado."""
    informacoes = metadados[artefato]

    caminho = RAIZ_PROJETO / informacoes["arquivo"]
    hash_esperado = informacoes["sha256"]

    assert caminho.is_file(), f"Arquivo não encontrado: {caminho}"

    hash_atual = calcular_sha256(caminho)

    assert hash_atual == hash_esperado.upper(), (
        f"Integridade comprometida: {artefato}. "
        f"Esperado: {hash_esperado}. Obtido: {hash_atual}"
    )
