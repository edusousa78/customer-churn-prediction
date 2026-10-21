
from explicar_previsao_individual import explicar_previsao


def analisar_cenario(nome, cliente):
    """Calcula e apresenta a previsão de um cenário."""
    resultado = explicar_previsao(cliente)

    return {
        "cenario": nome,
        "probabilidade": resultado["probabilidade"],
        "classificacao": resultado["classificacao"],
    }


def main():
    cliente_base = {
        "total_atendimentos": 7,
        "atendimentos_abertos": 4,
        "media_satisfacao": 1.5,
    }

    cenarios = [
        (
            "Cliente original",
            cliente_base.copy(),
        ),
        (
            "Satisfacao aumentada",
            {
                **cliente_base,
                "media_satisfacao": 4.0,
            },
        ),
        (
            "Menos atendimentos abertos",
            {
                **cliente_base,
                "atendimentos_abertos": 1,
            },
        ),
        (
            "Satisfacao maior e menos pendencias",
            {
                **cliente_base,
                "atendimentos_abertos": 1,
                "media_satisfacao": 4.0,
            },
        ),
    ]

    print("=== ANALISE DE SENSIBILIDADE DE CHURN ===")

    resultados = []

    for nome, cliente in cenarios:
        resultado = analisar_cenario(nome, cliente)
        resultados.append(resultado)

        print(f"\nCenario: {nome}")
        print(f"  Total de atendimentos: {cliente['total_atendimentos']}")
        print(f"  Atendimentos abertos: {cliente['atendimentos_abertos']}")
        print(f"  Satisfacao: {cliente['media_satisfacao']}")
        print(
            "  Probabilidade de churn: "
            f"{resultado['probabilidade'] * 100:.2f}%"
        )
        print(f"  Classificacao: {resultado['classificacao']}")

    probabilidade_base = resultados[0]["probabilidade"]

    print("\n=== COMPARACAO COM O CLIENTE ORIGINAL ===")

    for resultado in resultados[1:]:
        diferenca = (
            resultado["probabilidade"] - probabilidade_base
        ) * 100

        print(
            f"{resultado['cenario']}: "
            f"{diferenca:+.2f} pontos percentuais"
        )

    print(
        "\nNota: os cenarios representam simulacoes do modelo, "
        "nao efeitos causais comprovados."
    )


if __name__ == "__main__":
    main()
