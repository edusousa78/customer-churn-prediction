import math


def validar_cliente(cliente):
    campos_obrigatorios = [
        "total_atendimentos",
        "atendimentos_abertos",
        "media_satisfacao",
    ]

    # 1. Verificar campos obrigatórios
    for campo in campos_obrigatorios:
        if campo not in cliente:
            raise ValueError(
                f"Campo obrigatório ausente: {campo}"
            )

    # 2. Verificar tipos e valores numéricos
    for campo in campos_obrigatorios:
        valor = cliente[campo]

        if (
            isinstance(valor, bool)
            or not isinstance(valor, (int, float))
            or not math.isfinite(valor)
        ):
            raise ValueError(
                f"Valor numérico inválido: {campo}"
            )

    total = cliente["total_atendimentos"]
    abertos = cliente["atendimentos_abertos"]
    satisfacao = cliente["media_satisfacao"]

    # 3. Verificar contagens inteiras e não negativas
    if total < 0 or not float(total).is_integer():
        raise ValueError(
            "Total de atendimentos deve ser inteiro e não negativo."
        )

    if abertos < 0 or not float(abertos).is_integer():
        raise ValueError(
            "Atendimentos abertos devem ser inteiros e não negativos."
        )

    # 4. Verificar consistência
    if abertos > total:
        raise ValueError(
            "Atendimentos abertos não podem superar o total."
        )

    if not 1 <= satisfacao <= 5:
        raise ValueError(
            "Média de satisfação deve estar entre 1 e 5."
        )

    return True


def main():
    clientes_teste = [
        {
            "total_atendimentos": 3,
            "atendimentos_abertos": 1,
            "media_satisfacao": 4.2,
        },
        {
            "total_atendimentos": 2,
            "atendimentos_abertos": 5,
            "media_satisfacao": 3.0,
        },
        {
            "total_atendimentos": 4,
            "atendimentos_abertos": 1,
            "media_satisfacao": 7.5,
        },
        {
            "total_atendimentos": -1,
            "atendimentos_abertos": 0,
            "media_satisfacao": 2.5,
        },
        {
            "total_atendimentos": 2,
            "atendimentos_abertos": 0,
            "media_satisfacao": 4.0,
        },
    ]

    print("=== VALIDAÇÃO DE DADOS ===")

    for indice, cliente in enumerate(
        clientes_teste,
        start=1,
    ):
        try:
            validar_cliente(cliente)
            print(f"Cliente {indice}: VÁLIDO")

        except ValueError as erro:
            print(f"Cliente {indice}: INVÁLIDO")
            print(f"Motivo: {erro}")


# Executa os testes somente quando este arquivo
# é chamado diretamente pelo Python.
if __name__ == "__main__":
    main()