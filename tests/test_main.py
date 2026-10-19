import pandas as pd

def test_calculo_mensalidade_media():
    # Dados de teste
    mensalidades = pd.Series([100.0, 120.0, 150.0])
    # Cálculo da média
    media_calculada = mensalidades.mean()
    # Valor esperado
    assert round(media_calculada, 2) == 123.33
    