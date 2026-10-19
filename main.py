import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn as sns



clientes = pd.DataFrame({
    "cliente_id": [1, 2, 3],
    "mensalidade": [100.0, 120.0, 150.0],
    "cancelou": [0, 1, 0]
})


print("Bibliotecas carregadas com sucesso!")
print()
print(clientes)
print()
print("Mensalidade média:", np.mean(clientes["mensalidade"]))
print("Versão do scikit-learn:", sklearn.__version__)