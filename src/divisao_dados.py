import pandas as pd
from sklearn.model_selection import train_test_split


def executar_divisao():
    # Dados fictícios para aprendizagem
    X = pd.DataFrame({
        "total_atendimentos": [3, 1, 2, 0, 4, 1, 5, 2, 0, 3],
        "media_satisfacao": [1.5, 4.0, 3.0, 5.0, 2.0, 4.5, 1.0, 3.5, 5.0, 2.5]
    })

    y = pd.Series(
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        name="churn_30_dias"
    )

    # Separação dos dados: 80% treino, 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("X_TRAIN:", X_train.shape)
    print("X_TEST:", X_test.shape)
    print("Y_TRAIN:", y_train.shape)
    print("Y_TEST:", y_test.shape)
    print("INDICES TESTE:", X_test.index.tolist())
    print("TARGET TESTE:", y_test.tolist())


if __name__ == "__main__":
    executar_divisao()