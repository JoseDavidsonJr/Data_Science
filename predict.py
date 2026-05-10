import pandas as pd

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.metrics import (
    mean_absolute_error
)

def prever_despesas(df):

    resumo_ano = (

        df.groupby("ano")[

            [
                "valor_empenhado",
                "valor_pago",
                "valor_liquidado",
                "valor_retido"
            ]
        ]

        .sum()
        .reset_index()
    )

    train = resumo_ano[
        resumo_ano["ano"] <= 2023
    ]

    test = resumo_ano[
        resumo_ano["ano"] >= 2024
    ]

    anos_futuros = pd.DataFrame({
        "ano": [2026, 2027]
    })

    resultado = anos_futuros.copy()

    for coluna in [

        "valor_empenhado",

        "valor_pago",

        "valor_liquidado",

        "valor_retido"
    ]:

        X_train = train[["ano"]]
        y_train = train[coluna]

        X_test = test[["ano"]]
        y_test = test[coluna]

        modelo = LinearRegression()

        modelo.fit(
            X_train,
            y_train
        )

        previsoes = modelo.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            previsoes
        )

        print(f"\n{coluna}")
        print(f"MAE: {mae:.2f}")

        futuro = modelo.predict(
            anos_futuros
        )

        resultado[coluna] = futuro

    print(resultado)

    return resultado