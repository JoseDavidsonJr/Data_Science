import mlflow
import mlflow.sklearn
import joblib
import os

from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

def treinar_modelo(df):

    train = df[
        df["ano"] <= 2023
    ]

    test = df[
        df["ano"] >= 2024
    ]

    features = [

        "valor_empenhado",

        "valor_pago",

        "valor_liquidado",

        "valor_retido",

        "taxa_execucao",

        "hist_execucao"
    ]

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    mlflow.set_experiment(
        "facape_pipeline"
    )

    melhor_modelo = None
    melhor_f1 = 0

    for n_estimators in [10, 30, 50, 100]:

        with mlflow.start_run():

            pipe = Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42
                    )
                )
            ])

            pipe.fit(
                X_train,
                y_train
            )

            previsoes = pipe.predict(
                X_test
            )

            acc = accuracy_score(
                y_test,
                previsoes
            )

            f1 = f1_score(
                y_test,
                previsoes
            )

            print(f"\nModelo: {n_estimators}")

            print(f"Acurácia: {acc:.4f}")

            print(f"F1: {f1:.4f}")

            print(
                classification_report(
                    y_test,
                    previsoes
                )
            )

            mlflow.log_param(
                "n_estimators",
                n_estimators
            )

            mlflow.log_metric(
                "accuracy",
                acc
            )

            mlflow.log_metric(
                "f1",
                f1
            )

            mlflow.sklearn.log_model(
                pipe,
                "modelo"
            )

            if f1 > melhor_f1:

                melhor_f1 = f1
                melhor_modelo = pipe

    os.makedirs(
        "models",
        exist_ok=True
    )

    versao = datetime.now() \
        .strftime("%Y%m%d_%H%M%S")

    nome_modelo = (
        f"models/modelo_{versao}.pkl"
    )

    joblib.dump(
        melhor_modelo,
        nome_modelo
    )

    print(f"\nModelo salvo: {nome_modelo}")

    return melhor_modelo