import pandas as pd
import json
import os
import sqlite3
import mlflow
import mlflow.sklearn
import joblib

from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    f1_score
)

# ========================================
# CONFIG
# ========================================

pasta = input(
    "Digite o caminho da pasta dos JSONs: "
).strip().replace('"', '')

banco = input(
    "Digite o nome do banco SQLite: "
).strip()

os.makedirs(
    "models",
    exist_ok=True
)

dados = []

# ========================================
# LEITURA
# ========================================

for arquivo in os.listdir(pasta):

    if arquivo.lower().endswith(".json"):

        caminho = os.path.join(
            pasta,
            arquivo
        )

        ano = arquivo.lower() \
            .replace("despesas", "") \
            .replace(".json", "")

        ano = int("20" + ano)

        with open(
            caminho,
            "r",
            encoding="utf-8"
        ) as f:

            json_data = json.load(f)

        for item in json_data:

            fornecedor = (
                item.get("fornecedor", {})
                    .get("pessoa", {})
                    .get("nome")
            )

            valor_empenhado = (
                item.get("valorEmpenhado") or 0
            )

            valor_pago = (
                item.get("valorPago") or 0
            )

            valor_liquidado = (
                item.get("valorLiquidado") or 0
            )

            valor_retido = (
                item.get("valorRetido") or 0
            )

            valor_anulado = (
                item.get("valorAnulado") or 0
            )

            saldo = (
                item.get("valorSaldoAPagar") or 0
            )

            taxa_execucao = 0

            if valor_empenhado > 0:

                taxa_execucao = (
                    valor_pago /
                    valor_empenhado
                )

            dados.append({

                "ano": ano,

                "fornecedor": fornecedor,

                "valor_empenhado":
                    valor_empenhado,

                "valor_pago":
                    valor_pago,

                "valor_liquidado":
                    valor_liquidado,

                "valor_retido":
                    valor_retido,

                "valor_anulado":
                    valor_anulado,

                "saldo":
                    saldo,

                "taxa_execucao":
                    taxa_execucao
            })

df = pd.DataFrame(dados)

# ========================================
# FEATURE ENGINEERING
# ========================================

historico = (
    df.groupby(
        ["fornecedor", "ano"]
    )["taxa_execucao"]
    .mean()
    .reset_index()
)

historico["hist_execucao"] = (
    historico.groupby("fornecedor")
    ["taxa_execucao"]
    .shift(1)
)

historico["hist_execucao"] = (
    historico.groupby("fornecedor")
    ["hist_execucao"]
    .transform(
        lambda x:
        x.expanding().mean()
    )
)

historico = historico.fillna(0)

df = df.merge(

    historico[
        [
            "fornecedor",
            "ano",
            "hist_execucao"
        ]
    ],

    on=["fornecedor", "ano"],
    how="left"
)

# ========================================
# TARGET
# ========================================

df["target"] = (
    (
        df["taxa_execucao"] < 0.70
    )
    |
    (
        df["valor_anulado"] > 0
    )
).astype(int)

# ========================================
# SPLIT TEMPORAL
# ========================================

train = df[
    df["ano"] <= 2023
]

test = df[
    df["ano"] >= 2024
]

# ========================================
# DRIFT
# ========================================

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    reference_data=train,
    current_data=test
)

snapshot.save_html(
    "drift_report.html"
)

# ========================================
# FEATURES
# ========================================

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

# ========================================
# MLFLOW
# ========================================

mlflow.set_experiment(
    "facape_pipeline"
)

melhor_modelo = None
melhor_f1 = 0

# ========================================
# TREINO
# ========================================

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

# ========================================
# VERSIONAMENTO
# ========================================

versao = datetime.now() \
    .strftime("%Y%m%d_%H%M%S")

nome_modelo = (
    f"models/modelo_{versao}.pkl"
)

joblib.dump(
    melhor_modelo,
    nome_modelo
)

# ========================================
# SQLITE
# ========================================

conn = sqlite3.connect(banco)

df.to_sql(
    "despesas",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\nPipeline finalizado!")