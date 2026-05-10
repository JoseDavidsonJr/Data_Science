import pandas as pd
import json
import os
import sqlite3
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error
)

# =====================================================
# CONFIGURAÇÃO
# =====================================================

pasta = input(
    "Digite o caminho da pasta com os JSONs: "
).strip().replace('"', '')
banco = input(
    "Digite o nome do banco SQLite: "
).strip()

dados_organizados = []

# =====================================================
# LEITURA DOS JSONS
# =====================================================

print("Arquivos encontrados:")
print(os.listdir(pasta))

for arquivo in os.listdir(pasta):

    if arquivo.lower().endswith(".json"):

        print(f"\nLendo arquivo: {arquivo}")

        caminho_arquivo = os.path.join(pasta, arquivo)

        ano = arquivo.lower().replace("despesas", "").replace(".json", "")
        ano = int("20" + ano)

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        for item in dados:

            fornecedor = (
                item.get("fornecedor", {})
                    .get("pessoa", {})
                    .get("nome")
            )

            valor_empenhado = item.get("valorEmpenhado")
            valor_pago = item.get("valorPago")
            valor_liquidado = item.get("valorLiquidado")
            valor_retido = item.get("valorRetido")

            if valor_empenhado is None:
                valor_empenhado = 0

            if valor_pago is None:
                valor_pago = 0

            if valor_liquidado is None:
                valor_liquidado = 0

            if valor_retido is None:
                valor_retido = 0

            taxa_execucao = 0

            if valor_empenhado > 0:
                taxa_execucao = valor_pago / valor_empenhado

            linha = {
                "id_registro": item.get("id"),
                "fornecedor": fornecedor,
                "ano": ano,
                "valor_empenhado": valor_empenhado,
                "valor_pago": valor_pago,
                "valor_liquidado": valor_liquidado,
                "valor_retido": valor_retido,
                "taxa_execucao": taxa_execucao
            }

            dados_organizados.append(linha)

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(dados_organizados)

print("\n========================")
print("DATAFRAME")
print("========================")

print(df.head())
print(df.info())

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\n========================")
print("FEATURE ENGINEERING")
print("========================")

historico = (
    df.groupby(["fornecedor", "ano"])["taxa_execucao"]
      .mean()
      .reset_index()
)

historico["media_execucao_historica"] = (
    historico.groupby("fornecedor")["taxa_execucao"]
    .shift(1)
)

historico["media_execucao_historica"] = (
    historico.groupby("fornecedor")["media_execucao_historica"]
    .transform(lambda x: x.expanding().mean())
)

df = df.merge(
    historico[
        [
            "fornecedor",
            "ano",
            "media_execucao_historica"
        ]
    ],
    on=["fornecedor", "ano"],
    how="left"
)

df["media_execucao_historica"] = (
    df["media_execucao_historica"]
    .fillna(0)
)

# =====================================================
# FORNECEDOR RECORRENTE
# =====================================================

anos_fornecedor = (
    df.groupby("fornecedor")["ano"]
    .nunique()
    .reset_index()
)

anos_fornecedor["recorrente"] = (
    anos_fornecedor["ano"]
    .apply(lambda x: 1 if x > 1 else 0)
)

df = df.merge(
    anos_fornecedor[
        ["fornecedor", "recorrente"]
    ],
    on="fornecedor",
    how="left"
)

# =====================================================
# TREINO / TESTE TEMPORAL
# =====================================================

train = df[df["ano"] <= 2023]
test = df[df["ano"] >= 2024]

features = [
    "valor_empenhado",
    "valor_pago",
    "valor_liquidado",
    "valor_retido",
    "taxa_execucao",
    "media_execucao_historica"
]

X_train = train[features]
y_train = train["recorrente"]

X_test = test[features]
y_test = test["recorrente"]

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_experiment("facape_despesas")

# =====================================================
# MODELO CLASSIFICADOR
# =====================================================

print("\n========================")
print("MODELO CLASSIFICADOR")
print("========================")

for n_estimators in [10, 30, 50, 100, 200]:

    with mlflow.start_run():

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
            ))
        ])

        pipe.fit(X_train, y_train)

        previsoes = pipe.predict(X_test)

        acc = accuracy_score(y_test, previsoes)

        print(f"\nRandomForest ({n_estimators})")
        print(f"Acurácia: {acc:.4f}")

        print(classification_report(y_test, previsoes))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(pipe, "modelo_randomforest")

# =====================================================
# RESUMO ANUAL
# =====================================================

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

print("\n========================")
print("RESUMO ANUAL")
print("========================")

print(resumo_ano)

# =====================================================
# MODELO DE PREVISÃO TEMPORAL
# =====================================================

print("\n========================")
print("PREVISÃO TEMPORAL")
print("========================")

train_temporal = resumo_ano[
    resumo_ano["ano"] <= 2023
]

test_temporal = resumo_ano[
    resumo_ano["ano"] >= 2024
]

anos_futuros = pd.DataFrame({
    "ano": [2026, 2027]
})

resultado_previsoes = anos_futuros.copy()

for coluna in [
    "valor_empenhado",
    "valor_pago",
    "valor_liquidado",
    "valor_retido"
]:

    X_train = train_temporal[["ano"]]
    y_train = train_temporal[coluna]

    X_test = test_temporal[["ano"]]
    y_test = test_temporal[coluna]

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, previsoes)

    print(f"\n{coluna}")
    print(f"MAE: {mae:.2f}")

    futuro = modelo.predict(anos_futuros)

    resultado_previsoes[coluna] = futuro

print("\n========================")
print("PREVISÕES FUTURAS")
print("========================")

print(resultado_previsoes)

# =====================================================
# FORNECEDORES RECORRENTES
# =====================================================

print("\n========================")
print("FORNECEDORES RECORRENTES")
print("========================")

fornecedores_recorrentes = (
    anos_fornecedor[
        anos_fornecedor["recorrente"] == 1
    ]
)

print(fornecedores_recorrentes)

# =====================================================
# SQLITE
# =====================================================

print("\n========================")
print("SALVANDO SQLITE")
print("========================")

conn = sqlite3.connect(banco)

df.to_sql(
    "despesas",
    conn,
    if_exists="replace",
    index=False
)

resumo_ano.to_sql(
    "resumo_anual",
    conn,
    if_exists="replace",
    index=False
)

resultado_previsoes.to_sql(
    "previsoes_futuras",
    conn,
    if_exists="replace",
    index=False
)

fornecedores_recorrentes.to_sql(
    "fornecedores_recorrentes",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\nBanco salvo com sucesso!")

print("\n========================")
print("EXECUÇÃO FINALIZADA")
print("========================")
