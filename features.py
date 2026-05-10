import pandas as pd
import json
import os

def carregar_dados(pasta):

    dados = []

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

    df["target"] = (
        (
            df["taxa_execucao"] < 0.70
        )
        |
        (
            df["valor_anulado"] > 0
        )
    ).astype(int)

    return df