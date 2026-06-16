"""Lê os JSONs de despesas FACAPE e retorna um DataFrame limpo."""
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


def load_jsons() -> pd.DataFrame:
    rows = []
    # Busca tanto os originais quanto os novos via MCP
    files = sorted(ROOT.glob("despesas??.JSON")) + sorted(ROOT.glob("despesas??_mcp.JSON"))
    for f in files:
        ano = int("20" + f.name.replace("despesas", "")[:2])
        for item in json.loads(f.read_text(encoding="utf-8")):
            rows.append(
                {
                    "id": item.get("id"),
                    "fornecedor": item.get("fornecedor", {})
                    .get("pessoa", {})
                    .get("nome"),
                    "cpf_cnpj": item.get("fornecedor", {})
                    .get("pessoa", {})
                    .get("cpfCnpj"),
                    "ano": ano,
                    "valor_empenhado": item.get("valorEmpenhado") or 0.0,
                    "valor_liquidado": item.get("valorLiquidado") or 0.0,
                    "valor_pago": item.get("valorPago") or 0.0,
                    "valor_anulado": item.get("valorAnulado") or 0.0,
                    "valor_retido": item.get("valorRetido") or 0.0,
                    "valor_saldo_a_pagar": item.get("valorSaldoAPagar") or 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["fornecedor", "cpf_cnpj"])
    df = df[df["valor_empenhado"] > 0].copy()

    # tipo: 0 = PF (CPF 11 dígitos), 1 = PJ (CNPJ 14 dígitos)
    df["tipo"] = df["cpf_cnpj"].apply(
        lambda x: 0
        if len(str(x).replace(".", "").replace("-", "").replace("/", "").strip()) == 11
        else 1
    )

    # Taxas derivadas
    df["taxa_execucao"] = df["valor_pago"] / df["valor_empenhado"]
    df["perc_anulado"] = df["valor_anulado"] / df["valor_empenhado"]
    df["perc_retido"] = df["valor_retido"] / df["valor_empenhado"]
    df["perc_saldo"] = df["valor_saldo_a_pagar"] / df["valor_empenhado"]

    return df.reset_index(drop=True)


def save_to_sqlite(df: pd.DataFrame, db_path: Path = ROOT / "facape_despesas.db") -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql("despesas", conn, if_exists="replace", index=False)
    print(f"Salvo: {len(df)} registros em {db_path}")


if __name__ == "__main__":
    df = load_jsons()
    save_to_sqlite(df)
    print(df[["fornecedor", "ano", "valor_empenhado", "taxa_execucao"]].describe())
