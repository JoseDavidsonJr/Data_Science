import sqlite3

def salvar_banco(
    df,
    previsoes,
    banco
):

    conn = sqlite3.connect(
        banco
    )

    df.to_sql(
        "despesas",
        conn,
        if_exists="replace",
        index=False
    )

    previsoes.to_sql(
        "previsoes",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print(
        "\nBanco SQLite salvo!"
    )