import sqlite3
import time
import pandas as pd
from prometheus_client import start_http_server, Gauge
from pathlib import Path

# Configurações
DB_PATH = Path(__file__).parent.parent / "facape_despesas.db"
PORT = 8001

# Métricas
TOP_SUPPLIER_VOLUME = Gauge('facape_top_supplier_volume', 'Volume total empenhado por fornecedor', ['fornecedor'])
YEARLY_EMPENHADO = Gauge('facape_yearly_empenhado', 'Valor total empenhado por ano', ['ano'])
YEARLY_PAGO = Gauge('facape_yearly_pago', 'Valor total pago por ano', ['ano'])
YEARLY_EXEC_RATE = Gauge('facape_yearly_exec_rate', 'Taxa de execução média anual', ['ano'])

def update_metrics():
    if not DB_PATH.exists():
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Top 10 Fornecedores (2021-2025)
        df_top = pd.read_sql("""
            SELECT fornecedor, SUM(valor_empenhado) as total 
            FROM despesas 
            GROUP BY fornecedor 
            ORDER BY total DESC 
            LIMIT 10
        """, conn)
        for _, row in df_top.iterrows():
            TOP_SUPPLIER_VOLUME.labels(fornecedor=row['fornecedor']).set(row['total'])

        # 2. Empenhado vs Pago por Ano
        df_yearly = pd.read_sql("""
            SELECT ano, SUM(valor_empenhado) as empenhado, SUM(valor_pago) as pago, AVG(taxa_execucao) as taxa
            FROM despesas 
            GROUP BY ano
        """, conn)
        for _, row in df_yearly.iterrows():
            YEARLY_EMPENHADO.labels(ano=str(int(row['ano']))).set(row['empenhado'])
            YEARLY_PAGO.labels(ano=str(int(row['ano']))).set(row['pago'])
            YEARLY_EXEC_RATE.labels(ano=str(int(row['ano']))).set(row['taxa'])

        conn.close()
    except Exception as e:
        print(f"Erro ao atualizar métricas de negócio: {e}")

if __name__ == "__main__":
    start_http_server(PORT)
    print(f"Business Metrics Exporter rodando na porta {PORT}...")
    while True:
        update_metrics()
        time.sleep(300) # Atualiza a cada 5 minutos
