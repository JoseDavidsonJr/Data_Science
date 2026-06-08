import pandas as pd
import psycopg2
from datetime import datetime
from json import loads
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from prometheus_client import Gauge


# Configuração do Prometheus Gauge
# Criamos a métrica que o Prometheus vai "raspar" em tempo real
DRIFT_SCORE_GAUGE = Gauge(
    'facape_feature_drift_score', 
    'Score de drift estatístico (p-value) para as features da FACAPE', 
    ['feature_name']
)
DRIFT_DETECTED_GAUGE = Gauge(
    'facape_feature_drift_detected', 
    'Flag indicando se o drift foi detectado (1) ou não (0)', 
    ['feature_name']
)

def executar_pipeline_de_drift(dados_producao_df):
    """
    Executa o motor de cálculo do Evidently AI, expõe para o Prometheus
    e prepara a persistência para auditoria histórica no PostgreSQL.
    """
    # Carregar o seu baseline real (Dados de Treino de 2021 a 2023)
    # Como seu projeto lê JSONs, simulamos o carregamento da feature engineered
    # Em produção, você pode carregar um snapshot estático do seu conjunto de treino
    try:
        dados_treino_baseline = pd.read_json("data/baseline_treino_2021_2023.json")
    except FileNotFoundError:
        # Fallback de salvaguarda estruturada para o teste inicial
        dados_treino_baseline = pd.DataFrame({"hist_taxa_execucao": [0.85, 0.90, 0.75, 0.95, 0.88, 0.82]})

    # Executar o Motor de Cálculo do Evidently AI
    report_drift = Report(metrics=[
        ColumnDriftMetric(column_name="hist_taxa_execucao")
    ])
    
    report_drift.run(reference_data=dados_treino_baseline, current_data=dados_producao_df)
    resultado_dict = report_drift.as_dict()
    
    # Extração precisa das métricas extraídas pelo Evidently
    metrics_ref = resultado_dict['metrics'][0]['result']
    drift_score = metrics_ref['drift_score'] # p-value do teste estatístico
    drift_detectado = 1 if metrics_ref['drift_detected'] else 0

    # Atualizando o Prometheus (Tempo Real)
    DRIFT_SCORE_GAUGE.labels(feature_name="hist_taxa_execucao").set(drift_score)
    DRIFT_DETECTED_GAUGE.labels(feature_name="hist_taxa_execucao").set(drift_detectado)

    # Persistindo no Banco de Dados (Auditoria Histórica)
    timestamp_atual = datetime.utcnow()
    
    try:
        # Conexão utilizando as credenciais exatas do seu Docker-Compose
        conexao = psycopg2.connect(
            host="drift_db", # Nome do serviço contido no compose
            database="drift_metrics",
            user="admin",
            password="admin_password"
        )
        cursor = conexao.cursor()
        
        # Criação da tabela de auditoria se ela não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auditoria_drift_facape (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                feature VARCHAR(50) NOT NULL,
                score_p_value FLOAT NOT NULL,
                drift_detectado INT NOT NULL,
                volume_registros INT NOT NULL
            );
        """)
        
        # Inserindo o log imutável de auditoria
        query_insercao = """
            INSERT INTO auditoria_drift_facape (timestamp, feature, score_p_value, drift_detectado, volume_registros)
            VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(query_insercao, (timestamp_atual, "hist_taxa_execucao", drift_score, drift_detectado, len(dados_producao_df)))
        
        conexao.commit()
        cursor.close()
        conexao.close()
        print(f"[{timestamp_atual}] Log de auditoria persistido com sucesso no PostgreSQL.")
        
    except Exception as erro_banco:
        print(f"Erro ao persistir log de auditoria no PostgreSQL: {erro_banco}")

    return {"drift_score": drift_score, "drift_detectado": bool(drift_detectado)}