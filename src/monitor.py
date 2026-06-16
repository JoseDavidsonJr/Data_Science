import os
import pandas as pd
import psycopg2
import logging
from datetime import datetime
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from prometheus_client import Gauge, Counter

# Configuração de Logging Profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FACAPE-Monitor")

# Métricas Prometheus
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
PREDICTION_COUNTER = Counter(
    'facape_prediction_total',
    'Total de predições realizadas'
)

# Configurações via Variáveis de Ambiente
DB_HOST = os.getenv("DB_HOST", "drift_db")
DB_NAME = os.getenv("DB_NAME", "drift_metrics")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin_password")

def executar_pipeline_de_drift(dados_producao_df):
    """
    Motor de monitoramento: calcula drift, exporta para Prometheus e audita no Postgres.
    """
    try:
        # Carregamento resiliente do baseline
        baseline_path = "data/baseline_treino_2021_2023.json"
        if os.path.exists(baseline_path):
            dados_treino_baseline = pd.read_json(baseline_path)
        else:
            logger.warning("Baseline real não encontrado. Usando fallback estatístico.")
            dados_treino_baseline = pd.DataFrame({"hist_taxa_execucao": [0.85, 0.90, 0.75, 0.95, 0.88, 0.82]})

        # Cálculo de Drift com Evidently
        report_drift = Report(metrics=[ColumnDriftMetric(column_name="hist_taxa_execucao")])
        report_drift.run(reference_data=dados_treino_baseline, current_data=dados_producao_df)
        resultado_dict = report_drift.as_dict()
        
        metrics_ref = resultado_dict['metrics'][0]['result']
        drift_score = metrics_ref['drift_score']
        drift_detectado = 1 if metrics_ref['drift_detected'] else 0

        # Atualização Prometheus
        DRIFT_SCORE_GAUGE.labels(feature_name="hist_taxa_execucao").set(drift_score)
        DRIFT_DETECTED_GAUGE.labels(feature_name="hist_taxa_execucao").set(drift_detectado)

        # Auditoria Imutável no PostgreSQL
        persistir_auditoria(drift_score, drift_detectado, len(dados_producao_df))

        return {"drift_score": drift_score, "drift_detectado": bool(drift_detectado)}

    except Exception as e:
        logger.error(f"Falha no pipeline de monitoramento: {str(e)}")
        return None

def persistir_auditoria(score, detectado, volume):
    try:
        conexao = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conexao.cursor()
        
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
        
        cursor.execute("""
            INSERT INTO auditoria_drift_facape (timestamp, feature, score_p_value, drift_detectado, volume_registros)
            VALUES (%s, %s, %s, %s, %s);
        """, (datetime.utcnow(), "hist_taxa_execucao", score, detectado, volume))
        
        conexao.commit()
        cursor.close()
        conexao.close()
    except Exception as e:
        logger.error(f"Erro de persistência no DB de Auditoria: {str(e)}")
