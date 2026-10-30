# Databricks notebook source
# MAGIC %md
# MAGIC # ⚙️ 00 - Configuração do workshop
# MAGIC
# MAGIC Este notebook é executado pelos outros com `%run ./00_config`. **Não precisa rodá-lo sozinho.**
# MAGIC
# MAGIC Se algo precisar ser alterado, altere **somente aqui**.
# MAGIC
# MAGIC > 🔐 O LLM (GPT-OSS-20B) é acessado **somente pelo Unity Gateway**, através de um *model service*
# MAGIC > criado no notebook `00_setup_unity_gateway`.

# COMMAND ----------

import os
import sys

import mlflow

# --- Unity Catalog: onde ficam as tabelas e o model service do workshop ---
CATALOG = "workspace"             # catálogo padrão do Databricks Free Edition
SCHEMA = "llmops_workshop"
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.dataset_avaliacao"

# --- LLM via Unity Gateway ---
MODEL_SERVICE_NAME = "assistente_ej_llm"
MODEL_SERVICE = f"{CATALOG}.{SCHEMA}.{MODEL_SERVICE_NAME}"     # o que passamos como `model` nas chamadas
DESTINATION_MODEL = "models/system.ai.gpt-oss-20b"  # modelo para onde o model service encaminha
os.environ["LLM_MODEL_SERVICE"] = MODEL_SERVICE

# --- Token pessoal (PAT) para chamar o Unity Gateway a partir dos notebooks ---
# Guardado em um secret scope pelo notebook 00_setup_unity_gateway (passo 3).
USUARIO = spark.sql("SELECT current_user()").first()[0]
SECRET_SCOPE = "llmops_" + USUARIO.split("@")[0].replace(".", "_").replace("-", "_")
SECRET_KEY = "ug_pat"
try:
    os.environ["DATABRICKS_UG_TOKEN"] = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
    _status_token = "✅ token carregado do secret scope"
except Exception:
    os.environ.pop("DATABRICKS_UG_TOKEN", None)
    _status_token = "⚠️ token não encontrado: rode o passo 3 do notebook 00_setup_unity_gateway"

# --- Experimento MLflow: onde ficam traces, avaliações e métricas ---
EXPERIMENT_PATH = f"/Users/{USUARIO}/llmops_workshop_ej"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_PATH).experiment_id

# --- Permite importar o código do agente que está na pasta ./app ---
APP_DIR = os.path.abspath("app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

print(f"Usuário:          {USUARIO}")
print(f"Tabela avaliação: {EVAL_TABLE}")
print(f"Model service:    {MODEL_SERVICE}  ->  {DESTINATION_MODEL}")
print(f"Token Unity GW:   {_status_token}")
print(f"Experimento:      {EXPERIMENT_PATH}  (id = {EXPERIMENT_ID})")
