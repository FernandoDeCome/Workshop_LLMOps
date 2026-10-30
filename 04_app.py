# Databricks notebook source
# MAGIC %md
# MAGIC # 🖥️ 04 - Publicando o Agente em um Databricks App
# MAGIC
# MAGIC ## O que é um Databricks App?
# MAGIC Um site hospedado dentro do Databricks, com login e permissões já integrados.
# MAGIC Não precisamos de servidor, domínio ou senha de API.
# MAGIC
# MAGIC Nosso App tem duas partes:
# MAGIC - **Backend em Python (FastAPI)**: recebe a pergunta, chama o agente e devolve a resposta. São só 3 rotas (`/api/chat`, `/api/feedback`, `/api/info`).
# MAGIC - **Interface em HTML + CSS** (com um pouco de JavaScript para enviar as mensagens): sem build, sem TypeScript.
# MAGIC
# MAGIC > ⏱️ **No workshop:** o instrutor já publicou o App antes da aula. Abra a URL, converse e dê 👍/👎.
# MAGIC > O passo a passo abaixo é para você reproduzir depois, no seu próprio workspace.
# MAGIC
# MAGIC ## Arquitetura escolhida: agente **dentro** do App
# MAGIC ```
# MAGIC  Navegador (HTML/CSS) ──► FastAPI (app.py + agente.py) ──► Unity Gateway ──► model service ──► GPT-OSS-20B
# MAGIC                         │
# MAGIC                         └──► MLflow (traces, tokens, feedback 👍/👎)
# MAGIC ```
# MAGIC O App importa o **mesmo `agente.py`** que avaliamos no notebook 03 — o que foi avaliado é o que vai para o usuário.
# MAGIC
# MAGIC 📂 Arquivos na pasta `app/`:
# MAGIC | Arquivo | Função |
# MAGIC |---|---|
# MAGIC | `app.py` | Backend FastAPI (rotas da API e página inicial) |
# MAGIC | `static/` | Interface: `index.html`, `style.css`, `app.js` |
# MAGIC | `agente.py` | O agente (mesmo código dos notebooks) |
# MAGIC | `base_conhecimento.json` | Dados da empresa |
# MAGIC | `app.yaml` | Como iniciar o App e quais variáveis/recursos ele usa |
# MAGIC | `requirements.txt` | Bibliotecas Python |

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.4" openai databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

print(f"Caminho do código do App: {APP_DIR}")
print(f"Model service (recurso 'model'):     {MODEL_SERVICE}")
print(f"Experimento (recurso 'experiment'):  {EXPERIMENT_PATH}  (id = {EXPERIMENT_ID})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo a passo (pela interface)
# MAGIC
# MAGIC ### 1. Criar o App
# MAGIC 1. Menu lateral **+ New → App** (ou **Compute → Apps → Create app**).
# MAGIC 2. Escolha **Create a custom app**. Nome: `assistente-ej` (só letras minúsculas, números e `-`).
# MAGIC
# MAGIC ### 2. Adicionar os recursos (App resources)
# MAGIC O App roda com uma **identidade própria** (service principal). Precisamos dar a ela acesso ao model service do
# MAGIC Unity Gateway e ao MLflow. No App **não usamos PAT**: o agente pega automaticamente o token OAuth do service principal.
# MAGIC
# MAGIC | Tipo de recurso | Selecionar | Permissão | Resource key (**exatamente** assim) |
# MAGIC |---|---|---|---|
# MAGIC | Unity Gateway model service | `workspace.llmops_workshop.assistente_ej_llm` | Can query (EXECUTE) | `model` |
# MAGIC | MLflow experiment | `llmops_workshop_ej` | Can edit | `experiment` |
# MAGIC
# MAGIC > A *resource key* é o nome usado no `app.yaml` (`valueFrom: model`). Se forem diferentes, o App não acha o recurso.
# MAGIC
# MAGIC ### 3. Fazer o deploy
# MAGIC Clique em **Deploy** e informe como *source code path* a pasta `app` impressa acima
# MAGIC (`/Workspace/Users/<seu-email>/.../app`). O primeiro deploy leva alguns minutos (instala as bibliotecas).
# MAGIC
# MAGIC ### 4. Testar
# MAGIC Abra a URL do App, faça perguntas e dê 👍/👎. Depois vá em **Experiments → llmops_workshop_ej → Traces**
# MAGIC e filtre pela tag `origem = app`: lá estão as conversas reais, com tokens e feedback.
# MAGIC
# MAGIC ⚠️ **Free Edition:** Apps param automaticamente após 24h. Antes do workshop, clique em **Start** no App.

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Opcional) Deploy via código
# MAGIC Depois de **criar o App e configurar os recursos** pela interface (passos 1 e 2), dá para refazer o deploy
# MAGIC por código — útil quando alteramos o `agente.py`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment

NOME_APP = "assistente-ej"

w = WorkspaceClient()
deploy = w.apps.deploy_and_wait(app_name=NOME_APP, app_deployment=AppDeployment(source_code_path=APP_DIR))
print(f"Status: {deploy.status.state} - {deploy.status.message}")
print(f"URL:    {w.apps.get(NOME_APP).url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔁 O ciclo de LLMOps fecha aqui
# MAGIC 1. Usuários conversam no App → **traces + feedback** no MLflow.
# MAGIC 2. Respostas com 👎 viram **novos exemplos** no dataset de avaliação (notebook 01).
# MAGIC 3. Ajustamos prompt/modelo/ferramentas e **avaliamos de novo** (notebook 03).
# MAGIC 4. Se melhorou, fazemos **deploy** da nova versão (este notebook).
# MAGIC 5. Acompanhamos **custos e tokens** (notebook 05), pelo MLflow e pelo `system.ai_gateway.usage` do Unity Gateway.
