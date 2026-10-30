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
# MAGIC > ⏱️ Neste notebook você cria e publica **o seu** App com uma célula de código. O primeiro deploy leva de 3 a 5 minutos.
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
# MAGIC ## 1. Criar e publicar o App
# MAGIC O App roda com uma **identidade própria** (service principal). Ao criá-lo, dizemos a quais recursos essa identidade
# MAGIC pode acessar. No App **não usamos PAT**: o agente pega automaticamente o token OAuth do service principal.
# MAGIC
# MAGIC | Recurso | O que é | Permissão | Resource key |
# MAGIC |---|---|---|---|
# MAGIC | Unity Gateway model service | `workspace.llmops_workshop.assistente_ej_llm` | `EXECUTE` | `model` |
# MAGIC | MLflow experiment | `llmops_workshop_ej` | `CAN_EDIT` | `experiment` |
# MAGIC
# MAGIC A *resource key* é o nome usado no `app.yaml` (`valueFrom: model`). É por ela que o App descobre qual model service
# MAGIC e qual experimento usar.
# MAGIC
# MAGIC A célula abaixo cria o App (ou atualiza, se ele já existir) e faz o deploy do código da pasta `app/`.
# MAGIC Pode rodar de novo sempre que alterar o `agente.py`: ela só publica a nova versão.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.apps import App, AppDeployment

NOME_APP = "assistente-ej"  # só letras minúsculas, números e "-"

w = WorkspaceClient()
definicao = App.from_dict({
    "name": NOME_APP,
    "description": "Workshop LLMOps: Assistente Engenhar Jr. (GPT-OSS-20B via Unity Gateway)",
    "resources": [
        {"name": "model", "uc_securable": {"securable_full_name": MODEL_SERVICE,
                                            "securable_type": "MODEL_SERVICE", "permission": "EXECUTE"}},
        {"name": "experiment", "experiment": {"experiment_id": EXPERIMENT_ID, "permission": "CAN_EDIT"}},
    ],
})

try:
    w.apps.get(NOME_APP)
    w.apps.update(name=NOME_APP, app=definicao)
    print(f"• App {NOME_APP} já existia: recursos atualizados")
except NotFound:
    print(f"Criando o App {NOME_APP} (1 a 2 minutos)...")
    w.apps.create_and_wait(app=definicao)
    print("✅ App criado")

if w.apps.get(NOME_APP).compute_status.state.value != "ACTIVE":  # App parado (ex.: após 24h no Free Edition)
    print("Ligando o App...")
    w.apps.start_and_wait(NOME_APP)

print("Publicando o código (3 a 5 minutos no primeiro deploy)...")
deploy = w.apps.deploy_and_wait(app_name=NOME_APP, app_deployment=AppDeployment(source_code_path=APP_DIR))
print(f"Status: {deploy.status.state.value} - {deploy.status.message}")
print(f"\n👉 Abra o seu App: {w.apps.get(NOME_APP).url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Testar
# MAGIC Abra a URL impressa acima, faça perguntas e dê 👍/👎. Depois vá em **Experiments → llmops_workshop_ej → Traces**
# MAGIC e filtre pela tag `origem = app`: lá estão as suas conversas, com tokens e feedback.
# MAGIC
# MAGIC > Se o deploy falhar, veja os logs em **Compute → Apps → assistente-ej → Logs**.
# MAGIC
# MAGIC ⚠️ **Free Edition:** Apps param automaticamente após 24h. Para ligar de novo, rode a célula acima
# MAGIC ou clique em **Start** em **Compute → Apps**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔁 O ciclo de LLMOps fecha aqui
# MAGIC 1. Usuários conversam no App → **traces + feedback** no MLflow.
# MAGIC 2. Respostas com 👎 viram **novos exemplos** no dataset de avaliação (notebook 01).
# MAGIC 3. Ajustamos prompt/modelo/ferramentas e **avaliamos de novo** (notebook 03).
# MAGIC 4. Se melhorou, fazemos **deploy** da nova versão (este notebook).
# MAGIC 5. Acompanhamos **custos e tokens** (notebook 05), pelo MLflow e pelo `system.ai_gateway.usage` do Unity Gateway.
