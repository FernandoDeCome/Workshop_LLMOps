# Databricks notebook source
# MAGIC %md
# MAGIC # 🔐 00 - Setup do Unity Gateway (rodar uma vez, antes do workshop)
# MAGIC
# MAGIC ## Por que Unity Gateway?
# MAGIC Em vez de cada aplicação chamar o modelo "direto", todas as chamadas passam por um **portão único e governado**:
# MAGIC
# MAGIC ```
# MAGIC  Notebook / App ──► Unity Gateway ──► model service (UC) ──► GPT-OSS-20B
# MAGIC                       │  workspace.llmops_workshop.assistente_ej_llm
# MAGIC                       ├─ quem pode chamar?   → permissão EXECUTE no Unity Catalog
# MAGIC                       ├─ quanto foi usado?   → system.ai_gateway.usage (tokens, latência, quem chamou)
# MAGIC                       └─ quanto pode gastar? → budgets e rate limits
# MAGIC ```
# MAGIC
# MAGIC Um **model service** é um objeto do Unity Catalog (`catalogo.schema.nome`) que aponta para um modelo.
# MAGIC Se amanhã quisermos trocar o GPT-OSS-20B por outro modelo, **mudamos só o model service**: o código do agente continua igual.
# MAGIC
# MAGIC | Passo | Quem roda | Quando |
# MAGIC |---|---|---|
# MAGIC | 1. Criar o model service | Instrutor | Uma vez |
# MAGIC | 2. Dar permissão aos alunos | Instrutor | Uma vez |
# MAGIC | 3. Guardar o token pessoal | **Cada aluno** | Uma vez por pessoa |
# MAGIC | 4. Testar a chamada | Cada aluno | Sempre que quiser |

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.4" openai databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from openai import OpenAI

w = WorkspaceClient()
HOST = w.config.host.rstrip("/")
GATEWAY_URL = f"{HOST}/ai-gateway/mlflow/v1"   # endpoint OpenAI-compatível do Unity Gateway
print("Unity Gateway:", GATEWAY_URL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 · Criar o model service (instrutor)
# MAGIC **Pela interface:** menu lateral **AI Gateway → Create → Model service** → catálogo `workspace`, schema `llmops_workshop`,
# MAGIC nome `assistente_ej_llm` → destino **GPT OSS 20B** → **Create**.
# MAGIC
# MAGIC **Ou por código** (célula abaixo, faz a mesma coisa e pode ser rodada de novo sem problema):

# COMMAND ----------

from databricks.sdk.service.catalog import ModelService

try:
    servico = w.ai_gateway.get_model_service(f"model-services/{MODEL_SERVICE}")
    print(f"• model service {MODEL_SERVICE} já existe")
except NotFound:
    servico = w.ai_gateway.create_model_service(
        parent=f"schemas/{CATALOG}.{SCHEMA}",
        model_service_id=MODEL_SERVICE_NAME,
        model_service=ModelService.from_dict({
            "comment": "Workshop LLMOps: LLM do Assistente Engenhar Jr.",
            "config": {"routing": {"destinations": [{
                "name": "principal",
                "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                "pay_per_token_config": {"model": DESTINATION_MODEL},
                "traffic_percentage": 100,
            }]}},
        }),
    )
    print(f"✅ model service criado: {MODEL_SERVICE}")

print(json.dumps(servico.as_dict().get("config", {}), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 · Permissões (instrutor)
# MAGIC Quem não tem `EXECUTE` no model service **não consegue chamar o modelo** (recebe 404, como se ele não existisse).
# MAGIC Aqui liberamos para todos os usuários do workspace. Em uma empresa, daríamos só para o grupo do projeto.
# MAGIC
# MAGIC > ⏱️ Uma permissão nova leva ~1 minuto para valer no gateway.

# COMMAND ----------

GRUPO = "account users"   # todos os usuários; troque por um grupo específico se preferir

for comando in [
    f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{GRUPO}`",
    f"GRANT USE SCHEMA, SELECT, MODIFY, CREATE TABLE ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{GRUPO}`",
    f"GRANT EXECUTE ON MODEL SERVICE {MODEL_SERVICE} TO `{GRUPO}`",
]:
    try:
        spark.sql(comando)
        print("✅", comando)
    except Exception as erro:
        print("⚠️", comando, "->", str(erro)[:200])

display(spark.sql(f"SHOW GRANTS ON MODEL SERVICE {MODEL_SERVICE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 3 · Guardar seu token pessoal (cada aluno)
# MAGIC O Unity Gateway precisa de um token **seu** para saber quem está chamando. Nos notebooks usamos um
# MAGIC **Personal Access Token (PAT)**; no App, o token é automático (é o do próprio App).
# MAGIC
# MAGIC 1. Clique no seu usuário (canto superior direito) → **Settings → Developer → Access tokens → Generate new token**.
# MAGIC    Comentário: `llmops-workshop`, validade: 30 dias. **Copie o token.**
# MAGIC 2. Rode a célula abaixo: vai aparecer um campo **`pat`** no topo do notebook. Cole o token lá.
# MAGIC 3. Rode a célula seguinte: ela guarda o token em um **secret scope** (cofre de segredos) e apaga o campo.
# MAGIC
# MAGIC > 🔒 Nunca escreva tokens direto no código! Um secret scope guarda o valor de forma segura e ele não aparece nos outputs.

# COMMAND ----------

dbutils.widgets.text("pat", "", "Cole aqui seu token (PAT)")

# COMMAND ----------

pat = dbutils.widgets.get("pat").strip()

if not pat and os.environ.get("DATABRICKS_UG_TOKEN"):
    print(f"✅ Já existe um token guardado em {SECRET_SCOPE}/{SECRET_KEY}. Nada a fazer.")
else:
    assert pat, "Cole o token no campo 'pat' no topo do notebook e rode esta célula de novo."
    if SECRET_SCOPE not in [s.name for s in w.secrets.list_scopes()]:
        w.secrets.create_scope(scope=SECRET_SCOPE)
    w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY, string_value=pat)
    os.environ["DATABRICKS_UG_TOKEN"] = pat
    print(f"✅ Token guardado em {SECRET_SCOPE}/{SECRET_KEY}")

dbutils.widgets.remove("pat")
del pat

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 4 · Testar a chamada pelo Unity Gateway
# MAGIC Qualquer cliente compatível com OpenAI funciona: `base_url` = Unity Gateway e `model` = **nome do model service**.

# COMMAND ----------

cliente = OpenAI(base_url=GATEWAY_URL, api_key=os.environ["DATABRICKS_UG_TOKEN"])

resposta = cliente.chat.completions.create(
    model=MODEL_SERVICE,
    messages=[{"role": "user", "content": "Diga 'olá, Engenhar Jr.!' em uma frase curta."}],
    max_tokens=300,
    extra_headers={"Databricks-Ai-Gateway-Request-Tags": json.dumps({"projeto": "llmops_workshop", "origem": "setup"})},
)

conteudo = resposta.choices[0].message.content
if isinstance(conteudo, list):  # GPT-OSS devolve "raciocínio" + "texto" em partes separadas
    conteudo = "".join(p.get("text", "") for p in conteudo if isinstance(p, dict) and p.get("type") == "text")
print(conteudo)
print("\nTokens:", resposta.usage.total_tokens)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Se der erro
# MAGIC | Erro | Causa provável |
# MAGIC |---|---|
# MAGIC | `404` / "does not exist" | Falta `EXECUTE` no model service (passo 2), a permissão ainda não propagou (~1 min) ou o nome está errado |
# MAGIC | `401` / `403` | Token inválido ou expirado → refaça o passo 3 |
# MAGIC | Erro de destino / modelo | GPT-OSS-20B indisponível no workspace → confira em **Serving → Foundation models** |
# MAGIC
# MAGIC ✅ Tudo certo? Siga para **01_dataset_avaliacao**.
