# Databricks notebook source
# MAGIC %md
# MAGIC # 💰 05 - Análise de custos e consumo de tokens
# MAGIC
# MAGIC ## Como um LLM é cobrado?
# MAGIC Nosso agente chama o GPT-OSS-20B pelo **Unity Gateway**, no modelo **pay-per-token**: paga-se por **token** — um pedaço de palavra (~4 caracteres em inglês,
# MAGIC um pouco menos em português). Há dois preços:
# MAGIC - **Tokens de entrada** (*input*): tudo o que enviamos — prompt de sistema, ferramentas, histórico, resultados das ferramentas.
# MAGIC - **Tokens de saída** (*output*): o que o modelo gera. Costuma ser **3-4× mais caro** que a entrada.
# MAGIC
# MAGIC > 🆓 No **Free Edition** não há cobrança (o uso tem cotas). Aqui calculamos **quanto custaria** em um workspace pago —
# MAGIC > exatamente a análise que uma empresa faz antes de colocar um agente em produção.
# MAGIC
# MAGIC ## Fontes de dados
# MAGIC | Fonte | Visão | O que tem |
# MAGIC |---|---|---|
# MAGIC | **MLflow Traces** | Do **agente** | Tokens por interação, nº de chamadas ao LLM, ferramentas, latência, origem (`notebook`, `avaliacao`, `app`) |
# MAGIC | **`system.ai_gateway.usage`** | Do **Unity Gateway** | 1 linha por chamada ao model service: quem chamou, tokens, latência, status, request tags |
# MAGIC | **`system.billing.usage`** | Da **fatura** | Consumo faturado (DBUs), com atraso de algumas horas |
# MAGIC
# MAGIC Os traces explicam **por que** o agente gastou; o Unity Gateway mostra **quem** gastou, inclusive chamadas feitas fora do agente.

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.4" openai databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Preços (⚠️ ilustrativos — confira antes de apresentar)
# MAGIC Valores em **US$ por 1 milhão de tokens**. Consulte a tabela oficial em
# MAGIC [databricks.com/product/pricing/foundation-model-serving](https://www.databricks.com/product/pricing/foundation-model-serving)
# MAGIC e atualize o dicionário abaixo.

# COMMAND ----------

PRECOS_USD_POR_MILHAO = {
    # modelo de destino: {"input": US$/1M tokens de entrada, "output": US$/1M tokens de saída}
    "gpt-oss-20b": {"input": 0.07, "output": 0.30},
    "gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  # só para comparação
}

# Para qual modelo cada model service do Unity Gateway encaminha (ver 00_setup_unity_gateway)
DESTINO_POR_MODEL_SERVICE = {MODEL_SERVICE: DESTINATION_MODEL.split(".")[-1]}
COTACAO_USD_BRL = 5.40  # ajuste para a cotação do dia

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. (Opcional) Gerando um pouco de tráfego
# MAGIC Se o App ainda não foi usado, rode esta célula para simular algumas conversas.

# COMMAND ----------

from agente import AgenteEJ

agente = AgenteEJ()
perguntas_simuladas = [
    "Quais serviços vocês oferecem?",
    "Quanto custa um projeto elétrico de 45 horas?",
    "Vocês fazem obras?",
]
for p in perguntas_simuladas:
    agente.responder_pergunta(p, origem="simulacao")
print("Tráfego simulado gerado ✅")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extraindo tokens dos traces do MLflow

# COMMAND ----------

import json

import mlflow
import pandas as pd
from mlflow.entities import SpanType, Trace

traces = mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], max_results=2000)
print(f"{len(traces)} traces encontrados")


def _trace(valor) -> Trace:
    return Trace.from_json(valor) if isinstance(valor, str) else valor


def _data(valor):
    return pd.to_datetime(valor, unit="ms") if isinstance(valor, (int, float)) else pd.to_datetime(valor)


def tokens_do_trace(linha) -> tuple[int, int]:
    """Lê o total de tokens do trace; se não houver, soma os spans de LLM."""
    uso = (linha.get("trace_metadata") or {}).get("mlflow.trace.tokenUsage")
    if uso:
        uso = json.loads(uso) if isinstance(uso, str) else uso
        return uso.get("input_tokens", 0), uso.get("output_tokens", 0)
    entrada = saida = 0
    for span in _trace(linha["trace"]).data.spans:
        u = span.attributes.get("mlflow.chat.tokenUsage") or {}
        entrada += u.get("input_tokens", 0)
        saida += u.get("output_tokens", 0)
    return entrada, saida


def chamadas_llm(trace) -> int:
    return len(trace.search_spans(span_type=SpanType.CHAT_MODEL)) or len(trace.search_spans(span_type=SpanType.LLM))


linhas = []
for _, t in traces.iterrows():
    tags = t.get("tags") or {}
    if "origem" not in tags:  # ignora traces que não são do agente (ex.: chamadas dos juízes)
        continue
    if str(t.get("state", "OK")).upper().split(".")[-1] != "OK":  # ignora interações que terminaram com erro
        continue
    entrada, saida = tokens_do_trace(t)
    linhas.append({
        "trace_id": t["trace_id"],
        "data": _data(t["request_time"]),
        "origem": tags["origem"],
        "model_service": tags.get("model_service", MODEL_SERVICE),
        "tokens_entrada": entrada,
        "tokens_saida": saida,
        "chamadas_llm": chamadas_llm(_trace(t["trace"])),
        "latencia_s": (t["execution_duration"] or 0) / 1000,
    })

df = pd.DataFrame(linhas)
assert len(df) > 0, "Nenhum trace do agente encontrado. Rode a célula de tráfego simulado (seção 1)."


def custo_usd(linha) -> float:
    modelo = DESTINO_POR_MODEL_SERVICE.get(linha["model_service"], linha["model_service"])
    preco = PRECOS_USD_POR_MILHAO.get(modelo, {"input": 0, "output": 0})
    return (linha["tokens_entrada"] * preco["input"] + linha["tokens_saida"] * preco["output"]) / 1_000_000


df["tokens_total"] = df["tokens_entrada"] + df["tokens_saida"]
df["custo_usd"] = df.apply(custo_usd, axis=1)
df["custo_brl"] = df["custo_usd"] * COTACAO_USD_BRL
display(df.sort_values("data", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Resumo por origem
# MAGIC Quem mais consome: o desenvolvimento (notebook), a **avaliação** ou os usuários reais (**app**)?

# COMMAND ----------

resumo = df.groupby("origem").agg(
    interacoes=("trace_id", "count"),
    tokens_entrada=("tokens_entrada", "sum"),
    tokens_saida=("tokens_saida", "sum"),
    media_tokens_por_interacao=("tokens_total", "mean"),
    media_chamadas_llm=("chamadas_llm", "mean"),
    latencia_media_s=("latencia_s", "mean"),
    custo_total_brl=("custo_brl", "sum"),
).round(4).reset_index()
display(resumo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entrada × saída: para onde vão os tokens?

# COMMAND ----------

import matplotlib.pyplot as plt

fig, eixos = plt.subplots(1, 2, figsize=(13, 4.5))

totais = df[["tokens_entrada", "tokens_saida"]].sum()
eixos[0].bar(["Entrada", "Saída"], totais.values, color=["#4C78A8", "#F58518"])
eixos[0].set_title("Total de tokens: entrada × saída")
eixos[0].set_ylabel("tokens")

for n, grupo in df.groupby("chamadas_llm"):
    eixos[1].scatter(grupo["tokens_entrada"], grupo["tokens_saida"], label=f"{n} chamada(s) ao LLM", alpha=0.7)
eixos[1].set_title("Cada ponto é uma interação")
eixos[1].set_xlabel("tokens de entrada")
eixos[1].set_ylabel("tokens de saída")
eixos[1].legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 O que observar
# MAGIC - **A entrada domina.** O prompt de sistema e a descrição das ferramentas vão em **toda** chamada.
# MAGIC - **Cada uso de ferramenta = mais uma chamada ao LLM**, reenviando todo o histórico. Interações com 2-3 chamadas
# MAGIC   gastam 2-3× mais tokens de entrada.
# MAGIC - **Conversas longas crescem**: o App limita o histórico a 10 mensagens (`MAX_MENSAGENS_HISTORICO`) por isso.
# MAGIC - **Raciocínio custa:** no GPT-OSS-20B, boa parte dos tokens de **saída** é o "pensamento" do modelo, que o usuário nem vê.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quanto do custo é o prompt de sistema?

# COMMAND ----------

from agente import FERRAMENTAS_SCHEMA, SYSTEM_PROMPT

# Medimos de verdade: uma chamada com SÓ o prompt de sistema + as ferramentas + "oi".
# O número de tokens de entrada dessa chamada é o "custo fixo" que vai em TODA chamada do agente.
cliente = agente._cliente_gateway()
medicao = cliente.chat.completions.create(
    model=MODEL_SERVICE,
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "oi"}],
    tools=FERRAMENTAS_SCHEMA,
    max_tokens=50,
    extra_headers={"Databricks-Ai-Gateway-Request-Tags": json.dumps({"projeto": "llmops_workshop", "origem": "medicao"})},
)
tokens_fixos = medicao.usage.prompt_tokens
media_entrada_por_chamada = df["tokens_entrada"].sum() / df["chamadas_llm"].clip(lower=1).sum()

print(f"Tokens fixos por chamada (prompt de sistema + ferramentas): {tokens_fixos}")
print(f"Média real de tokens de entrada por chamada ao LLM:         {media_entrada_por_chamada:.0f}")
print(f"➡️ ~{100 * tokens_fixos / media_entrada_por_chamada:.0f}% da entrada é 'custo fixo' do prompt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Projeção: quanto custaria em produção?
# MAGIC Usamos a **média real** de tokens por interação para projetar cenários de uso mensal.

# COMMAND ----------

media_entrada = df["tokens_entrada"].mean()
media_saida = df["tokens_saida"].mean()

cenarios = [
    ("Piloto na EJ", 10, 5),
    ("Site da EJ", 100, 3),
    ("Universidade inteira", 2000, 2),
]
projecoes = []
for nome, usuarios_dia, perguntas_por_usuario in cenarios:
    interacoes_mes = usuarios_dia * perguntas_por_usuario * 30
    for endpoint, preco in PRECOS_USD_POR_MILHAO.items():
        custo_mes = interacoes_mes * (media_entrada * preco["input"] + media_saida * preco["output"]) / 1_000_000
        projecoes.append({
            "cenario": nome, "interacoes_mes": interacoes_mes, "modelo": endpoint,
            "custo_mes_usd": round(custo_mes, 2), "custo_mes_brl": round(custo_mes * COTACAO_USD_BRL, 2),
        })

display(pd.DataFrame(projecoes))

# COMMAND ----------

# MAGIC %md
# MAGIC > ⚠️ A projeção cobre só o **LLM do agente**. Em produção também entram:
# MAGIC > - **Compute do App** (cobrado por hora enquanto o App está ligado, independente do uso);
# MAGIC > - **Juízes de avaliação** (cada `mlflow.genai.evaluate` faz várias chamadas de LLM extras);
# MAGIC > - Se o agente estivesse em **Model Serving**, o compute do endpoint (ver README).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. A visão do Unity Gateway: `system.ai_gateway.usage`
# MAGIC Cada chamada ao model service gera uma linha, **independente de quem chamou** (agente, notebook de teste, outro sistema).
# MAGIC As *request tags* que o agente envia (`origem`) permitem separar o consumo por origem também aqui.
# MAGIC
# MAGIC > ⏱️ Esta tabela tem alguns minutos de atraso. Se estiver vazia, espere um pouco e rode de novo.

# COMMAND ----------

uso_gateway = spark.sql(f"""
    SELECT date(event_time)          AS dia,
           requester,
           request_tags['origem']    AS origem,
           count(*)                  AS chamadas_llm,
           sum(CASE WHEN status_code = 200 THEN 0 ELSE 1 END) AS erros,
           sum(input_tokens)         AS tokens_entrada,
           sum(output_tokens)        AS tokens_saida,
           percentile(latency_ms, 0.95) AS latencia_p95_ms
    FROM system.ai_gateway.usage
    WHERE endpoint_name = '{MODEL_SERVICE}'
      AND event_time >= current_timestamp() - INTERVAL 30 DAYS
    GROUP BY ALL
    ORDER BY dia DESC, chamadas_llm DESC
""")
display(uso_gateway)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Conferindo: MLflow × Unity Gateway
# MAGIC Os totais devem ser parecidos. Diferenças comuns:
# MAGIC - O gateway conta **todas** as chamadas (ex.: o teste do `00_setup_unity_gateway`, com `origem = setup`).
# MAGIC - O gateway conta **chamadas ao LLM**; o MLflow agrupa em **interações** (1 interação = 1 a 5 chamadas).
# MAGIC - O atraso de alguns minutos da tabela do gateway.

# COMMAND ----------

gw = uso_gateway.groupBy("origem").sum("chamadas_llm", "tokens_entrada", "tokens_saida").toPandas()
gw.columns = ["origem", "chamadas_llm_gateway", "tokens_entrada_gateway", "tokens_saida_gateway"]
gw["origem"] = gw["origem"].fillna("(sem tag)")  # chamadas ao gateway feitas sem a request tag "origem"
ml = df.groupby("origem").agg(chamadas_llm_mlflow=("chamadas_llm", "sum"),
                              tokens_entrada_mlflow=("tokens_entrada", "sum"),
                              tokens_saida_mlflow=("tokens_saida", "sum")).reset_index()
display(ml.merge(gw, on="origem", how="outer").fillna(0))

# COMMAND ----------

# MAGIC %md
# MAGIC > 💡 **Na empresa real**, o Unity Gateway também permite criar **budgets** (alerta e bloqueio de gastos) e
# MAGIC > **rate limits** por usuário no model service, além de um **dashboard de uso** pronto (AI Gateway → Usage dashboard).
# MAGIC > Alguns desses recursos são de administrador de conta e podem não existir no Free Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. (Opcional) Consumo faturado: `system.billing.usage`
# MAGIC As tabelas `system.billing.*` mostram o consumo faturado do workspace, em **DBUs**.
# MAGIC Podem não estar disponíveis/populadas no Free Edition — se der erro, tudo bem.

# COMMAND ----------

try:
    display(spark.sql(f"""
        SELECT u.usage_date,
               u.billing_origin_product AS produto,
               u.sku_name,
               SUM(u.usage_quantity)                    AS dbus,
               SUM(u.usage_quantity * p.pricing.default) AS custo_lista_usd
        FROM system.billing.usage u
        LEFT JOIN system.billing.list_prices p
          ON u.sku_name = p.sku_name
         AND u.usage_start_time >= p.price_start_time
         AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_date >= current_date() - INTERVAL 30 DAYS
          AND (u.usage_metadata.ai_gateway.endpoint_name = '{MODEL_SERVICE}'
               OR u.billing_origin_product = 'APPS')
        GROUP BY ALL
        ORDER BY u.usage_date DESC, dbus DESC
    """))
except Exception as erro:  # noqa: BLE001
    print(f"System tables indisponíveis neste workspace: {str(erro)[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Como reduzir custos? (discussão)
# MAGIC | Alavanca | Efeito | Como medir se a qualidade se manteve |
# MAGIC |---|---|---|
# MAGIC | Modelo menor/mais barato | 💰💰💰 | Trocar o destino do model service no Unity Gateway e rodar o notebook 03 |
# MAGIC | Prompt de sistema mais curto | 💰 em toda chamada | Notebook 03 (cuidado com `escopo_e_honestidade`!) |
# MAGIC | Limitar histórico da conversa | 💰 em conversas longas | Testar conversas multi-turno |
# MAGIC | `max_tokens` menor | Limita a saída (mais cara) | Juiz `resposta_concisa` |
# MAGIC | Juízes em código quando possível | 💰 na avaliação | — |
# MAGIC
# MAGIC > ✏️ **Exercício final:** escolha uma alavanca, aplique, rode o notebook 03 e este notebook de novo.
# MAGIC > O custo caiu? A qualidade se manteve? **Isso é LLMOps.** 🚀
