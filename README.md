# Workshop LLMOps no Databricks — Assistente Engenhar Jr.

Workshop prático para Empresa Júnior de Engenharia, usando **Databricks Free Edition**.
Cada participante usa o **próprio workspace Free Edition** ([crie o seu aqui](https://www.databricks.com/learn/free-edition)).

**Caso de uso:** um assistente virtual da EJ fictícia *Engenhar Jr.* que responde dúvidas sobre serviços e
calcula orçamentos usando 2 ferramentas (busca na base de conhecimento + cálculo de orçamento).

**LLM:** GPT-OSS-20B, acessado **exclusivamente pelo Unity Gateway** através do model service
`workspace.llmops_workshop.assistente_ej_llm` (endpoint OpenAI-compatível `https://<workspace>/ai-gateway/mlflow/v1`).
Nenhum código chama o modelo diretamente.

```
Notebooks (PAT do aluno) ─┐
                          ├─► Unity Gateway ─► model service (UC, EXECUTE) ─► system.ai.gpt-oss-20b
App (OAuth do App)  ──────┘        └─► system.ai_gateway.usage (quem, tokens, latência, request tags)
```

## Estrutura

| Ordem | Arquivo | Conteúdo |
|---|---|---|
| 0 | `00_setup_unity_gateway.py` | Cria o model service, dá `EXECUTE` aos alunos, guarda o PAT de cada aluno num secret scope e testa a chamada |
| — | `00_config.py` | Configuração compartilhada (catálogo, model service, token, experimento). Chamado via `%run` |
| 1 | `01_dataset_avaliacao.py` | Dataset de avaliação (10 exemplos, 4 categorias) como MLflow Evaluation Dataset no UC |
| 2 | `02_agente.py` | O agente: tool calling, prompt de sistema, tracing, tokens |
| 3 | `03_avaliacao.py` | `mlflow.genai.evaluate` com 2 juízes nativos + 2 customizados (+1 em código); comparação v1 × v2 |
| 4 | `04_app.py` + `app/` | Databricks App (backend FastAPI + interface HTML/CSS) com o agente embutido, traces e feedback 👍/👎 |
| 5 | `05_custos_tokens.py` | Tokens por origem (MLflow), visão do Unity Gateway (`system.ai_gateway.usage`), entrada × saída, peso do prompt, projeção de custo |

## Agenda (2h)

O conteúdo ocupa ~95 min e deixa **25 min para dúvidas**. Para caber, o instrutor faz antes da aula tudo o que é
demorado ou só precisa acontecer uma vez (ver "Preparação"): model service, permissões, deploy do App e uma rodada
completa dos notebooks.

| Horário | Bloco | O que os alunos fazem |
|---|---|---|
| 0:00 – 0:10 | Abertura: o que é LLMOps, arquitetura do workshop (diagrama acima) | Assistem |
| 0:10 – 0:15 | `00_setup_unity_gateway`, passos 3 e 4 | Criam o PAT e testam a chamada ao Unity Gateway |
| 0:15 – 0:25 | `01_dataset_avaliacao` | Rodam e discutem as categorias de exemplos |
| 0:25 – 0:45 | `02_agente` | Rodam as seções 1–3 e 5; abrem um trace e olham os tokens |
| 0:45 – 1:10 | `03_avaliacao` | Rodam a v1 e analisam os juízes na interface. A v2 é **demonstrada pelo instrutor** (já rodada antes) |
| 1:10 – 1:20 | `04_app` | Usam o App já publicado, dão 👍/👎 e encontram seus traces (`origem = app`) |
| 1:20 – 1:35 | `05_custos_tokens` | Rodam as seções 2–7 e discutem as alavancas de custo |
| 1:35 – 2:00 | Dúvidas e perguntas | |

**Se o tempo apertar, pule:** seção 4 do `01` (tabela de categorias), seção 4 do `02` (conversa com várias mensagens),
seção 4 do `03` (só mostre a comparação já pronta), seções 1, 5 e 8 do `05`. Os exercícios ✏️ ficam como lição de casa.

```
app/
├── agente.py               ← ÚNICA fonte do agente (usada pelos notebooks e pelo App)
├── base_conhecimento.json  ← dados da EJ (serviços, preços, FAQ)
├── app.py                  ← backend FastAPI (rotas /api/chat, /api/feedback, /api/info)
├── static/                 ← interface em HTML + CSS + um pouco de JavaScript (sem build)
├── app.yaml                ← comando + recursos do App
└── requirements.txt
```

## Decisão de arquitetura: Model Serving × agente dentro do App

### Recomendação para este workshop: **agente dentro do App**, com o código num módulo separado (`agente.py`)

| Critério | Agente no App ✅ | Agente em Model Serving |
|---|---|---|
| Passos de deploy | 1 (App) | 3 (log no MLflow → registrar no UC → `agents.deploy` / endpoint) + App |
| Tempo até funcionar | ~2-5 min | ~15-20 min para o endpoint subir, mais *cold start* com scale-to-zero |
| Limites do Free Edition | Até 3 Apps por conta | Número de endpoints limitado; App + endpoint consomem cota |
| Depuração para iniciantes | Código e erros num só lugar (logs do App) | Erros espalhados entre App, endpoint e empacotamento do modelo (dependências, `models-from-code`) |
| Custo de infraestrutura | Só o compute do App | Compute do App **+** do endpoint |
| Avaliação e tracing no MLflow | ✅ Mesmos traces, mesmo experimento | ✅ (+ inference tables) |
| Governança do LLM pelo Unity Gateway | ✅ O App chama o model service com o próprio service principal | ✅ O endpoint chama o model service |
| Reuso por outros sistemas (API REST) | ❌ Só pelo App | ✅ Qualquer cliente chama o endpoint |
| Versionamento / rollback | Pelo deploy do App (código) | Versões no UC Model Registry, rollback por versão |
| Escalar agente e interface separadamente | ❌ | ✅ |
| Governança da chamada **ao agente** (quem pode usar o agente em si) | Permissão de acesso ao App | Permissão no endpoint + inference tables |

**Por quê:** o agente não guarda estado (o histórico fica no `st.session_state` do App), tem um único consumidor
(o App) e o objetivo é didático. Um endpoint de Model Serving traria passos extras de empacotamento,
espera de provisionamento e mais uma peça para dar erro, sem trazer benefício que os alunos percebam.
O que importa para LLMOps continua lá: **o mesmo código é avaliado e publicado**, e todos os traces
(notebook, avaliação, App) vão para o **mesmo experimento** com a tag `origem`.

**Quando migrar para Model Serving:** quando houver **mais de um consumidor** (site, WhatsApp, outro sistema),
quando for preciso **versionar e fazer rollback** independente da interface, fazer A/B entre versões ou ter
governança completa (inference tables, rate limit por usuário). Como o agente já é um módulo isolado,
a migração é envolvê-lo em um `mlflow.pyfunc.ResponsesAgent`, registrar no UC e chamar `databricks.agents.deploy`.
O App passa a chamar o endpoint em vez de importar `agente.py`. Isso pode ser um bom "próximo passo" para
quem quiser continuar depois do workshop.

## Preparação (antes do workshop)

1. **Conferir o Unity Gateway e o modelo:** no workspace, confirme que o menu **AI Gateway** existe e que o
   GPT-OSS-20B aparece como destino de model service (`models/system.ai.gpt-oss-20b`, em `00_config.py`).
2. **Subir os arquivos para o workspace** (opção CLI):
   ```bash
   databricks auth login --host https://<seu-workspace>.cloud.databricks.com --profile free-ej
   databricks workspace import-dir ./workshop_llmops_ej /Users/<seu-email>/workshop_llmops_ej --profile free-ej
   ```
   Os `.py` com cabeçalho `# Databricks notebook source` viram notebooks; os arquivos de `app/` continuam arquivos.
   Alternativa para os alunos: **Workspace → Create → Git folder** com a URL
   `https://github.com/FernandoDeCome/Workshop_LLMOps.git` (não precisa de CLI).
3. **Rodar `00_setup_unity_gateway`** (passos 1 e 2 como instrutor; passo 3 e 4 com seu próprio PAT).
   No dia, cada aluno roda só os passos 3 e 4.
4. **Rodar 01 → 05 uma vez** para validar (e popular traces para a aula de custos).
5. **Criar e publicar o App** (`04_app.py`, passos 1-3) com as *resource keys* `model` (Unity Gateway model service)
   e `experiment`. Dê permissão **Can use** no App para os alunos e anote a URL para compartilhar.
6. **Rodar a v2 do `03_avaliacao`** com antecedência, para só mostrar a comparação durante a aula.
7. ⚠️ **No dia:** Apps do Free Edition param após 24h. Clique em **Start** no App antes de começar.

## Problemas comuns

| Sintoma | Causa provável / solução |
|---|---|
| `ModuleNotFoundError: agente` | O notebook não está na mesma pasta que `app/`, ou `%run ./00_config` não foi executado |
| `404` / "does not exist" ao chamar o LLM | Falta `EXECUTE` no model service, permissão ainda propagando (~1 min) ou nome errado |
| `401` / `403` nos notebooks | PAT ausente/expirado → refazer o passo 3 do `00_setup_unity_gateway` (e rodar `%run ./00_config` de novo) |
| LLM responde sem usar ferramentas | Modelo com *tool calling* fraco → testar outro modelo e comparar no notebook 03 |
| App não registra traces | Recurso `experiment` ausente/permissão diferente de *Can edit*, ou *resource key* diferente do `app.yaml` |
| App: `404`/`403` ao chamar o LLM | Recurso `model` (Unity Gateway model service) ausente ou *resource key* diferente do `app.yaml` |
| `system.ai_gateway.usage` vazia | Atraso de alguns minutos; ou a tabela não está disponível no workspace |
| `create_dataset`/`make_judge` não existem | Versão antiga do MLflow → rodar a célula `%pip install -U "mlflow[databricks]>=3.4"` |
