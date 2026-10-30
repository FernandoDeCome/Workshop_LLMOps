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
| 0 | `00_setup_unity_gateway.py` | Cria o model service no Unity Gateway, guarda o seu token (PAT) num secret scope e testa a primeira chamada ao LLM |
| — | `00_config.py` | Configuração compartilhada (catálogo, model service, token, experimento). Chamado via `%run` |
| 1 | `01_dataset_avaliacao.py` | Dataset de avaliação (10 exemplos, 4 categorias) como MLflow Evaluation Dataset no UC |
| 2 | `02_agente.py` | O agente: tool calling, prompt de sistema, tracing, tokens |
| 3 | `03_avaliacao.py` | `mlflow.genai.evaluate` com 2 juízes nativos + 2 customizados (+1 em código); comparação v1 × v2 |
| 4 | `04_app.py` + `app/` | Cria e publica, por código, o seu Databricks App (backend FastAPI + interface HTML/CSS) com o agente embutido |
| 5 | `05_custos_tokens.py` | Tokens por origem (MLflow), visão do Unity Gateway (`system.ai_gateway.usage`), entrada × saída, peso do prompt, projeção de custo |

## Antes de começar

1. **Crie o seu workspace Free Edition** em [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition)
   (é gratuito e leva 1 minuto).
2. **Traga este repositório para o seu workspace:** menu lateral **Workspace → Create → Git folder**, cole a URL
   `https://github.com/FernandoDeCome/Workshop_LLMOps.git` e clique em **Create Git folder**.
   Os arquivos `.py` aparecem como notebooks e a pasta `app/` fica com o código do App.
3. Abra os notebooks **na ordem** (00 → 05) e rode as células de cima para baixo. Os notebooks usam computação
   *serverless*: não precisa criar cluster.

Tudo o que é específico do seu workspace (endereço, usuário, pastas) é descoberto automaticamente pelo código.
Você não precisa trocar nenhuma variável.

## Agenda (2h)

O conteúdo ocupa ~95 min e deixa **25 min para dúvidas**.

| Horário | Bloco | O que você faz |
|---|---|---|
| 0:00 – 0:10 | Abertura: o que é LLMOps e a arquitetura do workshop (diagrama acima) | Cria o Git folder enquanto acompanha |
| 0:10 – 0:20 | `00_setup_unity_gateway` | Cria o model service, gera e guarda seu token, testa a primeira chamada ao LLM |
| 0:20 – 0:30 | `01_dataset_avaliacao` | Cria o dataset de avaliação e discute as categorias de exemplos |
| 0:30 – 0:50 | `02_agente` | Conversa com o agente, abre um trace e olha os tokens |
| 0:50 – 1:10 | `03_avaliacao` | Avalia o agente com os juízes e compara o prompt completo com o simplificado |
| 1:10 – 1:20 | `04_app` | Publica o seu App, conversa com ele, dá 👍/👎 e encontra os traces (`origem = app`) |
| 1:20 – 1:35 | `05_custos_tokens` | Analisa os tokens e discute como reduzir custos |
| 1:35 – 2:00 | Dúvidas e perguntas | |

> 💡 **Dica de tempo:** o primeiro deploy do App leva de 3 a 5 minutos. Se quiser, rode a célula do `04_app`
> logo depois de terminar o `02_agente` e siga para o `03_avaliacao` enquanto o App sobe.

**Se o tempo apertar, pule:** seção 4 do `01` (tabela de categorias), seção 4 do `02` (conversa com várias mensagens),
seções 1, 5 e 8 do `05`. Os exercícios ✏️ ficam como lição de casa.

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

**Por quê:** o agente não guarda estado (o histórico fica no navegador, em `static/app.js`), tem um único consumidor
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

## (Opcional) Usando a CLI em vez do Git folder

Se preferir trabalhar no seu computador, dá para subir os arquivos com a [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install):

```bash
databricks auth login --host https://<seu-workspace>.cloud.databricks.com --profile free-ej
databricks workspace import-dir ./Workshop_LLMOps /Users/<seu-email>/workshop_llmops_ej --profile free-ej
```

Os `.py` com cabeçalho `# Databricks notebook source` viram notebooks; os arquivos de `app/` continuam arquivos.

> ⚠️ **Free Edition:** o App para automaticamente após 24h. Para ligar de novo, rode a célula do `04_app`
> ou clique em **Start** em **Compute → Apps**.

## Problemas comuns

| Sintoma | Causa provável / solução |
|---|---|
| `ModuleNotFoundError: agente` | O notebook não está na mesma pasta que `app/`, ou `%run ./00_config` não foi executado |
| `404` / "does not exist" ao chamar o LLM | Falta `EXECUTE` no model service, permissão ainda propagando (~1 min) ou nome errado |
| `401` / `403` nos notebooks | PAT ausente/expirado → refazer o passo 3 do `00_setup_unity_gateway` (e rodar `%run ./00_config` de novo) |
| LLM responde sem usar ferramentas | Modelo com *tool calling* fraco → testar outro modelo e comparar no notebook 03 |
| App não registra traces ou dá `404`/`403` ao chamar o LLM | Recursos do App fora do esperado → rode de novo a célula do `04_app`, que recria os recursos `model` e `experiment` |
| Deploy do App falhou | Veja os logs em **Compute → Apps → assistente-ej → Logs** |
| Erro de limite ao criar o App | O Free Edition permite até 3 Apps por conta → apague um App que não usa em **Compute → Apps** |
| `429` / `REQUEST_LIMIT_EXCEEDED` | Limite de chamadas por segundo do Free Edition → espere alguns segundos e rode a célula de novo |
| `system.ai_gateway.usage` vazia | Atraso de alguns minutos; ou a tabela não está disponível no workspace |
| `create_dataset`/`make_judge` não existem | Versão antiga do MLflow → rodar a célula `%pip install -U "mlflow[databricks]>=3.4"` |
