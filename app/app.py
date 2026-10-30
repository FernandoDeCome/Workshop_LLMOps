"""
Databricks App: backend em Python (FastAPI) do Assistente Engenhar Jr.

- A interface é HTML + CSS + um pouco de JavaScript, na pasta static/ (sem build, sem TypeScript).
- O agente roda DENTRO do App (importado de agente.py) -> não precisamos de um Model Serving endpoint.
- O LLM é chamado pelo Unity Gateway com o token OAuth do service principal do App (sem PAT).
- Cada conversa gera um trace no MLflow (mesmo experimento dos notebooks) com a tag origem=app.
- O usuário pode dar 👍/👎, que vira um "feedback humano" no trace (ótimo para melhorar o dataset de avaliação).

Rotas:
  GET  /               -> página do chat (static/index.html)
  GET  /api/info       -> dados da empresa para a barra lateral
  POST /api/chat       -> envia o histórico da conversa e recebe a resposta do agente
  POST /api/feedback   -> registra 👍/👎 no trace do MLflow
"""

import os
from pathlib import Path

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.entities import AssessmentSource, AssessmentSourceType
from pydantic import BaseModel

from agente import BASE, AgenteEJ

MAX_MENSAGENS_HISTORICO = 10  # limita o histórico enviado ao LLM -> controla o custo em tokens
STATIC = Path(__file__).parent / "static"

mlflow.set_tracking_uri("databricks")
if os.getenv("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])

agente = AgenteEJ()
app = FastAPI(title="Assistente Engenhar Jr.")


class Mensagem(BaseModel):
    role: str      # "user" ou "assistant"
    content: str


class PedidoChat(BaseModel):
    mensagens: list[Mensagem]


class PedidoFeedback(BaseModel):
    trace_id: str
    positivo: bool


def usuario_logado(request: Request) -> str:
    # O Databricks Apps informa quem está logado neste cabeçalho
    return request.headers.get("X-Forwarded-Email", "anonimo")


@app.get("/api/info")
def info():
    return {
        "empresa": BASE["empresa"],
        "servicos": [{"nome": s["nome"], "valor_hora": s["valor_hora"]} for s in BASE["servicos"]],
        "modelo": agente.model_service,
    }


@app.post("/api/chat")
def chat(pedido: PedidoChat):
    historico = [m.model_dump() for m in pedido.mensagens[-MAX_MENSAGENS_HISTORICO:]]
    resultado = agente.responder(historico, origem="app")
    return {
        "resposta": resultado["resposta"],
        "uso_tokens": resultado["uso_tokens"],
        "ferramentas_usadas": resultado["ferramentas_usadas"],
        "trace_id": resultado["trace_id"],
    }


@app.post("/api/feedback")
def feedback(pedido: PedidoFeedback, request: Request):
    mlflow.log_feedback(
        trace_id=pedido.trace_id,
        name="feedback_usuario",
        value=pedido.positivo,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=usuario_logado(request)),
    )
    return {"ok": True}


@app.get("/")
def pagina_inicial():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
