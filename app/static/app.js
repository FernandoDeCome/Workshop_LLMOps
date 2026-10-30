// Lógica da página: guarda o histórico da conversa no navegador e conversa com o backend Python.
const historico = [];   // [{role, content}] — o agente não guarda estado, quem guarda é a interface
let tokensSessao = 0;

const elMensagens = document.getElementById("mensagens");
const elFormulario = document.getElementById("formulario");
const elPergunta = document.getElementById("pergunta");
const elBotao = elFormulario.querySelector("button");

// Converte **negrito** e escapa HTML (as respostas do agente vêm em markdown simples)
function formatar(texto) {
  const seguro = texto.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return seguro.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function adicionarMensagem(role, texto, classeExtra = "") {
  const div = document.createElement("div");
  div.className = `mensagem ${role} ${classeExtra}`;
  div.innerHTML = formatar(texto);
  elMensagens.appendChild(div);
  elMensagens.scrollTop = elMensagens.scrollHeight;
  return div;
}

function adicionarDetalhes(div, resultado) {
  const u = resultado.uso_tokens;
  const ferramentas = resultado.ferramentas_usadas.length
    ? ` · ferramentas: ${resultado.ferramentas_usadas.join(", ")}` : "";
  const detalhes = document.createElement("div");
  detalhes.className = "detalhes";
  detalhes.innerHTML = `<span>${u.input_tokens} tokens de entrada · ${u.output_tokens} de saída${ferramentas}</span>
    <span class="espaco"></span>
    <button class="botao-feedback" data-positivo="true" title="Resposta boa">👍</button>
    <button class="botao-feedback" data-positivo="false" title="Resposta ruim">👎</button>`;
  detalhes.querySelectorAll(".botao-feedback").forEach((botao) => {
    botao.addEventListener("click", () => enviarFeedback(resultado.trace_id, botao, detalhes));
  });
  div.appendChild(detalhes);
}

async function enviarFeedback(traceId, botao, detalhes) {
  detalhes.querySelectorAll(".botao-feedback").forEach((b) => (b.disabled = true));
  botao.classList.add("escolhido");
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: traceId, positivo: botao.dataset.positivo === "true" }),
  });
}

elFormulario.addEventListener("submit", async (evento) => {
  evento.preventDefault();
  const pergunta = elPergunta.value.trim();
  if (!pergunta) return;

  historico.push({ role: "user", content: pergunta });
  adicionarMensagem("user", pergunta);
  elPergunta.value = "";
  elBotao.disabled = true;
  const pensando = adicionarMensagem("assistant", "Pensando...", "pensando");

  try {
    const resposta = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mensagens: historico }),
    });
    if (!resposta.ok) throw new Error(`erro ${resposta.status}`);
    const resultado = await resposta.json();

    pensando.remove();
    historico.push({ role: "assistant", content: resultado.resposta });
    adicionarDetalhes(adicionarMensagem("assistant", resultado.resposta), resultado);
    tokensSessao += resultado.uso_tokens.total_tokens;
    document.getElementById("tokens-sessao").textContent = tokensSessao.toLocaleString("pt-BR");
  } catch (erro) {
    pensando.remove();
    historico.pop();
    adicionarMensagem("assistant", `Ops, algo deu errado (${erro.message}). Tente de novo.`, "erro");
  } finally {
    elBotao.disabled = false;
    elPergunta.focus();
  }
});

document.getElementById("nova-conversa").addEventListener("click", () => {
  historico.length = 0;
  tokensSessao = 0;
  elMensagens.innerHTML = "";
  document.getElementById("tokens-sessao").textContent = "0";
});

// Preenche a barra lateral com os dados da empresa
fetch("/api/info").then((r) => r.json()).then((info) => {
  document.getElementById("nome-empresa").textContent = info.empresa.nome;
  document.getElementById("descricao").textContent = info.empresa.descricao;
  document.getElementById("modelo").textContent = info.modelo;
  document.getElementById("servicos").innerHTML = info.servicos
    .map((s) => `<li><strong>${s.nome}</strong>R$ ${s.valor_hora}/hora</li>`).join("");
});
