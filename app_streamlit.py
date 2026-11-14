import os
import uuid

import pandas as pd
import streamlit as st

try:
    from langchain_community.callbacks import StreamlitCallbackHandler
except ImportError:
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="BookBridge · Bia", page_icon="📚", layout="wide")
st.title("📚 BookBridge · atendimento com agente cognitivo")


def chave_configurada() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            return True
    except Exception:
        pass
    return False


if not chave_configurada():
    chave = st.sidebar.text_input("OPENAI_API_KEY", type="password", help="Usada apenas nesta sessão do navegador.")
    if not chave:
        st.info("Informe a OPENAI_API_KEY na barra lateral, ou defina a variável de ambiente antes de iniciar o app.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = chave

import bookbridge_agente as bb


@st.cache_resource(show_spinner="Carregando banco, índice FAISS e agente...")
def carregar_agente():
    return bb.inicializar()


agente = carregar_agente()

if "sessao" not in st.session_state:
    st.session_state.sessao = "web-" + uuid.uuid4().hex[:8]
    st.session_state.mensagens = []

with st.sidebar:
    st.subheader("Sessão")
    cliente_id = st.text_input("ID do cliente", value="cliente_demo", help="Chave da memória de longo prazo (preferências).")
    st.caption(f"Sessão: {st.session_state.sessao}")
    if st.button("Nova sessão"):
        st.session_state.sessao = "web-" + uuid.uuid4().hex[:8]
        st.session_state.mensagens = []
        st.rerun()
    mostrar_raciocinio = st.toggle("Mostrar raciocínio do agente", value=True)
    st.caption("A memória de curto prazo vive na sessão; as preferências gravadas persistem por ID de cliente.")


def desenhar_verificacao(meta: dict) -> None:
    v = meta["verificacao"]
    with st.expander(f"Verificação: {'aprovada' if v['aprovada'] else 'reprovada'} · ferramentas: {', '.join(meta['ferramentas']) or '-'} · {meta['latencia_s']} s"):
        sustentacao = v["taxa_sustentacao"]
        st.write(f"Afirmações verificadas: **{v['afirmacoes']}** · taxa de sustentação: **{'n/d' if sustentacao is None else format(sustentacao, '.0%')}** · correções: **{meta['correcoes']}**")
        if v["nli"]:
            st.markdown("**NLI frente à evidência das ferramentas**")
            st.dataframe(pd.DataFrame(v["nli"]), use_container_width=True, hide_index=True)
        oficiais = [d for d in v["deterministica"] if d["status"] != "NÃO MAPEADO"]
        if oficiais:
            st.markdown("**Checagem contra a fonte oficial**")
            st.dataframe(pd.DataFrame(oficiais), use_container_width=True, hide_index=True)
        st.markdown(f"**Coerência lógica:** {'coerente' if v['coerencia']['coerente'] else 'incoerente'} · {v['coerencia']['justificativa']}")
        for motivo in v["motivos"]:
            st.warning(motivo)
        for alerta in v.get("alertas", []):
            st.info(f"Alerta (não bloqueia): {alerta}")


for m in st.session_state.mensagens:
    with st.chat_message(m["papel"]):
        st.markdown(m["texto"])
        if m.get("meta"):
            desenhar_verificacao(m["meta"])

pergunta = st.chat_input("Pergunte sobre livros, preços, frete, prazos, políticas ou faça um pedido")
if pergunta:
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"):
        callbacks = []
        if mostrar_raciocinio:
            callbacks.append(StreamlitCallbackHandler(st.container(), expand_new_thoughts=True))
        with st.spinner("Consultando ferramentas e verificando a resposta..."):
            r = agente.responder(pergunta, st.session_state.sessao, cliente_id, callbacks=callbacks)
        st.markdown(r["resposta"])
        meta = {k: r[k] for k in ("verificacao", "ferramentas", "latencia_s", "correcoes")}
        desenhar_verificacao(meta)
    st.session_state.mensagens.append({"papel": "assistant", "texto": r["resposta"], "meta": meta})
