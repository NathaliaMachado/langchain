"""Agente BookBridge: módulo gerado a partir do notebook, com as mesmas ferramentas, memória e verificação."""
import os

if __name__ != "__main__":
    os.environ.setdefault("BOOKBRIDGE_MODO", "modulo")

# --- cfg ---
import os
import re
import json
import time
import uuid
import sqlite3
import threading
from getpass import getpass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd


def obter_segredo(nome: str, obrigatorio: bool = False) -> Optional[str]:
    valor = os.environ.get(nome)
    if not valor:
        try:
            from google.colab import userdata
            valor = userdata.get(nome)
        except Exception:
            valor = None
    if not valor and obrigatorio and os.environ.get("BOOKBRIDGE_MODO") != "modulo":
        valor = getpass(f"{nome} (Enter para seguir em modo sem credenciais): ")
    if valor:
        os.environ[nome] = valor
    return valor or None


CHAVE_OPENAI = obter_segredo("OPENAI_API_KEY", obrigatorio=True)
CHAVE_LANGSMITH = obter_segredo("LANGSMITH_API_KEY")

os.environ["LANGSMITH_PROJECT"] = "bookbridge-agentes-cognitivos"
os.environ["LANGSMITH_TRACING"] = "true" if CHAVE_LANGSMITH else "false"

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

MODELO_CHAT = "gpt-4o-mini"
MODELO_EMBEDDING = "text-embedding-3-small"

if CHAVE_OPENAI:
    llm = ChatOpenAI(model=MODELO_CHAT, temperature=0, max_tokens=2048)
    embeddings = OpenAIEmbeddings(model=MODELO_EMBEDDING)
else:
    llm = None
    embeddings = DeterministicFakeEmbedding(size=256)

PASTA_INDICE = "indice_faiss_bookbridge" + ("" if CHAVE_OPENAI else "_dummy")

print(f"Modelo de chat      : {MODELO_CHAT if llm else 'indisponível (sem OPENAI_API_KEY)'}")
print(f"Embeddings          : {MODELO_EMBEDDING if CHAVE_OPENAI else 'determinístico (modo sem credenciais)'}")
print(f"LangSmith           : {'ativo, projeto ' + os.environ['LANGSMITH_PROJECT'] if CHAVE_LANGSMITH else 'inativo (avaliação local; defina LANGSMITH_API_KEY para registrar no LangSmith)'}")

from datetime import date
from operator import itemgetter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unicodedata
import httpx
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- compat ---
try:
    from langchain_classic.agents import AgentExecutor, AgentType, create_openai_tools_agent, initialize_agent
    from langchain_classic.chains import LLMChain
    from langchain_classic.chains.router import MultiPromptChain
    from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain_classic.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
    from langchain_classic.memory import ConversationBufferMemory
    ORIGEM_CLASSICOS = "langchain_classic"
except ImportError:
    from langchain.agents import AgentExecutor, AgentType, create_openai_tools_agent, initialize_agent
    from langchain.chains import LLMChain
    from langchain.chains.router import MultiPromptChain
    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
    from langchain.memory import ConversationBufferMemory
    ORIGEM_CLASSICOS = "langchain"

print("Construtos clássicos carregados de:", ORIGEM_CLASSICOS)

# --- schemas_triagem ---
from pydantic import BaseModel, Field


class Entidades(BaseModel):
    titulo_livro: Optional[str] = Field(default=None, description="Título do livro citado, como o cliente escreveu")
    quantidade: Optional[int] = Field(default=None, description="Quantidade de exemplares")
    cep: Optional[str] = Field(default=None, description="CEP citado, somente dígitos")
    forma_pagamento: Optional[str] = Field(default=None, description="Pix, cartão ou outra forma citada")


class Triagem(BaseModel):
    intencao: Literal[
        "consultar_preco", "consultar_estoque", "consultar_politica", "calcular_frete",
        "fazer_pedido", "acompanhar_pedido", "solicitar_devolucao", "recomendar_livro", "saudacao", "fora_de_escopo",
    ] = Field(description="Intenção principal da mensagem")
    entidades: Entidades
    urgencia: Literal["baixa", "normal", "alta"]
    resumo: str = Field(description="Resumo em até 15 palavras")

# --- dados ---
DB_PATH = "bookbridge.db"

LIVROS = [
    (1, "Effective Java", "Joshua Bloch", "Programação", 54.00, 12, "Inglês"),
    (2, "Designing Data-Intensive Applications", "Martin Kleppmann", "Engenharia de Dados", 62.00, 9, "Inglês"),
    (3, "AI Engineering", "Chip Huyen", "Inteligência Artificial", 69.00, 15, "Inglês"),
    (4, "Designing Machine Learning Systems", "Chip Huyen", "Inteligência Artificial", 59.00, 7, "Inglês"),
    (5, "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow", "Aurélien Géron", "Inteligência Artificial", 78.00, 0, "Inglês"),
    (6, "Deep Learning", "Ian Goodfellow, Yoshua Bengio e Aaron Courville", "Inteligência Artificial", 89.00, 5, "Inglês"),
    (7, "Build a Large Language Model (From Scratch)", "Sebastian Raschka", "Inteligência Artificial", 49.00, 20, "Inglês"),
    (8, "Clean Code", "Robert C. Martin", "Programação", 42.00, 18, "Inglês"),
    (9, "Pattern Recognition and Machine Learning", "Christopher Bishop", "Inteligência Artificial", 95.00, 3, "Inglês"),
    (10, "Speech and Language Processing", "Dan Jurafsky e James H. Martin", "Inteligência Artificial", 84.00, 6, "Inglês"),
]

PEDIDOS = [
    (1001, "Ana Souza", 3, 1, "entregue", "2026-09-02"),
    (1002, "Bruno Lima", 1, 2, "em transporte", "2026-10-28"),
    (1003, "Carla Dias", 7, 1, "processando", "2026-11-04"),
    (1004, "Ana Souza", 8, 1, "entregue", "2026-09-02"),
    (1005, "Diego Alves", 6, 1, "cancelado", "2026-10-10"),
]

SCHEMA_SQL = """Tabelas do banco SQLite:
livros(id INTEGER, titulo TEXT, autor TEXT, categoria TEXT, preco_usd REAL, estoque INTEGER, idioma TEXT)
pedidos(id INTEGER, cliente TEXT, livro_id INTEGER, quantidade INTEGER, status TEXT, data_pedido TEXT)
Relação: pedidos.livro_id referencia livros.id.
Valores de status: 'processando', 'em transporte', 'entregue', 'cancelado'."""


def criar_banco() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE livros (
            id INTEGER PRIMARY KEY, titulo TEXT NOT NULL, autor TEXT NOT NULL, categoria TEXT NOT NULL,
            preco_usd REAL NOT NULL, estoque INTEGER NOT NULL, idioma TEXT NOT NULL);
        CREATE TABLE pedidos (
            id INTEGER PRIMARY KEY, cliente TEXT NOT NULL, livro_id INTEGER NOT NULL,
            quantidade INTEGER NOT NULL, status TEXT NOT NULL, data_pedido TEXT NOT NULL,
            FOREIGN KEY (livro_id) REFERENCES livros(id));
        """
    )
    con.executemany("INSERT INTO livros VALUES (?,?,?,?,?,?,?)", LIVROS)
    con.executemany("INSERT INTO pedidos VALUES (?,?,?,?,?,?)", PEDIDOS)
    con.commit()
    con.close()


def conexao_somente_leitura() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


DOCUMENTOS_POLITICAS = [
    {
        "fonte": "BB-FRETE-2026-V2",
        "dominio": "frete e entrega",
        "titulo": "POLÍTICA DE FRETE E ENTREGA (COD: BB-FRETE-2026-V2)",
        "secoes": [
            "1. TARIFAS POR REGIÃO: o frete é calculado pela região da UF de destino. Sudeste: R$ 19,90. Sul e Centro-Oeste: R$ 24,90. Nordeste: R$ 29,90. Norte: R$ 39,90.",
            "2. ISENÇÃO DE FRETE: pedidos com subtotal igual ou superior a R$ 250,00 recebem isenção total do frete, em qualquer região. O subtotal considera apenas o valor dos livros, sem o frete.",
            "3. PRAZOS DE ENTREGA: livros em estoque nacional são entregues em 2 a 5 dias úteis no Sudeste e em 4 a 8 dias úteis nas demais regiões. Livros importados sob encomenda, sem estoque nacional, levam de 10 a 15 dias úteis contados da confirmação do pagamento.",
            "4. RASTREAMENTO: o código de rastreio é enviado por e-mail em até 24 horas após a postagem. Endereço incorreto informado pelo cliente gera novo frete de reenvio por conta do cliente.",
        ],
    },
    {
        "fonte": "BB-TROCAS-2026-V3",
        "dominio": "trocas e devoluções",
        "titulo": "POLÍTICA DE TROCAS, DEVOLUÇÕES E ESTORNOS (COD: BB-TROCAS-2026-V3)",
        "secoes": [
            "1. DIREITO DE ARREPENDIMENTO: o cliente pode desistir da compra no prazo de 7 (sete) dias corridos contados da data de recebimento do pedido, sem necessidade de justificativa, desde que o exemplar esteja lacrado ou sem marcas de uso.",
            "2. LOGÍSTICA REVERSA: para exercer o arrependimento, o cliente abre um chamado informando o número do pedido. A BookBridge emite um código de postagem reversa gratuito, válido por 10 dias corridos.",
            "3. ESTORNO: após o recebimento e a conferência do exemplar devolvido, o estorno integral do valor dos livros é processado em até 5 dias úteis, na mesma forma de pagamento. O frete pago na compra é estornado quando a desistência ocorre dentro do prazo de arrependimento.",
            "4. EXEMPLAR AVARIADO OU DEFEITUOSO: livros que chegarem com dano de transporte, páginas faltantes ou erro de impressão podem ser reenviados ou estornados, à escolha do cliente, em até 30 dias corridos do recebimento, mediante envio de fotos.",
            "5. ITENS NÃO ELEGÍVEIS: livros com lacre violado ou marcas de uso não são elegíveis ao arrependimento, salvo defeito de fabricação comprovado.",
        ],
    },
    {
        "fonte": "BB-PAGAMENTO-2026-V1",
        "dominio": "pagamento e câmbio",
        "titulo": "PAGAMENTO, CÂMBIO E TRIBUTOS DE IMPORTAÇÃO (COD: BB-PAGAMENTO-2026-V1)",
        "secoes": [
            "1. PREÇOS E CÂMBIO: os preços do catálogo são cadastrados em dólares americanos (USD). O valor em reais é calculado pela cotação USD/BRL do momento da compra, acrescida de spread operacional de 2% (dois por cento) sobre a cotação.",
            "2. TRIBUTOS: os tributos de importação já estão incluídos no preço final exibido ao cliente. Não há cobrança de taxas na entrega.",
            "3. FORMAS DE PAGAMENTO: aceitamos Pix, com desconto de 5% sobre o valor dos livros, e cartão de crédito em até 6 parcelas sem juros para compras a partir de R$ 300,00. Não aceitamos boleto bancário.",
            "4. CONFIRMAÇÃO: o pagamento via Pix é confirmado em até 10 minutos e o de cartão em até 2 horas. Pedidos não pagos em 24 horas são cancelados automaticamente.",
        ],
    },
    {
        "fonte": "BB-FIDELIDADE-2026-V1",
        "dominio": "fidelidade e atendimento",
        "titulo": "PROGRAMA DE FIDELIDADE E ATENDIMENTO (COD: BB-FIDELIDADE-2026-V1)",
        "secoes": [
            "1. ACÚMULO: cada R$ 10,00 gastos em livros geram 1 ponto BookPoints. Os pontos expiram em 12 meses.",
            "2. RESGATE: cada 500 pontos equivalem a R$ 25,00 de desconto, limitado a 20% do subtotal do pedido.",
            "3. ATENDIMENTO: o atendimento humano funciona de segunda a sexta, das 9h às 18h, horário de Brasília. O assistente virtual funciona 24 horas por dia, mas não executa estornos nem altera pedidos já enviados.",
            "4. PRIVACIDADE: os dados do cliente são usados apenas para processar pedidos e personalizar recomendações; o cliente pode pedir a exclusão do histórico de preferências a qualquer momento.",
        ],
    },
]

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preparar_chunks() -> List[Document]:
    divisor_secundario = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks: List[Document] = []
    for doc in DOCUMENTOS_POLITICAS:
        for secao in doc["secoes"]:
            texto = f"{doc['titulo']}\n{secao}"
            pedacos = [texto] if len(texto) <= 700 else divisor_secundario.split_text(texto)
            for pedaco in pedacos:
                chunks.append(Document(page_content=pedaco, metadata={"fonte": doc["fonte"], "dominio": doc["dominio"]}))
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"CHK_{i:02d}"
    return chunks


def construir_indice():
    chunks = preparar_chunks()
    indice = FAISS.from_documents(chunks, embeddings)
    indice.save_local(PASTA_INDICE)
    return indice, chunks


def carregar_indice():
    return FAISS.load_local(PASTA_INDICE, embeddings, allow_dangerous_deserialization=True)

# --- tools_api ---
import httpx
from langchain_core.tools import tool

COTACAO_CONTINGENCIA = 5.50

UF_REGIAO = {
    "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "BA": "Nordeste", "SE": "Nordeste", "AL": "Nordeste", "PE": "Nordeste", "PB": "Nordeste",
    "RN": "Nordeste", "CE": "Nordeste", "PI": "Nordeste", "MA": "Nordeste",
    "AM": "Norte", "PA": "Norte", "AC": "Norte", "RO": "Norte", "RR": "Norte", "AP": "Norte", "TO": "Norte",
}


def _http_json(url: str, timeout: float = 8.0) -> Any:
    resposta = httpx.get(url, timeout=timeout, follow_redirects=True)
    resposta.raise_for_status()
    return resposta.json()


def obter_cotacao_usd_brl() -> Dict[str, Any]:
    fontes = [
        ("frankfurter.dev", "https://api.frankfurter.dev/v1/latest?base=USD&symbols=BRL",
         lambda j: (j["rates"]["BRL"], j["date"])),
        ("frankfurter.app", "https://api.frankfurter.app/latest?from=USD&to=BRL",
         lambda j: (j["rates"]["BRL"], j["date"])),
        ("open.er-api.com", "https://open.er-api.com/v6/latest/USD",
         lambda j: (j["rates"]["BRL"], j["time_last_update_utc"])),
    ]
    for nome, url, extrair in fontes:
        try:
            taxa, data = extrair(_http_json(url))
            return {"taxa": float(taxa), "data": str(data), "fonte": nome, "ao_vivo": True}
        except Exception:
            continue
    return {"taxa": COTACAO_CONTINGENCIA, "data": "indisponível", "fonte": "fallback-offline", "ao_vivo": False}


@tool
def cotacao_usd_brl(consulta: str = "USD/BRL") -> str:
    """Consulta em uma API pública externa a cotação atual do dólar americano (USD) em reais (BRL).
    Use sempre que for preciso converter preços do catálogo, que estão em USD, para reais.
    A entrada é ignorada; pode ser 'USD/BRL'."""
    dados = obter_cotacao_usd_brl()
    aviso = "" if dados["ao_vivo"] else " [AVISO: valor de contingência, API indisponível]"
    return f"Cotação USD/BRL: R$ {dados['taxa']:.4f} (data {dados['data']}, fonte {dados['fonte']}){aviso}"


def regiao_estimada_por_prefixo(cep_limpo: str) -> str:
    p = int(cep_limpo[:2])
    if p <= 39:
        return "Sudeste"
    if p <= 65:
        return "Nordeste"
    if p <= 69 or p == 77:
        return "Norte"
    if p <= 76 or p in (78, 79):
        return "Centro-Oeste"
    return "Sul"


def obter_endereco_por_cep(cep: str) -> Dict[str, Any]:
    cep_limpo = re.sub(r"\D", "", str(cep))
    if len(cep_limpo) != 8:
        return {"erro": "CEP inválido: informe 8 dígitos."}
    tentativas = [
        ("viacep", f"https://viacep.com.br/ws/{cep_limpo}/json/",
         lambda j: None if j.get("erro") else {"cidade": j["localidade"], "uf": j["uf"]}),
        ("brasilapi", f"https://brasilapi.com.br/api/cep/v1/{cep_limpo}",
         lambda j: {"cidade": j["city"], "uf": j["state"]}),
    ]
    for nome, url, extrair in tentativas:
        try:
            achado = extrair(_http_json(url))
            if achado:
                achado.update({"fonte": nome, "cep": cep_limpo, "ao_vivo": True})
                return achado
        except Exception:
            continue
    return {"cep": cep_limpo, "cidade": None, "uf": None, "regiao_estimada": regiao_estimada_por_prefixo(cep_limpo),
            "fonte": "fallback-offline", "ao_vivo": False}


@tool
def consultar_cep(cep: str) -> str:
    """Consulta em APIs públicas externas (ViaCEP, com BrasilAPI de reserva) a cidade e a UF de um CEP brasileiro.
    Use para descobrir a UF de destino antes de calcular o frete. A entrada é o CEP, com ou sem hífen."""
    d = obter_endereco_por_cep(cep)
    if "erro" in d:
        return d["erro"]
    if not d["ao_vivo"]:
        return (f"CEP {d['cep']}: API indisponível; região estimada pelo prefixo: {d['regiao_estimada']} "
                f"[AVISO: estimativa, UF não confirmada]")
    return f"CEP {d['cep']}: {d['cidade']}/{d['uf']} (região {UF_REGIAO.get(d['uf'], 'desconhecida')}, fonte {d['fonte']})"

# --- tools_rag ---
K_RECUPERACAO = 3
INDICE = None
ESTRATEGIAS: Dict[str, Any] = {}
ESTRATEGIA_ATUAL = {"nome": "simples"}


def recuperar_simples(consulta: str, k: int = K_RECUPERACAO):
    return INDICE.similarity_search_with_score(consulta, k=k)


ESTRATEGIAS["simples"] = recuperar_simples


def recuperar(consulta: str, k: int = K_RECUPERACAO):
    return ESTRATEGIAS[ESTRATEGIA_ATUAL["nome"]](consulta, k)


def formatar_trechos(resultados) -> str:
    blocos = []
    for doc, score in resultados:
        m = doc.metadata
        blocos.append(f"[Fonte: {m['fonte']} | Chunk: {m['chunk_id']} | Distância L2: {score:.3f}]\n{doc.page_content}")
    return "\n\n".join(blocos)


@tool
def buscar_politicas(consulta: str) -> str:
    """Busca semântica nas políticas oficiais da BookBridge: frete e prazos de entrega, trocas, devoluções e estornos,
    pagamento, câmbio e tributos, fidelidade e atendimento. Retorna trechos com fonte e chunk.
    Use para qualquer regra, prazo, tarifa ou condição da loja; nunca responda regras de memória.
    Atenção: desistência ou arrependimento da compra (7 dias) e troca por defeito ou avaria (30 dias) são regras diferentes;
    na consulta, diga qual delas o cliente pergunta."""
    if INDICE is None:
        return "Índice de políticas indisponível."
    return formatar_trechos(recuperar(consulta))

# --- tools_sql ---
import sqlite3 as _sqlite3
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

PALAVRAS_PROIBIDAS_SQL = ("insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
                          "attach", "detach", "pragma", "vacuum", "reindex")


class ConsultaSQL(BaseModel):
    sql: str = Field(description="Uma única consulta SELECT válida para SQLite, sem ponto e vírgula no meio")
    raciocinio: str = Field(description="Uma frase explicando como a pergunta virou a consulta")


def validar_sql(sql: str) -> str:
    limpo = sql.strip().rstrip(";").strip()
    minusculo = limpo.lower()
    if ";" in limpo:
        raise ValueError("somente uma instrução é permitida")
    if not minusculo.startswith("select"):
        raise ValueError("somente consultas SELECT são permitidas")
    if "--" in limpo or "/*" in limpo:
        raise ValueError("comentários não são permitidos")
    for palavra in PALAVRAS_PROIBIDAS_SQL:
        if re.search(rf"\b{palavra}\b", minusculo):
            raise ValueError(f"palavra proibida: {palavra}")
    if "sqlite_" in minusculo:
        raise ValueError("tabelas internas do SQLite não são acessíveis")
    return limpo


def executar_sql_somente_leitura(sql: str, limite: int = 20) -> List[Dict[str, Any]]:
    limpo = validar_sql(sql)
    con = conexao_somente_leitura()
    try:
        linhas = con.execute(f"SELECT * FROM ({limpo}) LIMIT {int(limite)}").fetchall()
        return [dict(l) for l in linhas]
    finally:
        con.close()


PROMPT_TEXT_TO_SQL = ChatPromptTemplate.from_messages([
    ("system",
     "Você traduz perguntas em português para UMA consulta SQLite somente leitura.\n{schema}\n"
     "Regras: use apenas SELECT. Para buscar por título ou autor use LOWER(coluna) LIKE '%termo%' com o termo em minúsculas. "
     "Preços estão em USD na coluna preco_usd. estoque = 0 significa sem estoque nacional. "
     "Selecione somente as colunas necessárias para responder."),
    ("human", "{pergunta}"),
])


def text_to_sql(pergunta: str, tentativas: int = 2) -> Dict[str, Any]:
    gerador = llm.with_structured_output(ConsultaSQL, method="function_calling")
    mensagens = PROMPT_TEXT_TO_SQL.format_messages(schema=SCHEMA_SQL, pergunta=pergunta)
    historico_erros: List[str] = []
    for tentativa in range(tentativas + 1):
        consulta = gerador.invoke(mensagens)
        try:
            linhas = executar_sql_somente_leitura(consulta.sql)
            return {"sql": consulta.sql, "linhas": linhas, "tentativas": tentativa + 1, "erros": historico_erros}
        except (ValueError, _sqlite3.Error) as erro:
            historico_erros.append(f"{consulta.sql} -> {erro}")
            mensagens = mensagens + [HumanMessage(
                content=f"A consulta anterior falhou: {erro}. SQL anterior: {consulta.sql}. Corrija e devolva uma nova consulta."
            )]
    return {"sql": None, "linhas": [], "tentativas": tentativas + 1, "erros": historico_erros}


@tool
def consultar_dados_sql(pergunta: str) -> str:
    """Responde perguntas sobre dados estruturados da BookBridge: catálogo (título, autor, categoria, preço em USD, estoque)
    e pedidos (cliente, status, data). Recebe a pergunta em português, converte em SQL somente leitura, executa e devolve
    o SQL usado e as linhas. Use para preço, disponibilidade, autor, categoria e situação de pedidos."""
    r = text_to_sql(pergunta)
    if r["sql"] is None:
        return "Não consegui montar uma consulta válida para essa pergunta. Erros: " + " | ".join(r["erros"])
    if not r["linhas"]:
        return f"SQL executado: {r['sql']}\nResultado: nenhum registro encontrado."
    return f"SQL executado: {r['sql']}\nResultado: {json.dumps(r['linhas'], ensure_ascii=False)}"

# --- ptcf ---
PTCF_PERSONA = (
    "PERSONA: Você é a Bia, atendente virtual da BookBridge, livraria de livros técnicos importados. "
    "Seu tom é cordial, direto e preciso; você nunca inventa fatos."
)
PTCF_TASK = (
    "TAREFA: ajudar o cliente a consultar o catálogo, orçar e registrar pedidos, e esclarecer regras da loja. "
    "Antes de agir, decomponha o pedido em subtarefas e use as ferramentas na ordem certa. "
    "Se faltar um dado necessário (título ambíguo, CEP, quantidade), pergunte em vez de supor. "
    "Se o pedido já traz todos os dados necessários, use a ferramenta sem fazer perguntas."
)
PTCF_CONTEXT = (
    "CONTEXTO: preços do catálogo estão em USD. Para converter, use a cotação da ferramenta de câmbio e a ferramenta de cálculo; "
    "nunca faça contas de cabeça e use apenas a cotação retornada pela ferramenta de câmbio nesta mesma consulta. Regras de frete, prazo, troca e pagamento só valem se vierem da busca de políticas. "
    "Dados de livros e pedidos só valem se vierem da consulta SQL. O ID do cliente desta sessão é '{cliente_id}'."
)
PTCF_FORMAT = (
    "FORMATO: responda em português, em até 6 linhas. Cite valores com unidade (R$ ou US$). "
    "Ao usar regras de políticas, cite a fonte entre colchetes, por exemplo [BB-FRETE-2026-V2]. "
    "Antes de registrar um pedido, resuma itens, total e frete e peça confirmação explícita do cliente. "
    "Só chame a ferramenta de registro depois de uma mensagem do cliente confirmando esse resumo, e nunca registre duas vezes o mesmo pedido. "
    "Antes de dizer que a base não cobre um assunto da empresa, consulte a busca de políticas; se nada relevante vier, diga que não há informação, sem inventar. "
    "Se a mensagem do cliente já confirma explicitamente os itens e o total, registre o pedido sem pedir nova confirmação. "
    "Reproduza prazos, valores e limites exatamente como constam na política ou no retorno das ferramentas, sem combinar prazos diferentes "
    "(por exemplo, desistência e estorno) em uma única frase. Quando o frete for grátis, diga apenas isso e cite o limite de isenção somente se o cliente perguntar. "
    "No resumo de um pedido, informe título, quantidade, subtotal em reais e frete."
)
PTCF_EXEMPLOS = (
    "EXEMPLOS DE DECISÃO:\n"
    "- Cliente: 'Quanto custa o do Chip?' -> há dois livros dela no catálogo; pergunte qual antes de consultar preço.\n"
    "- Cliente: 'Quanto é o frete pro Rio?' -> falta o subtotal para saber se há isenção; pergunte quais livros e quantas unidades.\n"
    "- Cliente: 'Posso pagar no boleto?' -> consulte a política de pagamento antes de responder.\n"
    "- Cliente: 'Quanto tempo leva para o Recife? O livro tem estoque.' -> UF e estoque bastam; chame estimar_prazo_entrega sem pedir o título.\n"
    "- Cliente: 'Oi!' -> cumprimente; não use ferramentas."
)
PTCF_SISTEMA = "\n\n".join([PTCF_PERSONA, PTCF_TASK, PTCF_CONTEXT, PTCF_FORMAT, PTCF_EXEMPLOS])

# --- planejador ---
from langchain_core.prompts import FewShotChatMessagePromptTemplate


class PlanoAgente(BaseModel):
    raciocinio: List[str] = Field(description="Passos de raciocínio curtos, em ordem")
    ferramentas: List[str] = Field(description="Nomes das ferramentas que serão usadas, em ordem")
    dados_faltantes: List[str] = Field(description="Dados que o cliente ainda precisa informar; lista vazia se nada falta")
    pergunta_esclarecimento: Optional[str] = Field(default=None, description="Pergunta ao cliente, ou null se nada falta")


EXEMPLOS_FEW_SHOT = [
    {
        "input": "Quanto custa o do Chip?",
        "output": json.dumps({
            "raciocinio": ["O cliente citou apenas a autora, não o título.", "O catálogo tem dois livros dela, então o pedido é ambíguo."],
            "ferramentas": [], "dados_faltantes": ["título"],
            "pergunta_esclarecimento": "Você quer o AI Engineering ou o Designing Machine Learning Systems?"}, ensure_ascii=False),
    },
    {
        "input": "Quanto custa o Clean Code em reais?",
        "output": json.dumps({
            "raciocinio": ["Título claro e único.", "Preço está em USD no SQL.", "Preciso da cotação e do cálculo com spread."],
            "ferramentas": ["consultar_dados_sql", "cotacao_usd_brl", "calcular_preco_em_brl"], "dados_faltantes": [],
            "pergunta_esclarecimento": None}, ensure_ascii=False),
    },
]

exemplo_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
bloco_few_shot = FewShotChatMessagePromptTemplate(example_prompt=exemplo_prompt, examples=EXEMPLOS_FEW_SHOT)

prompt_planejador = ChatPromptTemplate.from_messages([
    ("system", PTCF_PERSONA + "\n\n"
     "TAREFA: antes de agir, produza um plano. Pense passo a passo e registre o raciocínio no campo 'raciocinio'. "
     "Se faltar dado essencial ou o pedido for ambíguo, preencha 'pergunta_esclarecimento'; se estiver claro, deixe null.\n\n"
     "CONTEXTO: ferramentas disponíveis: consultar_dados_sql (catálogo e pedidos), cotacao_usd_brl, calcular_preco_em_brl, "
     "consultar_cep, calcular_frete, buscar_politicas, estimar_prazo_entrega, registrar_pedido, recomendar_leitura. "
     "O catálogo tem 10 livros; Chip Huyen tem 2 livros no catálogo.\n\n"
     "FORMATO: JSON no esquema pedido."),
    bloco_few_shot,
    ("human", "{input}"),
])
planejador = prompt_planejador | llm.with_structured_output(PlanoAgente, method="function_calling")

prompt_simples = ChatPromptTemplate.from_messages([
    ("system", "Planeje como atender o pedido do cliente de uma livraria."),
    ("human", "{input}"),
])
planejador_simples = prompt_simples | llm.with_structured_output(PlanoAgente, method="function_calling")

# --- reparo ---
from langchain_core.exceptions import OutputParserException

PROMPT_REPARO = ChatPromptTemplate.from_template(
    "A saída abaixo falhou ao ser validada pelo parser.\n\nSaída: {saida}\n\nErro: {erro}\n\n"
    "Esquema esperado:\n{formato}\n\nCorrija e devolva somente o JSON válido."
)


def parse_com_reparo(texto_bruto: str, parser: PydanticOutputParser, tentativas: int = 2):
    ultimo_erro = None
    for tentativa in range(tentativas + 1):
        try:
            return parser.parse(texto_bruto), tentativa
        except OutputParserException as erro:
            ultimo_erro = str(erro)[:300]
            if tentativa == tentativas:
                break
            texto_bruto = (PROMPT_REPARO | llm | StrOutputParser()).invoke(
                {"saida": texto_bruto, "erro": ultimo_erro, "formato": parser.get_format_instructions()}
            )
    raise OutputParserException(f"Falha após {tentativas} reparos: {ultimo_erro}")

# --- callbacks ---
from langchain_core.callbacks import BaseCallbackHandler


class RegistroExecucao(BaseCallbackHandler):
    def __init__(self, verboso: bool = True):
        self.verboso = verboso
        self.eventos: List[Dict[str, Any]] = []
        self.tokens_entrada = 0
        self.tokens_saida = 0
        self._inicio: Dict[Any, float] = {}
        self._t0 = time.perf_counter()

    def _registrar(self, tipo: str, **dados) -> None:
        evento = {"t": round(time.perf_counter() - self._t0, 2), "tipo": tipo, **dados}
        self.eventos.append(evento)
        if self.verboso:
            detalhes = " | ".join(f"{k}={v}" for k, v in dados.items())
            print(f"  [{evento['t']:>6.2f}s] {tipo:<20} {detalhes}")

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        nome = (serialized or {}).get("name") or kwargs.get("name")
        self._inicio[run_id] = time.perf_counter()
        self._registrar("ferramenta_inicio", ferramenta=nome, entrada=str(input_str)[:90])

    def on_tool_end(self, output, *, run_id, **kwargs):
        duracao = time.perf_counter() - self._inicio.pop(run_id, time.perf_counter())
        conteudo = getattr(output, "content", output)
        self._registrar("ferramenta_fim", duracao_s=round(duracao, 2), saida=str(conteudo).replace("\n", " ")[:90])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._registrar("ferramenta_erro", erro=str(error)[:120])

    def on_llm_end(self, response, **kwargs):
        for geracoes in response.generations:
            for g in geracoes:
                uso = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.tokens_entrada += uso.get("input_tokens", 0)
                self.tokens_saida += uso.get("output_tokens", 0)

    def on_chain_error(self, error, **kwargs):
        self._registrar("cadeia_erro", erro=str(error)[:120])

    def resumo(self) -> Dict[str, Any]:
        ferramentas = [e["ferramenta"] for e in self.eventos if e["tipo"] == "ferramenta_inicio"]
        return {"ferramentas": ferramentas, "erros": sum(e["tipo"].endswith("erro") for e in self.eventos),
                "tokens_entrada": self.tokens_entrada, "tokens_saida": self.tokens_saida,
                "duracao_s": round(time.perf_counter() - self._t0, 2)}

# --- memoria ---
from datetime import date

from langchain_core.vectorstores import InMemoryVectorStore

MEMORIA_LONGA = InMemoryVectorStore(embedding=embeddings)


def registrar_memoria(cliente_id: str, fato: str, tipo: str = "preferencia") -> None:
    MEMORIA_LONGA.add_documents([Document(
        page_content=fato,
        metadata={"cliente_id": cliente_id, "tipo": tipo, "data": date.today().isoformat()},
    )])


def buscar_memoria(cliente_id: str, consulta: str, k: int = 3) -> List[Document]:
    return MEMORIA_LONGA.similarity_search(consulta, k=k, filter=lambda d: d.metadata.get("cliente_id") == cliente_id)


@tool
def gravar_preferencia(cliente_id: str, fato: str) -> str:
    """Grava na memória de longo prazo do cliente uma preferência ou fato duradouro dito por ele
    (forma de pagamento preferida, orçamento, área de interesse, endereço de entrega habitual).
    Use quando o cliente pedir para lembrar ou disser algo que valerá em sessões futuras."""
    registrar_memoria(cliente_id, fato)
    return f"Memorizado para o cliente {cliente_id}: {fato}"


@tool
def consultar_preferencias(cliente_id: str, consulta: str) -> str:
    """Busca na memória de longo prazo do cliente preferências e fatos de sessões anteriores.
    Use quando o cliente se referir ao passado ('como sempre', 'o que eu disse', 'minha preferência')."""
    achados = buscar_memoria(cliente_id, consulta)
    if not achados:
        return "Nenhuma memória encontrada para este cliente."
    return "\n".join(f"- [{d.metadata['data']}] {d.page_content}" for d in achados)

# --- contexto ---
from langchain_core.messages import SystemMessage


class GerenciadorContexto:
    def __init__(self, janela_mensagens: int = 6):
        self.janela = janela_mensagens

    def montar(self, historico: list, consulta: str, cliente_id: str) -> list:
        recentes = trim_messages(historico, max_tokens=self.janela, strategy="last", token_counter=len,
                                 start_on="human", include_system=False)
        antigas = historico[: len(historico) - len(recentes)]
        partes = []
        if antigas:
            texto = "\n".join(f"{m.type}: {m.content}" for m in antigas)
            resumo = (ChatPromptTemplate.from_template(
                "Resuma em até 4 frases a conversa abaixo, mantendo nomes, números e preferências:\n\n{texto}")
                | llm | StrOutputParser()).invoke({"texto": texto})
            partes.append(f"Resumo da conversa anterior: {resumo}")
        memorias = buscar_memoria(cliente_id, consulta)
        if memorias:
            partes.append("Memórias do cliente (longo prazo):\n" + "\n".join(f"- {d.page_content}" for d in memorias))
        prefixo = [SystemMessage(content="\n\n".join(partes))] if partes else []
        return prefixo + list(recentes)

# --- executor ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def formatar_erro_validacao(erro) -> str:
    """Devolve ao modelo, como texto, o erro de validação dos argumentos de uma ferramenta, para que ele corrija a chamada."""
    return ("Argumentos inválidos para a ferramenta: " + str(erro).splitlines()[0] +
            ". Se faltar algum dado, obtenha-o antes com outra ferramenta ou pergunte ao cliente; não invente valores.")


def construir_executor(ferramentas, prompt_sistema: str = None, memoria=None, max_iteracoes: int = 8) -> AgentExecutor:
    for ferramenta in ferramentas:
        ferramenta.handle_validation_error = formatar_erro_validacao
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema or PTCF_SISTEMA),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agente = create_openai_tools_agent(llm, ferramentas, prompt)
    return AgentExecutor(agent=agente, tools=ferramentas, memory=memoria, max_iterations=max_iteracoes,
                         return_intermediate_steps=True, handle_parsing_errors=True)

# --- tools_dominio ---
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARIFA_FRETE_BRL = {"Sudeste": 19.90, "Sul": 24.90, "Centro-Oeste": 24.90, "Nordeste": 29.90, "Norte": 39.90}
LIMITE_FRETE_GRATIS_BRL = 250.00
SPREAD_CAMBIAL = 0.02


@tool
def calcular_preco_em_brl(valor_usd: float, cotacao: float) -> str:
    """Converte um valor em USD para reais (BRL) com a regra oficial da loja: valor_usd x cotação, mais spread de 2%.
    Use sempre esta ferramenta para contas de câmbio, com a cotação devolvida por cotacao_usd_brl; nunca calcule de cabeça."""
    oficial = obter_cotacao_usd_brl()["taxa"]
    if abs(cotacao - oficial) / oficial > 0.01:
        return (f"Cotação rejeitada: {cotacao:.4f} difere da cotação oficial atual ({oficial:.4f}). "
                f"Chame cotacao_usd_brl e repita o cálculo com o valor retornado.")
    bruto = valor_usd * cotacao
    final = bruto * (1 + SPREAD_CAMBIAL)
    return (f"US$ {valor_usd:.2f} x cotação {cotacao:.4f} = R$ {bruto:.2f}; "
            f"com spread de {SPREAD_CAMBIAL * 100:.0f}% = R$ {final:.2f}")


@tool
def calcular_frete(uf: str, subtotal_brl: float) -> str:
    """Calcula o frete em reais a partir da UF de destino (duas letras) e do subtotal dos livros em reais.
    Aplica a tarifa da região e a isenção para subtotal igual ou superior a R$ 250,00."""
    sigla = uf.strip().upper()
    regiao = UF_REGIAO.get(sigla)
    if regiao is None:
        return f"UF inválida: '{uf}'. Informe a sigla de duas letras, como RJ."
    tarifa = TARIFA_FRETE_BRL[regiao]
    if subtotal_brl >= LIMITE_FRETE_GRATIS_BRL:
        return (f"Região {regiao}; subtotal R$ {subtotal_brl:.2f} atinge o mínimo de R$ {LIMITE_FRETE_GRATIS_BRL:.2f}: "
                f"frete grátis (tarifa base R$ {tarifa:.2f}).")
    return (f"Região {regiao}; subtotal R$ {subtotal_brl:.2f} abaixo de R$ {LIMITE_FRETE_GRATIS_BRL:.2f}: "
            f"frete R$ {tarifa:.2f}. Faltam R$ {LIMITE_FRETE_GRATIS_BRL - subtotal_brl:.2f} para o frete grátis.")


PROMPT_RECOMENDACAO = ChatPromptTemplate.from_messages([
    ("system",
     "Você é consultor de leitura técnica da BookBridge. Recomende de 1 a 3 livros SOMENTE do catálogo abaixo, "
     "priorizando os que têm estoque, e justifique cada um em uma frase ligada ao perfil. Cite o preço em USD.\n\nCatálogo:\n{catalogo}"),
    ("human", "{perfil}"),
])


@tool
def recomendar_leitura(perfil_e_objetivo: str) -> str:
    """Recomenda livros do catálogo a partir do perfil e do objetivo de estudo do cliente, em linguagem natural.
    Ferramenta semântica: o resultado depende da interpretação do modelo sobre o catálogo."""
    linhas = executar_sql_somente_leitura("SELECT titulo, autor, categoria, preco_usd, estoque FROM livros", limite=50)
    catalogo = "\n".join(f"- {l['titulo']} ({l['autor']}; {l['categoria']}; US$ {l['preco_usd']:.2f}; estoque {l['estoque']})" for l in linhas)
    return (PROMPT_RECOMENDACAO | llm | StrOutputParser()).invoke({"catalogo": catalogo, "perfil": perfil_e_objetivo})


PEDIDOS_SERVICO: List[Dict[str, Any]] = []
ULTIMOS_PEDIDOS: Dict[Any, Any] = {}
JANELA_DUPLICIDADE_S = 600


class PedidosHandler(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass

    def _responder(self, codigo: int, corpo: Dict[str, Any]) -> None:
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_POST(self):
        if self.path != "/pedidos":
            return self._responder(404, {"erro": "rota inexistente"})
        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            carga = json.loads(self.rfile.read(tamanho) or b"{}")
            cliente, livro_id, quantidade = str(carga["cliente"]).strip(), int(carga["livro_id"]), int(carga["quantidade"])
        except (KeyError, ValueError, TypeError):
            return self._responder(400, {"erro": "payload inválido: cliente, livro_id e quantidade são obrigatórios"})
        if not cliente or quantidade <= 0:
            return self._responder(400, {"erro": "cliente vazio ou quantidade inválida"})
        con = conexao_somente_leitura()
        livro = con.execute("SELECT titulo, preco_usd, estoque FROM livros WHERE id = ?", (livro_id,)).fetchone()
        con.close()
        if livro is None:
            return self._responder(404, {"erro": f"livro {livro_id} não encontrado"})
        reservado = sum(p["quantidade"] for p in PEDIDOS_SERVICO if p["livro_id"] == livro_id)
        if quantidade > livro["estoque"] - reservado:
            return self._responder(409, {"erro": "estoque insuficiente", "estoque_disponivel": livro["estoque"] - reservado})
        pedido = {"id": 2000 + len(PEDIDOS_SERVICO) + 1, "cliente": cliente, "livro_id": livro_id, "titulo": livro["titulo"],
                  "quantidade": quantidade, "total_usd": round(livro["preco_usd"] * quantidade, 2), "status": "recebido"}
        PEDIDOS_SERVICO.append(pedido)
        self._responder(201, pedido)

    def do_GET(self):
        partes = self.path.strip("/").split("/")
        if len(partes) == 2 and partes[0] == "pedidos" and partes[1].isdigit():
            achado = next((p for p in PEDIDOS_SERVICO if p["id"] == int(partes[1])), None)
            return self._responder(200, achado) if achado else self._responder(404, {"erro": "pedido não encontrado"})
        self._responder(404, {"erro": "rota inexistente"})


def iniciar_servico_pedidos() -> str:
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), PedidosHandler)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{servidor.server_address[1]}"


if "URL_SERVICO_PEDIDOS" not in globals():
    URL_SERVICO_PEDIDOS = iniciar_servico_pedidos()


@tool
def registrar_pedido(cliente: str, titulo_exato: str, quantidade: int, confirmado_pelo_cliente: bool = False) -> str:
    """Registra um pedido no microsserviço interno de pedidos da BookBridge (HTTP POST /pedidos).
    Use confirmado_pelo_cliente=true somente quando o cliente, em mensagem posterior ao resumo de itens, quantidade, total e frete,
    confirmou explicitamente. Informe o título exato do catálogo."""
    if not confirmado_pelo_cliente:
        return ("Pedido não registrado: falta a confirmação explícita do cliente. Resuma itens, quantidade, total e frete "
                "e peça a confirmação antes de registrar.")
    chave_pedido = (cliente.strip().lower(), titulo_exato.strip().lower(), quantidade)
    anterior = ULTIMOS_PEDIDOS.get(chave_pedido)
    if anterior and time.time() - anterior[0] < JANELA_DUPLICIDADE_S:
        return f"Pedido idêntico já registrado (número {anterior[1]}); não registre novamente."
    achados = executar_sql_somente_leitura(
        f"SELECT id FROM livros WHERE LOWER(titulo) = '{titulo_exato.strip().lower().replace(chr(39), '')}'", limite=1)
    if not achados:
        return f"Título '{titulo_exato}' não encontrado no catálogo; consulte o título exato antes de registrar."
    try:
        r = httpx.post(f"{URL_SERVICO_PEDIDOS}/pedidos", timeout=8.0,
                       json={"cliente": cliente, "livro_id": achados[0]["id"], "quantidade": quantidade})
    except httpx.HTTPError as erro:
        return f"Serviço de pedidos indisponível: {erro}"
    corpo = r.json()
    if r.status_code == 201:
        ULTIMOS_PEDIDOS[chave_pedido] = (time.time(), corpo["id"])
        return (f"Pedido {corpo['id']} registrado (HTTP 201): {corpo['quantidade']}x {corpo['titulo']}, "
                f"total US$ {corpo['total_usd']:.2f}, status {corpo['status']}.")
    return f"Pedido recusado (HTTP {r.status_code}): {corpo.get('erro')}"


def handler_estimar_prazo(evento: Dict[str, Any], contexto: Any = None) -> Dict[str, Any]:
    uf = str(evento.get("uf", "")).strip().upper()
    regiao = UF_REGIAO.get(uf)
    if regiao is None:
        return {"statusCode": 400, "body": json.dumps({"erro": f"UF inválida: {uf}"}, ensure_ascii=False)}
    em_estoque = bool(evento.get("em_estoque", True))
    faixa = (10, 15) if not em_estoque else ((2, 5) if regiao == "Sudeste" else (4, 8))
    corpo = {"uf": uf, "regiao": regiao, "em_estoque": em_estoque, "prazo_min_dias_uteis": faixa[0], "prazo_max_dias_uteis": faixa[1]}
    return {"statusCode": 200, "body": json.dumps(corpo, ensure_ascii=False)}


REGISTRO_FUNCOES_SERVERLESS = {"estimar-prazo-entrega": handler_estimar_prazo}


def invocar_funcao(nome: str, evento: Dict[str, Any]) -> Dict[str, Any]:
    return REGISTRO_FUNCOES_SERVERLESS[nome](evento, {"request_id": str(uuid.uuid4())})


@tool
def estimar_prazo_entrega(uf: str, em_estoque: bool = True) -> str:
    """Estima o prazo de entrega em dias úteis chamando a função serverless 'estimar-prazo-entrega'.
    Informe a UF de destino e se o livro tem estoque nacional (estoque maior que zero); sem estoque o prazo é de importação sob encomenda."""
    resposta = invocar_funcao("estimar-prazo-entrega", {"uf": uf, "em_estoque": em_estoque})
    corpo = json.loads(resposta["body"])
    if resposta["statusCode"] != 200:
        return f"Erro {resposta['statusCode']}: {corpo['erro']}"
    return (f"Prazo para {corpo['uf']} ({corpo['regiao']}), {'com' if corpo['em_estoque'] else 'sem'} estoque nacional: "
            f"{corpo['prazo_min_dias_uteis']} a {corpo['prazo_max_dias_uteis']} dias úteis.")

# --- rag_estrategias ---
PROMPT_MULTIPLAS_CONSULTAS = ChatPromptTemplate.from_messages([
    ("system", "Reescreva a pergunta do cliente de 3 formas diferentes, usando o vocabulário formal de políticas comerciais de uma "
               "livraria (arrependimento, estorno, logística reversa, tributos, região, parcelas, assistente virtual, prazo). "
               "Responda uma reformulação por linha, sem numeração nem comentários."),
    ("human", "{pergunta}"),
])

PROMPT_DOCUMENTO_HIPOTETICO = ChatPromptTemplate.from_messages([
    ("system", "Escreva um parágrafo curto, no estilo de uma política formal de uma livraria online, que responderia à pergunta do "
               "cliente. Use o vocabulário de políticas comerciais. Não precisa ser fiel a números reais."),
    ("human", "{pergunta}"),
])


def fusao_por_posicao(listas: List[list], k: int):
    pontos: Dict[str, float] = {}
    melhor: Dict[str, Any] = {}
    for lista in listas:
        for posicao, (doc, score) in enumerate(lista, 1):
            cid = doc.metadata["chunk_id"]
            pontos[cid] = pontos.get(cid, 0.0) + 1.0 / (60 + posicao)
            if cid not in melhor or score < melhor[cid][1]:
                melhor[cid] = (doc, score)
    ordenados = sorted(pontos, key=lambda c: pontos[c], reverse=True)[:k]
    return [melhor[c] for c in ordenados]


def recuperar_multiplas_consultas(consulta: str, k: int = K_RECUPERACAO):
    texto = (PROMPT_MULTIPLAS_CONSULTAS | llm | StrOutputParser()).invoke({"pergunta": consulta})
    variantes = [consulta] + [l.strip() for l in texto.splitlines() if l.strip()][:3]
    return fusao_por_posicao([INDICE.similarity_search_with_score(v, k=max(k, 5)) for v in variantes], k)


def recuperar_documento_hipotetico(consulta: str, k: int = K_RECUPERACAO):
    hipotetico = (PROMPT_DOCUMENTO_HIPOTETICO | llm | StrOutputParser()).invoke({"pergunta": consulta})
    return fusao_por_posicao([INDICE.similarity_search_with_score(hipotetico, k=max(k, 5)),
                              INDICE.similarity_search_with_score(consulta, k=max(k, 5))], k)


ESTRATEGIAS["multiplas_consultas"] = recuperar_multiplas_consultas
ESTRATEGIAS["documento_hipotetico"] = recuperar_documento_hipotetico

# --- verificacao ---
class Afirmacao(BaseModel):
    texto: str = Field(description="Afirmação factual atômica extraída da resposta")
    chave: Optional[str] = Field(default=None, description="Chave canônica da lista fornecida, se a afirmação for numérica e mapeável; senão null")
    valor: Optional[float] = Field(default=None, description="Valor numérico afirmado, na unidade da chave")


class ListaAfirmacoes(BaseModel):
    afirmacoes: List[Afirmacao]


class VereditoAfirmacao(BaseModel):
    afirmacao: str
    veredito: Literal["APOIA", "REFUTA", "NEUTRA"]
    justificativa: str = Field(description="Uma frase, citando o trecho da evidência")


class RelatorioNLI(BaseModel):
    vereditos: List[VereditoAfirmacao]


class CoerenciaLogica(BaseModel):
    coerente: bool
    falacias: List[str] = Field(description="Falhas detectadas: 'Contradição premissa-conclusão' ou 'Erro de cálculo'; lista vazia se nenhuma")
    justificativa: str


def construir_verdade() -> Dict[str, Dict[str, float]]:
    verdade: Dict[str, Dict[str, float]] = {}
    for l in executar_sql_somente_leitura("SELECT titulo, preco_usd, estoque FROM livros", limite=100):
        titulo = l["titulo"].lower()
        verdade[f"preco_usd|{titulo}"] = {"valor": float(l["preco_usd"]), "tol": 0.01}
        verdade[f"estoque|{titulo}"] = {"valor": float(l["estoque"]), "tol": 0.0}
    for regiao, tarifa in TARIFA_FRETE_BRL.items():
        verdade[f"frete_brl|{regiao.lower()}"] = {"valor": tarifa, "tol": 0.0}
    verdade.update({
        "frete_gratis_minimo_brl": {"valor": LIMITE_FRETE_GRATIS_BRL, "tol": 0.0},
        "arrependimento_dias_corridos": {"valor": 7.0, "tol": 0.0},
        "desconto_pix_pct": {"valor": 5.0, "tol": 0.0},
        "parcelas_max_sem_juros": {"valor": 6.0, "tol": 0.0},
        "spread_cambial_pct": {"valor": SPREAD_CAMBIAL * 100, "tol": 0.0},
        "pontos_validade_meses": {"valor": 12.0, "tol": 0.0},
        "estorno_prazo_dias_uteis": {"valor": 5.0, "tol": 0.0},
        "avaria_prazo_dias_corridos": {"valor": 30.0, "tol": 0.0},
    })
    return verdade


PROMPT_EXTRACAO = ChatPromptTemplate.from_messages([
    ("system", "Extraia da resposta de um atendente as afirmações factuais atômicas: preços, estoques, prazos, tarifas, regras e valores "
               "calculados. Ignore saudações, perguntas e opiniões. Para cada afirmação numérica que corresponda a uma destas chaves "
               "canônicas, informe a chave e o valor afirmado (em reais, dólares, dias, unidades ou percentual, conforme a chave):\n{chaves}\n"
               "Associe uma chave somente quando o assunto da afirmação for exatamente o da chave (por exemplo, prazo de arrependimento, "
               "prazo de estorno e prazo para avaria são chaves diferentes). Não mapeie quantidades pedidas, subtotais, totais ou valores calculados "
               "na conversa: use chave apenas para o preço unitário em dólares, o estoque disponível, tarifas e regras de política. "
               "A chave frete_gratis_minimo_brl é somente o subtotal mínimo para isenção de frete, nunca o valor do frete cobrado nem a "
               "expressão 'frete grátis' em si. Informe o valor somente se o número aparecer literalmente na afirmação, nunca como padrão. "
               "Na dúvida, use null na chave."),
    ("human", "{resposta}"),
])

PROMPT_NLI = ChatPromptTemplate.from_messages([
    ("system", "Você é um auditor de inferência de linguagem natural. Para cada afirmação, compare com a EVIDÊNCIA, que são os retornos "
               "reais das ferramentas nesta execução, e classifique: APOIA se a evidência sustenta a afirmação, REFUTA se a evidência a "
               "contradiz, NEUTRA se a evidência não permite concluir. Paráfrases e resumos fiéis ao que a evidência diz são APOIA; "
               "use REFUTA somente quando a evidência afirma algo incompatível com a afirmação. Use somente a evidência; "
               "conhecimento externo não vale.\n\n"
               "EVIDÊNCIA:\n{evidencia}"),
    ("human", "Afirmações:\n{afirmacoes}"),
])

PROMPT_COERENCIA = ChatPromptTemplate.from_messages([
    ("system", "Você audita a coerência lógica da resposta de um atendente. Verifique apenas: (1) se a conclusão decorre das premissas "
               "presentes na evidência e (2) se há contradição entre partes da própria resposta. Os valores monetários calculados vêm de "
               "ferramentas determinísticas cujos resultados estão na evidência: não refaça nem questione essas contas. Afirmações que a "
               "evidência não cobre não são incoerência (isso é avaliado em outra etapa). Omitir detalhes, resumir ou reformular também não é "
               "incoerência. Marque incoerente somente diante de contradição real.\n\nEVIDÊNCIA:\n{evidencia}"),
    ("human", "Resposta a auditar:\n{resposta}"),
])


def evidencia_dos_passos(passos) -> str:
    if not passos:
        return "(nenhuma ferramenta foi chamada nesta execução)"
    return "\n\n".join(f"[{acao.tool}({json.dumps(acao.tool_input, ensure_ascii=False)})]\n{observacao}" for acao, observacao in passos)


def numeros_na_evidencia(texto: str) -> List[float]:
    achados = []
    for m in re.findall(r"\d+(?:[.,]\d+)?", texto):
        try:
            achados.append(float(m.replace(",", ".")))
        except ValueError:
            pass
    return achados


def checar_deterministico(afirmacoes: List[Afirmacao], verdade: Dict[str, Dict[str, float]], evidencia: str = "") -> List[Dict[str, Any]]:
    resultado = []
    numeros = numeros_na_evidencia(evidencia)
    for a in afirmacoes:
        if a.chave in verdade and a.valor is not None and any(abs(a.valor - n) <= 0.005 for n in numeros_na_evidencia(a.texto)):
            oficial = verdade[a.chave]
            delta = abs(a.valor - oficial["valor"])
            if delta == 0:
                status = "CONFIRMADO"
            elif delta <= oficial["tol"]:
                status = "CONFIRMADO (tolerância)"
            elif any(abs(a.valor - n) <= oficial["tol"] + 0.005 for n in numeros):
                status = "DIVERGENTE (valor consta na evidência)"
            else:
                status = "CONTRADITÓRIO"
            resultado.append({"afirmacao": a.texto, "chave": a.chave, "alegado": a.valor, "oficial": oficial["valor"], "status": status})
        else:
            resultado.append({"afirmacao": a.texto, "chave": a.chave, "alegado": a.valor, "oficial": None, "status": "NÃO MAPEADO"})
    return resultado


def invocar_estruturado(cadeia, entrada, tentativas: int = 3):
    ultimo = None
    for _ in range(tentativas):
        try:
            return cadeia.invoke(entrada)
        except Exception as erro:
            ultimo = erro
    raise ultimo


def verificar_resposta(resposta: str, evidencia: str) -> Dict[str, Any]:
    verdade = construir_verdade()
    extracao = invocar_estruturado(PROMPT_EXTRACAO | llm.with_structured_output(ListaAfirmacoes, method="function_calling"),
        {"chaves": "\n".join(sorted(verdade)), "resposta": resposta})
    afirmacoes = extracao.afirmacoes
    deterministica = checar_deterministico(afirmacoes, verdade, evidencia)

    nli: List[Dict[str, Any]] = []
    if afirmacoes:
        relatorio = invocar_estruturado(PROMPT_NLI | llm.with_structured_output(RelatorioNLI, method="function_calling"),
            {"evidencia": evidencia, "afirmacoes": "\n".join(f"- {a.texto}" for a in afirmacoes)})
        nli = [v.model_dump() for v in relatorio.vereditos]

    coerencia = invocar_estruturado(PROMPT_COERENCIA | llm.with_structured_output(CoerenciaLogica, method="function_calling"),
        {"evidencia": evidencia, "resposta": resposta}).model_dump()

    motivos = []
    alertas = []
    motivos += [f"REFUTA: {v['afirmacao']} ({v['justificativa']})" for v in nli if v["veredito"] == "REFUTA"]
    veredito_nli = {v["afirmacao"]: v["veredito"] for v in nli}
    for posicao, d in enumerate(deterministica):
        if d["status"] != "CONTRADITÓRIO":
            continue
        veredito = veredito_nli.get(d["afirmacao"])
        if veredito is None and len(nli) == len(deterministica):
            veredito = nli[posicao]["veredito"]
        texto = f"{d['afirmacao']} (alegado {d['alegado']}, oficial {d['oficial']})"
        chave_catalogo = d["chave"].startswith(("preco_usd|", "estoque|"))
        if veredito == "REFUTA":
            continue  # já registrado como motivo pela camada NLI
        if veredito == "APOIA" or chave_catalogo:
            alertas.append(f"CONFLITO com a chave oficial sem refutação pela evidência (provável erro de mapeamento): {texto}")
        else:
            motivos.append(f"CONTRADITÓRIO com a fonte oficial: {texto}")
    if not coerencia["coerente"]:
        alertas.append(f"INCOERENTE: {', '.join(coerencia['falacias'])} ({coerencia['justificativa']})")

    apoiadas = sum(v["veredito"] == "APOIA" for v in nli)
    return {
        "aprovada": not motivos,
        "taxa_sustentacao": (apoiadas / len(nli)) if nli else None,
        "afirmacoes": len(afirmacoes),
        "nli": nli,
        "deterministica": deterministica,
        "coerencia": coerencia,
        "motivos": motivos,
        "alertas": alertas,
    }

# --- tools_lista ---
FERRAMENTAS_INTEGRADOR = [
    cotacao_usd_brl, consultar_cep, calcular_preco_em_brl, calcular_frete, consultar_dados_sql, buscar_politicas,
    estimar_prazo_entrega, registrar_pedido, recomendar_leitura, gravar_preferencia, consultar_preferencias,
]

# --- agente_final ---
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

JANELA_MENSAGENS = 12
STORE_SESSOES: Dict[str, InMemoryChatMessageHistory] = {}
MENSAGEM_SEGURA = ("Não consegui confirmar essa informação com segurança. Vou encaminhar sua dúvida ao atendimento humano, "
                   "que funciona de segunda a sexta, das 9h às 18h.")


def obter_historico(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in STORE_SESSOES:
        STORE_SESSOES[session_id] = InMemoryChatMessageHistory()
    return STORE_SESSOES[session_id]


def montar_agente_com_historico(ferramentas, prompt_sistema: str = None) -> RunnableWithMessageHistory:
    executor = construir_executor(ferramentas, prompt_sistema=prompt_sistema)
    janela = RunnablePassthrough.assign(chat_history=RunnableLambda(
        lambda x: trim_messages(x["chat_history"], max_tokens=JANELA_MENSAGENS, strategy="last", token_counter=len,
                                start_on="human", include_system=False) if x.get("chat_history") else []))
    return RunnableWithMessageHistory(
        janela | executor, obter_historico,
        input_messages_key="input", history_messages_key="chat_history", output_messages_key="output",
    )


class AgenteBookBridge:
    def __init__(self, executor_com_historico: RunnableWithMessageHistory, verificar: bool = True,
                 corrigir: bool = True, max_correcoes: int = 1):
        self.executor = executor_com_historico
        self.verificar = verificar
        self.corrigir = corrigir
        self.max_correcoes = max_correcoes

    def _executar(self, texto: str, sessao: str, cliente_id: str, callbacks):
        config = {"configurable": {"session_id": sessao}, "callbacks": callbacks or []}
        try:
            saida = self.executor.invoke({"input": texto, "cliente_id": cliente_id}, config=config)
        except Exception as erro:
            print(f"[aviso] falha na execução do agente ({type(erro).__name__}); resposta segura aplicada.")
            return MENSAGEM_SEGURA, []
        return saida["output"], saida.get("intermediate_steps", [])

    def responder(self, pergunta: str, sessao: str, cliente_id: str = "cliente_demo", callbacks=None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        historico = obter_historico(sessao)
        base = list(historico.messages)
        resposta, passos = self._executar(pergunta, sessao, cliente_id, callbacks)
        ferramentas = [acao.tool for acao, _ in passos]
        verificacao = verificar_resposta(resposta, evidencia_dos_passos(passos)) if self.verificar else None
        motivos_primeira = list(verificacao["motivos"]) if verificacao else []

        correcoes = 0
        while self.corrigir and verificacao and not verificacao["aprovada"] and correcoes < self.max_correcoes:
            correcoes += 1
            historico.clear()
            for m in base:
                historico.add_message(m)
            instrucao = (f"{pergunta}\n\n[INSTRUÇÃO DO VERIFICADOR INTERNO] Sua resposta anterior foi reprovada: "
                         f"{'; '.join(verificacao['motivos'])}. Consulte as ferramentas necessárias e responda de novo, "
                         f"afirmando apenas o que as ferramentas devolverem.")
            resposta, passos = self._executar(instrucao, sessao, cliente_id, callbacks)
            ferramentas += [acao.tool for acao, _ in passos]
            verificacao = verificar_resposta(resposta, evidencia_dos_passos(passos))

        resposta_original = resposta
        bloqueada = bool(self.corrigir and verificacao and not verificacao["aprovada"])
        if bloqueada:
            resposta = MENSAGEM_SEGURA
        if correcoes or bloqueada:
            historico.clear()
            for m in base:
                historico.add_message(m)
            historico.add_user_message(pergunta)
            historico.add_ai_message(resposta)

        return {"resposta": resposta, "resposta_original": resposta_original, "ferramentas": ferramentas,
                "verificacao": verificacao, "motivos_primeira": motivos_primeira, "evidencia": evidencia_dos_passos(passos), "correcoes": correcoes, "bloqueada": bloqueada,
                "latencia_s": round(time.perf_counter() - t0, 2)}


def inicializar() -> "AgenteBookBridge":
    global INDICE
    if llm is None:
        raise RuntimeError("OPENAI_API_KEY não encontrada: defina a variável de ambiente antes de iniciar o agente.")
    if not os.path.exists(DB_PATH):
        criar_banco()
    if os.path.isdir(PASTA_INDICE):
        INDICE = carregar_indice()
    else:
        INDICE, _ = construir_indice()
    if os.path.exists("config_recuperacao.json"):
        with open("config_recuperacao.json") as f:
            ESTRATEGIA_ATUAL["nome"] = json.load(f)["estrategia"]
    return AgenteBookBridge(montar_agente_com_historico(FERRAMENTAS_INTEGRADOR))


def imprimir_resposta(r: dict) -> None:
    v = r["verificacao"]
    print(f"\nBia: {r['resposta']}")
    print(f"\n  ferramentas: {r['ferramentas'] or '-'}  |  latência {r['latencia_s']} s  |  correções: {r['correcoes']}  |  "
          f"verificação: {'APROVADA' if v['aprovada'] else 'REPROVADA'}")
    if r["motivos_primeira"]:
        print("  [verificador] 1ª resposta reprovada por:", r["motivos_primeira"])
    if not v["aprovada"]:
        print("  [verificador] motivos da reprovação final:", v["motivos"])


def chat_cli(pergunta_unica: str = None) -> None:
    agente = inicializar()
    sessao = "cli-" + uuid.uuid4().hex[:6]
    cliente = os.environ.get("BOOKBRIDGE_CLIENTE", "cliente_demo")
    if pergunta_unica:
        imprimir_resposta(agente.responder(pergunta_unica, sessao, cliente, callbacks=[RegistroExecucao(verboso=True)]))
        return
    print("BookBridge · atendimento (digite 'sair' para encerrar). Sessão:", sessao)
    while True:
        try:
            pergunta = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if pergunta:
            imprimir_resposta(agente.responder(pergunta, sessao, cliente, callbacks=[RegistroExecucao(verboso=True)]))


if __name__ == "__main__":
    import argparse
    analisador = argparse.ArgumentParser(description="Agente de atendimento BookBridge")
    analisador.add_argument("--pergunta", help="executa uma única pergunta e encerra")
    chat_cli(analisador.parse_args().pergunta)
