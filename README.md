# BookBridge · Agente cognitivo com LangChain

Agente de atendimento de uma livraria técnica. Reúne onze ferramentas (RAG sobre índice FAISS, text-to-SQL somente leitura, APIs públicas de câmbio e CEP, microsserviço HTTP de pedidos, função serverless de prazo, cálculos determinísticos e memória de longo prazo), memória de sessão com `RunnableWithMessageHistory` e uma camada de verificação (fonte oficial, NLI e coerência lógica).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `Agentes_Cognitivos_LangChain_BookBridge.ipynb` | Notebook completo, executável de ponta a ponta |
| `bookbridge_agente.py` | Agente empacotado (gerado pelo notebook), com interface de linha de comando |
| `app_streamlit.py` | Interface web |
| `requirements.txt` | Dependências (a comparação NLI opcional da seção 5.3 também exige `transformers` e `torch`) |

## Credenciais

Nenhuma chave vai no código. No Colab, cadastre `OPENAI_API_KEY` (e, opcionalmente, `LANGSMITH_API_KEY`) em **Secrets**. Fora do Colab, use variáveis de ambiente.

## Execução

```
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LANGSMITH_API_KEY=...        # opcional: rastreamento e avaliação no LangSmith
python bookbridge_agente.py         # conversa no terminal
python bookbridge_agente.py --pergunta "Quanto custa em reais o AI Engineering?"
streamlit run app_streamlit.py      # interface web
```

O primeiro uso cria `bookbridge.db` e o índice FAISS; execute o notebook antes para gerar `config_recuperacao.json` com a estratégia de recuperação escolhida pelos testes (sem ele, a busca simples é usada).

## Dados

Os dados (catálogo, pedidos e políticas) são fictícios e foram criados para o projeto. As informações de câmbio e CEP são obtidas por meio de APIs públicas, com contingência identificada na saída.
