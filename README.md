# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema para ingestão de PDFs e busca semântica via CLI, utilizando LangChain, PostgreSQL/pgVector e Google Gemini.

## O que essa aplicação faz?

Esta aplicação permite você:

1. **Ingerir um PDF**: O documento é carregado, dividido em chunks de 1000 caracteres (com sobreposição de 150 caracteres) e convertido em vetores (embeddings) usando o modelo de embedding do Google Gemini.

2. **Armazenar vetores no banco**: Os embeddings são armazenados em um banco PostgreSQL com a extensão `pgvector`, criando um índice vetorial para busca semântica rápida.

3. **Fazer perguntas em linguagem natural**: Através de um chat CLI, você faz perguntas sobre o conteúdo do PDF.

4. **Obter respostas baseadas no contexto**: A aplicação:
   - Vetoriza sua pergunta
   - Busca os 10 chunks mais relevantes no banco
   - Usa esses chunks como contexto
   - Envia o contexto + pergunta para o Google Gemini
   - Retorna uma resposta que só usa informações do documento (não inventa)

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API do Google Gemini

## Configuração

### 1. Instale as dependências

```bash
pip3 install -r requirements.txt
```

### 2. Configure as variáveis de ambiente

Edite o arquivo `.env` e preencha com seus valores:

| Variável | Descrição | Exemplo |
|---|---|---|
| `GOOGLE_API_KEY` | Chave da API Google Gemini | `AIzaSy...` |
| `GOOGLE_MODEL` | Modelo de embedding | `models/embedding-001` |
| `GOOGLE_LLM_MODEL` | Modelo LLM do Gemini | `gemini-2.0-flash` |
| `PGVECTOR_URL` | URL de conexão PostgreSQL | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PGVECTOR_COLLECTION` | Nome da coleção vetorial | `my_collection` |
| `PDF_PATH` | Caminho para o PDF | `../document.pdf` |

### 3. Suba o banco de dados

```bash
docker compose up -d
```

O Docker Compose inicializa o PostgreSQL 17 com a extensão `pgvector` habilitada automaticamente. Aguarde cerca de 10 segundos até o container estar saudável.

## Execução

### Passo 1 — Ingestão do PDF

Execute **uma única vez** para carregar e vetorizar o documento:

```bash
cd src
python3 ingest.py
```

Saída esperada:
```
Ingeridos N chunks na coleção 'my_collection'.
```

### Passo 2 — Chat interativo

```bash
cd src
python3 chat.py
```

Digite sua pergunta e pressione Enter. Para encerrar, digite `sair`, `quit` ou pressione `Ctrl+C`.

Exemplo de uso:
```
Chat RAG iniciado. Digite 'sair' ou 'quit' para encerrar.

Você: Qual é o tema principal do documento?
Assistente: ...

Você: sair
Encerrando o chat.
```

## Arquitetura

```
document.pdf
    └─► ingest.py
            ├─ PyPDFLoader              (carrega páginas do PDF)
            ├─ RecursiveCharacterTextSplitter  (chunks: 1000 chars, overlap: 150)
            ├─ GoogleGenerativeAIEmbeddings    (vetoriza cada chunk)
            └─ PGVector.add_documents() ──► PostgreSQL + pgvector

pergunta do usuário
    └─► chat.py
            └─► search.py
                    ├─ GoogleGenerativeAIEmbeddings    (vetoriza a pergunta)
                    ├─ PGVector.similarity_search(k=10)  ◄── PostgreSQL
                    ├─ monta prompt com contexto recuperado
                    └─ ChatGoogleGenerativeAI.invoke()  ──► resposta
```

## Estrutura do Projeto

```
.
├── docker-compose.yml      # PostgreSQL 17 com pgvector
├── document.pdf            # PDF a ser ingerido
├── requirements.txt        # Dependências Python
├── .env.example            # Template de variáveis de ambiente
└── src/
    ├── ingest.py           # Carrega, divide e vetoriza o PDF
    ├── search.py           # Busca semântica + geração de resposta
    └── chat.py             # Interface CLI interativa
```
