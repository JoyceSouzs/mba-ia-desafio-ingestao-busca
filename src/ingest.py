import os
import time
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Variável de ambiente {k} não definida")

PDF_PATH = os.getenv("PDF_PATH", "../document.pdf")


def load_and_split():
    """Carrega PDF e divide em chunks otimizados para RAG"""
    logger.info(f"Carregando PDF: {PDF_PATH}")
    documents = PyPDFLoader(PDF_PATH).load()
    logger.info(f"PDF carregado: {len(documents)} páginas")

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=False,
        separators=["\n\n", "\n", ".", " ", ""]  # Melhor separação
    ).split_documents(documents)

    # Filtrar chunks muito pequenos (menos de 50 caracteres)
    splits = [chunk for chunk in splits if len(chunk.page_content) >= 50]

    if not splits:
        raise SystemExit(0)

    logger.info(f"Documento dividido em {len(splits)} chunks")
    return splits


def ingest_pdf_with_batching(batch_size=10, delay=2):
    """
    Ingere PDF com batching para respeitar limites de cota da API

    Args:
        batch_size: Número de chunks a processar por vez (reduz cota)
        delay: Segundos de espera entre batches (respeita rate limiting)
    """
    documents = load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_MODEL"))

    logger.info(f"Iniciando ingestão com batch_size={batch_size}, delay={delay}s")

    try:
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PGVECTOR_COLLECTION"),
            connection=os.getenv("PGVECTOR_URL"),
            use_jsonb=True,
        )

        # Processar em lotes para evitar exceder cota
        total_chunks = len(documents)
        processed = 0

        for i in range(0, total_chunks, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            try:
                logger.info(f"Processando batch {batch_num}/{total_batches} "
                          f"({len(batch)} chunks, {processed}/{total_chunks} total)")

                vectorstore.add_documents(documents=batch)
                processed += len(batch)

                # Aguardar entre batches para respeitar rate limit
                if i + batch_size < total_chunks:
                    logger.info(f"Aguardando {delay}s antes do próximo batch...")
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Erro ao processar batch {batch_num}: {e}")
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.error("❌ Cota de API excedida! Processamento parado.")
                    logger.error(f"Chunks processados: {processed}/{total_chunks}")
                    raise
                raise

        logger.info(f"✅ Ingestão concluída! {processed} chunks armazenados "
                   f"na coleção '{os.getenv('PGVECTOR_COLLECTION')}'")

    except Exception as e:
        logger.error(f"Erro fatal durante ingestão: {e}")
        raise


if __name__ == "__main__":
    try:
        ingest_pdf_with_batching(batch_size=5, delay=3)  # 5 chunks por vez, 3s de espera
    except Exception as e:
        logger.error(f"Ingestão falhou: {e}")
        exit(1)
