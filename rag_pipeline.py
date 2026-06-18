from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def dedup_key(doc)-> tuple:
    """같은 chunk 판별 기준 : (source, page, start_index)"""
    m = doc.metadata
    return (m.get("source"), m.get("page"), m.get("start_index"))

def format_sources(docs):
    seen = set()
    results = []

    for doc in docs:
        key = dedup_key(doc)

        if key not in seen:
            seen.add(key)

            results.append({
                "source": key[0],
                "page": key[1],
                "start_index": key[2],
                "snippet": doc.page_content[:120]
            })

    return results


def load_db():
    PERSIST_DIR = "./chroma_company_docs"
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db


def get_retriever(k: int =3 ):
    """Chroma index를 RAG chain에 끼울 수 있는 retriever로 전환"""
    return load_db().as_retriever(search_kwargs={"k":k})
    