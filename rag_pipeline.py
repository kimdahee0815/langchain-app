import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from embeddings_factory import get_embeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__),"chroma_company_docs")
DEFAULT_DOC_PATH = os.path.join(os.path.dirname(__file__), "job_posting.pdf")

# 기준 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_START_INDEX = True

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
    
# 인덱스 생성 : 처음 1번만 실행
def build_index(doc_path: str = DEFAULT_DOC_PATH, persist_dir: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Load -> Split -> Embed -> Chroma 저장"""
    # 1. Load
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()
    print(f"[rag_pipeline] 문서 로드: {len(docs)}")
    
    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=ADD_START_INDEX)
    chunks = splitter.split_documents(docs)
    print(f"[rag_pipline] chunk 분할: {len(chunks)}건")
    
    # 3. Embed + Store
    embeddings = get_embeddings()
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print(f"[rag_pipeline] index 생성 완료: {persist_dir}")
    return db