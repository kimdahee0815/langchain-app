"""interview_rag_pipeline.py — 면접 코치 RAG 데이터 파이프라인

교안 핵심 패턴 (rag_pipeline.py 구조 재사용):
- get_embeddings(): 같은 좌표계 계약 — rag_pipeline.py와 동일 factory
- build_job_index(): Load → Split → Embed → Chroma 저장
    - chunk_size=500, chunk_overlap=50, add_start_index=True (기준 설정 3종)
    - persist_directory: ./chroma_job_docs (사내 QA와 분리)
- get_job_retriever(): Chroma 재로드 + as_retriever

파일 구조 규칙:
- ./chroma_job_docs: 면접 코치 직무 문서 트랙 전용 (혼합 금지)
- ./chroma_company_docs와 반드시 분리 — 같은 폴더에 누적하면 품질 오염
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import get_embeddings  # 같은 좌표계 계약 — factory는 1곳에서만

# ─── 상수 ────────────────────────────────────────────────────────────

DEFAULT_JOB_DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_docs", "job_posting.txt")
DEFAULT_JOB_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_job_docs")

# 기준 설정 3종 (교안 고정값 — 사내 QA 트랙과 동일)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_START_INDEX = True


# ─── Index 생성 ─────────────────────────────────────────────────────

def build_job_index(
    doc_path: str = DEFAULT_JOB_DOC_PATH,
    persist_dir: str = DEFAULT_JOB_PERSIST_DIR,
) -> Chroma:
    """Load → Split → Embed → Chroma 저장 (직무 문서 전용).

    교안 규칙:
    - persist_directory를 재사용하면 추가 누적됨 — 설정 변경 시 폴더 삭제 후 재구축
    - from_documents의 인자명은 embedding= (embedding_function= 아님)
    """
    # 1. Load
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()
    print(f"[interview_rag_pipeline] 직무 문서 로드: {len(docs)}건")

    # 2. Split (기준 설정 3종)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=ADD_START_INDEX,
    )
    chunks = splitter.split_documents(docs)
    print(f"[interview_rag_pipeline] chunk 분할: {len(chunks)}건")

    # 3. Embed + Store
    embeddings = get_embeddings()
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,  # 인자명 주의: from_documents는 embedding=
        persist_directory=persist_dir,
    )
    print(f"[interview_rag_pipeline] 직무 index 생성 완료: {persist_dir}")
    return db


# ─── Retriever factory ──────────────────────────────────────────────

def get_job_retriever(persist_dir: str = DEFAULT_JOB_PERSIST_DIR, k: int = 3):
    """Chroma 재로드 + as_retriever (직무 문서 전용).

    교안 규칙:
    - 재로드 시 인자명은 embedding_function= (from_documents의 embedding=과 다름!)
    - 검색 시에도 반드시 같은 get_embeddings() factory 사용
    """
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),  # 인자명 주의: 생성자는 embedding_function=
    )
    return db.as_retriever(search_kwargs={"k": k})


# ─── smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil

    # 기존 index 초기화 (재현 가능한 테스트)
    if os.path.exists(DEFAULT_JOB_PERSIST_DIR):
        shutil.rmtree(DEFAULT_JOB_PERSIST_DIR)
        print(f"[interview_rag_pipeline] 기존 직무 index 삭제: {DEFAULT_JOB_PERSIST_DIR}")

    # index 생성
    db = build_job_index()

    # k=2 smoke 검색
    query = "백엔드 엔지니어의 핵심 역량은?"
    results = db.similarity_search(query, k=2)
    print(f"\n[smoke] query: {query}")
    for doc in results:
        print(doc.page_content[:120])
        print(doc.metadata)
        print()

    # retriever smoke
    retriever = get_job_retriever(k=3)
    retriever_results = retriever.invoke("이 직무의 면접 프로세스는?")
    print(f"[retriever smoke] {len(retriever_results)}건 검색됨")
    for doc in retriever_results:
        print(f"  - {doc.page_content[:80]}...")
