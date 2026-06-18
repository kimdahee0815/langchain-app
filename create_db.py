from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

pdf_docs = PyPDFLoader("docs/sample.pdf").load()
txt_docs = TextLoader("docs/sample.txt",
    encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,   # 한 청크의 목표 크기 - 문자 수 기준, 토큰이 아니다.
        chunk_overlap=50,   # 인접 청크와 서로 겹치는 문자 수
        add_start_index=True  # 기본값 False - 명시해야 metadata에 start_index가 생긴다
    )

chunks = splitter.split_documents(pdf_docs)
print(f"chunks count: {len(chunks)}")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

doc_vectors = embeddings.embed_documents([c.page_content for c in chunks[:5]])

print(len(doc_vectors), len(doc_vectors[0])) # 문서 수, 벡터 차원 출력

def create_db(chunks):
    # 벡터DB가 저장될 위치
    PERSIST_DIR = "./chroma_company_docs"
    db = Chroma.from_documents(
        chunks,                         # 원문 + 메타데이터를 통째로
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db

def load_db():
    PERSIST_DIR = "./chroma_company_docs"
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db

# 벡터 DB 문서 정보를 저장
create_db(chunks=chunks)

# db = load_db()

# query = "휴가 신청 절차는 어떻게 되나요?"
# result = db.similarity_search(query=query, k=2) # k-2: 좌표가 가장 가까운 chunk 후보 2개 (유사도가 높은 순으로 정렬해서 상위 2개)

# for doc in result:
#     print(doc.page_content[:50])
#     print({
#         "source": doc.metadata.get("source"),
#         "page": doc.metadata.get("page"),
#         "start_index": doc.metadata.get("start_index")
#     })

# 랭체인 파이프라인에서는 이것을 사용.
# db.as_retriever()