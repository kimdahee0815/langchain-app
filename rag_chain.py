import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from rag_pipeline import format_sources, get_retriever

load_dotenv()

# ── 전이 계약: 다음 셀프(self2)는 아래 CUSTOMIZE 3곳의 '값만' 바꿔 도메인을 전이해요 ──
# CUSTOMIZE: domain PDF path — 사내 QA 트랙 값 (rag_pipeline.py가 색인한 문서와 동일)
DOC_PATHS = ["company_policy.pdf"]
# CUSTOMIZE: PERSIST_DIR — 트랙별 index 분리 (rag_pipeline.py가 재로드하는 폴더와 동일)
PERSIST_DIR = "./chroma_company_docs"
# CUSTOMIZE: system prompt — 도메인 정책의 단일 교체 지점 ({context}는 함수가 붙여요)
SYSTEM_PROMPT = "다음 근거 문서만 사용해 답하세요. 모르면 모른다고 답하세요."

retriever = get_retriever(k=3)
model = init_chat_model("openai:gpt-4o-mini")

question = "휴가 신청 절차는 어떻게 되나요?"  # 사내 QA 고정 질문

docs = retriever.invoke(question)
print(f"len(docs)={len(docs)}", docs[0].metadata)


def format_docs(docs) -> str:
    """LLM prompt에 넣을 context '문자열'만 만들어요 (sources와 역할 분리)."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, system_prompt):
    """retriever와 system prompt를 받아 {answer, sources} 체인을 돌려주는 공통 함수.

    다음 셀프(self2)가 이 함수를 import해 재사용하므로 시그니처를 바꾸지 않아요.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{context}"),
        ("human", "{question}"),
    ])

    answer_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),   # 같은 docs에서 context 파생
            "question": lambda x: x["question"],
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return (
        # [1단계] 검색은 여기서 '단 1회' — docs와 question을 함께 다음 단계로 복사해요.
        RunnableParallel(
            docs=lambda x: retriever.invoke(x["question"]),  # ← 이 파일의 retriever 호출은 이 줄 하나뿐
            question=lambda x: x["question"],
        )
        # [2단계] 동일 docs에서 answer와 sources를 '파생'해요 (retriever 재호출 금지).
        | RunnableParallel(
            answer=answer_chain,
            sources=lambda x: format_sources(x["docs"]),
        )
    )


rag_chain = build_rag_chain(retriever, SYSTEM_PROMPT)


def ask_rag(question: str) -> dict:
    """질문 1건 진입점 — {"answer": str, "sources": list[dict]} 계약 (self2가 그대로 호출해요)."""
    return rag_chain.invoke({"question": question})


if __name__ == "__main__":
    result = ask_rag(question)

    # hard exit 점검: answer와 sources 둘 다 non-empty여야만 완료예요.
    if not result["answer"].strip():
        sys.exit("[hard exit] answer가 비어 있어요 — prompt/context 연결부터 점검하세요.")
    if not result["sources"]:
        sys.exit("[hard exit] sources가 빈 list예요 — metadata/변환부터 점검하세요.")

    print(result)  # ← 이 출력이 응답 로그 = Day 4 비교 기준선이에요