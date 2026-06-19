"""rag_graph.py — LangGraph RAG 품질 루프 (Day 4 S5-S6)

교안 핵심 패턴:
- RagState(TypedDict) 6필드: question, docs, answer, quality, attempts, route_log
- node = state를 인자로 받고, 변경분 dict만 반환 (return docs / return state / return None 금지)
- router = state를 읽기만 하고 label 문자열만 반환 (state update 금지)
- compile = 배선 확인이지 품질 보증이 아님
- should_retry: Literal["retry", "generate"] — MAX_RETRIES cap으로 무한 루프 방지
- prepare_retry: attempts 증가 + route_log 4필드 누적 (새 list 반환, append 금지)
- route_log 4필드: attempts, quality_ok, decision, reason

배선 구조:
  START → retrieve → grade → [should_retry?]
                              ├─ "retry"    → prepare_retry → retrieve (최대 1회)
                              └─ "generate" → generate → END
"""

import os
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver 

from chains import build_rag_chain
from rag_pipeline import (
    DEFAULT_PERSIST_DIR,
    build_index,
    format_sources,
    get_retriever,
)

# ─── 상수 ────────────────────────────────────────────────────────────

MAX_RETRIES = 1  # 재검색은 전체 실행에서 1회만 허용
MIN_SOURCES = 2  # 최소 근거 문서 수 (grade 기준)


# ─── State Schema ────────────────────────────────────────────────────

class RagState(TypedDict):
    question: str               # 사용자의 질문 (입력)
    docs: list[Any]             # 검색된 Document 조각들
    answer: str                 # 최종 답변 (출력)
    quality: dict[str, Any]     # 평가 결과 — 통과 여부 + 이유
    attempts: int               # 재검색 시도 횟수
    route_log: list[dict]       # 분기 기록 — 4필드 dict 목록


# ─── Node 함수들 ─────────────────────────────────────────────────────

def retrieve(state: RagState) -> dict[str, Any]:
    """질문으로 문서를 검색한다. rag_pipeline.get_retriever() 실제 연동."""
    question = state["question"]

    # index가 없으면 자동 빌드
    if not os.path.exists(DEFAULT_PERSIST_DIR):
        build_index()

    retriever = get_retriever(k=3)
    docs: list[Document] = retriever.invoke(question)
    return {"docs": docs}


def grade(state: RagState) -> dict[str, Any]:
    """검색 결과를 평가한다. 근거 문서가 MIN_SOURCES건 미만이면 품질 미달."""
    docs = state.get("docs", [])
    passed = len(docs) >= MIN_SOURCES
    reason = "sources ok" if passed else "insufficient sources"
    return {"quality": {"passed": passed, "reason": reason}}


def generate(state: RagState) -> dict[str, Any]:
    """답변을 생성한다. build_rag_chain()으로 실제 LLM 호출."""
    question = state["question"]
    docs = state.get("docs", [])

    # context 구성: 검색된 chunk들의 page_content를 줄바꿈으로 결합
    context = "\n\n".join(
        doc.page_content if isinstance(doc, Document) else str(doc)
        for doc in docs
    )

    # LLM 답변 생성
    rag_chain = build_rag_chain()
    answer = rag_chain.invoke({"context": context, "question": question})

    # route_log에 generate 진입 기록
    quality_ok = state.get("quality", {}).get("passed", False)
    new_log = {
        "attempts": state.get("attempts", 0),
        "quality_ok": quality_ok,
        "decision": "generate",
        "reason": "quality ok" if quality_ok else "attempts cap reached",
    }

    return {
        "answer": answer,
        "route_log": [*state.get("route_log", []), new_log],
    }


# ─── Router ──────────────────────────────────────────────────────────

def should_retry(state: RagState) -> Literal["retry", "generate"]:
    """router: state를 읽기만 하고 label 문자열만 반환. state update 금지.

    교안 규칙:
    - 이 함수 안에 대입문이 한 줄도 없어야 함
    - 반환 타입 Literal 힌트는 graph 시각화를 위한 계약
    """
    quality_ok = state.get("quality", {}).get("passed", False)
    if not quality_ok and state.get("attempts", 0) < MAX_RETRIES:
        return "retry"
    return "generate"


# ─── 변경 전담 Node ──────────────────────────────────────────────────

def prepare_retry(state: RagState) -> dict[str, Any]:
    """state 변경은 node의 책임: attempts 증가 + route_log 4필드 기록.

    교안 규칙:
    - append 금지 — 기존 list를 펼쳐서 새 list로 반환해야 누적 보장
    - in-place 변경(state["route_log"].append(...))은 채널 갱신으로 인정되지 않음
    """
    new_log = {
        "attempts": state.get("attempts", 0) + 1,
        "quality_ok": state.get("quality", {}).get("passed", False),
        "decision": "retry",
        "reason": "insufficient sources",
    }
    return {
        "attempts": state.get("attempts", 0) + 1,
        "route_log": [*state.get("route_log", []), new_log],
    }


# ─── Graph 빌드 ─────────────────────────────────────────────────────

def build_rag_graph(checkpointer=None):
    """StateGraph 기반 RAG 품질 루프 graph를 빌드하고 compile하여 반환.

    배선:
      START → retrieve → grade → [should_retry?]
                                  ├─ "retry"    → prepare_retry → retrieve
                                  └─ "generate" → generate → END
    """
    builder = StateGraph(RagState)

    # Node 등록
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("prepare_retry", prepare_retry)

    # Edge 배선
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        should_retry,
        {"retry": "prepare_retry", "generate": "generate"},  # path_map: label → node
    )
    builder.add_edge("prepare_retry", "retrieve")  # 재검색 루프
    builder.add_edge("generate", END)               # 종료 경로 명시

    graph = builder.compile(checkpointer=checkpointer) 
    print("[rag_graph] compile ok:", type(graph).__name__)
    return graph


# ─── 초기 state factory ─────────────────────────────────────────────

def make_initial_state(question: str) -> RagState:
    """질문 문자열로 초기 RagState를 생성."""
    return {
        "question": question,
        "docs": [],
        "answer": "",
        "quality": {},
        "attempts": 0,
        "route_log": [],
    }

# ─── Checkpointer 연결 (모듈 수준 1회) ───────────────────────────────
# InMemorySaver는 개발/테스트용 — 프로세스 종료 시 저장 내용도 함께 사라짐.
# (실서비스용 Postgres/Redis/SQLite 영속 saver는 오늘 범위 밖, 이 주석으로만 기록)
checkpointer = InMemorySaver()          # ← 모듈 수준에서 '1회'만 생성 (함수 내부 금지)
app = build_rag_graph(checkpointer)     # ← checkpointer 연결된 graph

# ─── smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 단계 1: smoke — router 생존 ──
    print("router:", should_retry({"quality": {"passed": False}, "attempts": 0}),  # retry
        should_retry({"quality": {"passed": False}, "attempts": 1}))             # generate
    print("state keys:", list(RagState.__annotations__.keys()))
    print("-" * 55)

    # ── 단계 3: 같은 thread 2회 (qa:demo-1) — thread_id는 input이 아니라 config에! ──
    config_1 = {"configurable": {"thread_id": "qa:demo-1"}}

    r1 = app.invoke(make_initial_state("휴가 규정은?"), config_1)   # 첫 호출 = 전체 초기 state
    print(f"[qa:demo-1 #1] attempts={r1['attempts']} | route={[e['decision'] for e in r1['route_log']]}")
    print(f"[qa:demo-1 #1] answer  ={r1['answer'][:40]} ...")

    # 2회차는 '변경분(question)만' 넘김 — 나머지 state는 checkpoint에서 이어받음 (= 세션 유지)
    r2 = app.invoke({"question": "병가 규정은?"}, config_1)
    print(f"[qa:demo-1 #2] attempts={r2['attempts']} | route={[e['decision'] for e in r2['route_log']]}")
    print(f"[qa:demo-1 #2] answer  ={r2['answer'][:40]} ...")
    print("-" * 55)

    # ── 단계 4: 다른 thread 1회 (qa:demo-2) — 빈 이력에서 시작 ──
    config_2 = {"configurable": {"thread_id": "qa:demo-2"}}

    r3 = app.invoke(make_initial_state("휴가 규정은?"), config_2)
    print(f"[qa:demo-2 #1] attempts={r3['attempts']} | route={[e['decision'] for e in r3['route_log']]}")
    print(f"[qa:demo-2 #1] answer  ={r3['answer'][:40]} ...")
    print("-" * 55)

    # ── 단계 5: get_state snapshot 점검 (qa:demo-1 최신 checkpoint) ──
    snapshot = app.get_state(config_1)   # 반환은 dict가 아니라 StateSnapshot
    print("snapshot type:", type(snapshot).__name__)
    print("next =", snapshot.next)       # () = 끝까지 돈 상태
    for key in ("answer", "sources", "route_log", "attempts"):
        val = snapshot.values.get(key, "(state에 없음)")
        if key == "answer" and isinstance(val, str):
            val = val[:40] + " ..."
        print(f"  {key:9s} =", val)