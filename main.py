"""backend/app.py — FastAPI 서버 (Day 2 S5, Day 5 S4)

교안 핵심 패턴:
- thin route adapter: route 파일에 chain 조립 코드를 두지 않음
- mapping table: req.message → {"question": req.message} → result → {"reply": result}
- chain은 모듈 레벨 1회 생성 — 모든 요청이 같은 chain 공유 (stateless)
- async def endpoint 안에서는 await chain.ainvoke() (동기 invoke 금지 — event loop 블로킹)
- 422 = FastAPI 문 앞에서 막힘 (요청 검증), 500 = chain으로 가는 복도에서 넘어짐 (배선)

Endpoints:
  POST /chat           — 기본 채팅 (LCEL chain)
  POST /chat/structured — 구조화 출력 (InterviewScore)
  POST /chat/parallel   — 병렬+분기+Fallback chain
  POST /rag            — RAG QA (LangGraph 품질 루프, JSON 응답)
  GET  /rag/stream     — RAG QA (SSE 스트리밍)
"""

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# chain factory import — route 파일에는 조립 코드를 두지 않음
from chains import build_chat_chain, build_parallel_chain, build_structured_chain
from rag_graph import build_rag_graph, make_initial_state
from rag_pipeline import format_sources
from schemas import ChatRequest, ChatResponse, RagResponse, SourceItem

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import interview_router, stream_router
from backend import agents_router
from backend import feedback_router
from backend import resume_router

# ─── Chain/Graph holder (lifespan에서 1회 초기화) ────────────────────

_chains: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 chain/graph를 1회 생성 (교안: 모듈 레벨 1회 생성 원칙)."""
    _chains["chat"] = build_chat_chain()
    _chains["structured"] = build_structured_chain()
    _chains["parallel"] = build_parallel_chain()
    _chains["rag_graph"] = build_rag_graph()
    print("[backend] ✅ Chain/Graph 초기화 완료")
    yield
    _chains.clear()


# ─── App 설정 ────────────────────────────────────────────────────────

app = FastAPI(
    title="10주차 관통예제 — RAG 사내 문서 QA 챗봇",
    description="LangChain LCEL + LangGraph + Chroma RAG 통합 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router.router)
app.include_router(interview_router.router)
app.include_router(feedback_router.router)
app.include_router(resume_router.router)
app.include_router(stream_router.router)

# ─── Request / Response 모델 ─────────────────────────────────────────

class StructuredRequest(BaseModel):
    question: str = Field(description="면접 질문")
    answer: str = Field(description="면접 답변")


# ─── POST /chat — 기본 채팅 ──────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """기본 면접 코치 채팅.

    mapping table:
      req.message → {"question": req.message}
      ainvoke 결과 (str) → {"reply": result}
    """
    # INPUT KEY: question
    result = await _chains["chat"].ainvoke({"question": req.message})
    return {"reply": result}


# ─── POST /chat/structured — 구조화 출력 ─────────────────────────────

@app.post("/chat/structured")
async def chat_structured(req: StructuredRequest):
    """면접 답변 평가 (InterviewScore 구조화 출력).

    - with_structured_output → 결과가 Pydantic 인스턴스
    - .model_dump()로 JSON 직렬화
    """
    result = await _chains["structured"].ainvoke({
        "question": req.question,
        "answer": req.answer,
    })
    return result.model_dump()


# ─── POST /chat/parallel — 병렬+분기+Fallback ────────────────────────

@app.post("/chat/parallel")
async def chat_parallel(req: ChatRequest):
    """RunnableParallel + RunnableBranch + Fallback 통합 응답.

    결과: {"answer": str, "faq": str}
    """
    result = await _chains["parallel"].ainvoke({"question": req.message})
    return result


# ─── POST /rag — RAG QA (JSON 응답) ─────────────────────────────────

@app.post("/rag", response_model=RagResponse)
async def rag_qa(req: ChatRequest):
    """RAG 사내 문서 QA — LangGraph 품질 루프 기반.

    - StateGraph invoke → 전체 state 반환
    - answer + sources + quality_passed + attempts 추출
    """
    initial_state = make_initial_state(req.message)

    # LangGraph는 동기 invoke — asyncio.to_thread로 event loop 블로킹 방지
    final_state = await asyncio.to_thread(_chains["rag_graph"].invoke, initial_state)

    # source 포맷팅
    sources = format_sources(final_state.get("docs", []))
    source_items = [SourceItem(**s) for s in sources]

    return RagResponse(
        answer=final_state.get("answer", ""),
        sources=source_items,
        quality_passed=final_state.get("quality", {}).get("passed", False),
        attempts=final_state.get("attempts", 0),
    )


# ─── GET /rag/stream — RAG QA (SSE 스트리밍) ────────────────────────

@app.get("/rag/stream")
async def rag_stream(message: str = Query(description="사용자 질문")):
    """RAG QA 스트리밍 (Server-Sent Events).

    Day 2 S5 seam 주석에서 예고된 astream 패턴.
    SSE 프로토콜: 'data: {...}\\n\\n' 형식
    """

    async def event_generator():
        initial_state = make_initial_state(message)

        # graph invoke (동기 → thread)
        final_state = await asyncio.to_thread(_chains["rag_graph"].invoke, initial_state)

        answer = final_state.get("answer", "")
        sources = format_sources(final_state.get("docs", []))

        # 답변을 청크 단위로 스트리밍 (실제 astream 효과 시뮬레이션)
        words = answer.split()
        buffer = ""
        for i, word in enumerate(words):
            buffer += word + " "
            if (i + 1) % 5 == 0 or i == len(words) - 1:
                yield f"data: {json.dumps({'type': 'token', 'content': buffer.strip()}, ensure_ascii=False)}\n\n"
                buffer = ""
                await asyncio.sleep(0.05)

        # 출처 정보 전송
        yield f"data: {json.dumps({'type': 'sources', 'content': sources}, ensure_ascii=False)}\n\n"

        # 종료 신호
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── Health check ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "10주차 관통예제 RAG 챗봇"}
