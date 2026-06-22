"""backend/interview_router.py — 면접 코치 통합 라우터

interview_app.py(FastAPI app)의 모든 route를 APIRouter 형식으로 통합한다.

통합 규칙:
- @app.post("/interview/X")  →  @router.post("/X")   (router prefix="/interview")
- lifespan의 _chains 초기화   →  모듈 레벨 1회 생성    (main.py에 lifespan이 없음)
- FastAPI / CORSMiddleware / app 관련 코드는 제거       (main.py가 담당)

Endpoints (실제 경로 = prefix "/interview" + 아래 path):
    POST  /stream                — 면접 코치 SSE (OpenAI 직접 + 세션 이력)
    POST  /session/create        — 세션 생성
    GET   /session/{id}/history  — 세션 이력 조회
    PATCH /session/{id}/role     — 면접관 유형 변경
    POST  /chat                  — 기본 채팅 (LCEL chain)
    POST  /structured            — 답변 구조화 평가 (InterviewScore)
    POST  /parallel              — 질문 생성 + 팁 (병렬)
    WPOST  /rag                   — 직무 RAG QA (LangGraph 품질 루프)
    POST  /rag/thread            — thread_id 대화 유지
    POST  /rag/eval              — RAG 기반 답변 평가
    GET   /rag/stream            — 직무 RAG 스트리밍 (SSE)
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai import APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field

# ── LangChain / Graph 레이어 (루트 레벨 모듈 → bare import) ──────────
from interview_chains import (
    build_interview_chat_chain,
    build_interview_parallel_chain,
    build_interview_rag_eval_chain,
    build_interview_structured_chain,
)
from interview_rag_graph import (
    build_interview_rag_graph,
    make_interview_initial_state,
    run_interview_graph,
)
from interview_schemas import (
    InterviewRagEvalRequest,
    InterviewRagRequest,
    InterviewRagResponse,
)
from rag_pipeline import format_sources
from schemas import ChatRequest, ChatResponse, SourceItem

# ── web 레이어 (backend 패키지) ─────────────────────────────────────
from backend.sessions import (
    add_message,
    create_session,
    get_history,
    get_session_role,
    set_session_role,
)

load_dotenv()

router = APIRouter(prefix="/interview", tags=["interview"])

# ─── Chain/Graph holder (lifespan → router 모듈 레벨 1회 생성) ───────
# main.py에 lifespan이 없으므로, interview_app.py의 lifespan이 만들던
# chain/graph를 router import 시점에 1회 생성한다 (교안: 모듈 레벨 1회 생성).
_chains: dict = {
    "interview_chat": build_interview_chat_chain(),
    "interview_structured": build_interview_structured_chain(),
    "interview_parallel": build_interview_parallel_chain(),
    "interview_rag_graph": build_interview_rag_graph(),
    "interview_rag_eval": build_interview_rag_eval_chain(),
}
print("[interview_router] ✅ 면접 코치 Chain/Graph 초기화 완료")


# ═══════════════════════════════════════════════════════════════════
#  여기서부터 기존 interview_router 내용 (SSE 피드백 + 세션 관리)
# ═══════════════════════════════════════════════════════════════════

class InterviewStreamRequest(BaseModel):
    """면접 코치 `/interview/stream` 엔드포인트가 받는 요청 모델입니다."""

    question: str = Field(
        ...,
        min_length=1,
        description="면접관이 제시한 질문입니다.",
        examples=["자기소개를 해 주세요."],
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="지원자가 입력한 답변입니다.",
        examples=["안녕하세요, 저는 ..."],
    )
    role: str = Field(
        default="general",
        description="면접관 유형입니다. general · technical · hr 중 하나를 사용합니다.",
        examples=["technical"],
    )
    session_id: str | None = Field(
        default=None,
        description="UUID 기반 면접 세션 ID입니다. self2에서 연결합니다.",
    )
    model: str = Field(default="gpt-5.4-nano", description="사용할 OpenAI 모델명입니다.")


def get_interview_openai_client() -> AsyncOpenAI:
    """환경 변수에서 OPENAI_API_KEY를 읽어 AsyncOpenAI 클라이언트를 만듭니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured",
        )
    return AsyncOpenAI(api_key=api_key)


ROLE_PROMPTS: dict[str, str] = {
    "general": "당신은 일반 면접관입니다. 지원자의 답변을 종합적으로 평가하고 개선점을 한국어로 피드백하세요.",
    "technical": "당신은 기술 면접관입니다. 지원자의 기술 역량과 문제 해결 방식을 집중 평가하고 한국어로 피드백하세요.",
    "hr": "당신은 인사 면접관입니다. 지원자의 인성, 협업 능력, 조직 적합성을 평가하고 한국어로 피드백하세요.",
}


async def interview_event_generator(
    request: InterviewStreamRequest,
) -> AsyncIterator[str]:
    """면접 코치 피드백을 SSE data 이벤트로 스트리밍합니다."""
    client = get_interview_openai_client()
    system_prompt = ROLE_PROMPTS.get(request.role, ROLE_PROMPTS["general"])

    # 세션 이력 연결
    history: list[dict] = []
    if request.session_id:
        try:
            history = get_history(request.session_id)
        except KeyError:
            pass  # 없는 세션이면 이력 없이 진행

    user_content = (
        f"[면접 질문]\n{request.question}\n\n"
        f"[지원자 답변]\n{request.answer}\n\n"
        "위 답변을 면접관 역할에 맞게 평가하고 개선 피드백을 제공해 주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_content},
    ]

    full_response_parts: list[str] = []
    last_chunk = None

    try:
        stream = await client.chat.completions.create(
            model=request.model,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
            messages=messages,
        )

        async for chunk in stream:
            last_chunk = chunk
            delta = chunk.choices[0].delta if chunk.choices else None
            token = (delta.content or "") if delta else ""

            if not token:
                continue

            full_response_parts.append(token)
            yield f"data: {token}\n\n"

    except RateLimitError:
        yield 'data: {"error": "rate_limited"}\n\n'
        yield "data: [DONE]\n\n"
        return
    except APIError as e:
        yield f'data: {{"error": "api_error", "detail": "{str(e)}"}}\n\n'
        yield "data: [DONE]\n\n"
        return

    # 토큰 사용량 추적 (TODO 3에서 연결)
    if last_chunk and last_chunk.usage and request.session_id:
        pass

    # 세션 이력 저장: 이번 턴의 user/assistant 메시지를 다음 요청에서 사용
    if request.session_id:
        full_response = "".join(full_response_parts)
        try:
            add_message(request.session_id, "user", user_content)
            add_message(request.session_id, "assistant", full_response)
        except KeyError:
            pass

    yield "data: [DONE]\n\n"


@router.post("/stream")
async def interview_stream(request: InterviewStreamRequest) -> StreamingResponse:
    return StreamingResponse(
        interview_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class SessionCreateResponse(BaseModel):
    session_id: str
    role: str


class SessionCreateRequest(BaseModel):
    role: str = Field(default="general", description="초기 면접관 유형")


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_interview_session(body: SessionCreateRequest) -> SessionCreateResponse:
    session_id = create_session(body.role)
    return SessionCreateResponse(session_id=session_id, role=body.role)


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]
    role: str
    message_count: int


@router.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_interview_history(session_id: str) -> HistoryResponse:
    """세션 ID로 면접 이력을 조회합니다."""
    try:
        messages = get_history(session_id)
        role = get_session_role(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
        role=role,
        message_count=len(messages),
    )


# 허용 면접관 유형
ALLOWED_ROLES = {"general", "technical", "hr"}


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., description="변경할 면접관 유형 (general · technical · hr)")


class RoleUpdateResponse(BaseModel):
    session_id: str
    role: str
    message: str


@router.patch("/session/{session_id}/role", response_model=RoleUpdateResponse)
async def update_interview_role(session_id: str, body: RoleUpdateRequest) -> RoleUpdateResponse:
    if body.role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"허용되지 않은 role: {body.role}. 허용값: {ALLOWED_ROLES}",
        )
    try:
        set_session_role(session_id, body.role)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    return RoleUpdateResponse(
        session_id=session_id,
        role=body.role,
        message=f"면접관 유형이 {body.role} 로 변경되었습니다.",
    )


# ═══════════════════════════════════════════════════════════════════
#  여기서부터 interview_app.py에서 통합된 route (router 형식으로 변환)
# ═══════════════════════════════════════════════════════════════════

# ─── POST /interview/chat — 면접 코치 기본 채팅 ─────────────────────

@router.post("/chat", response_model=ChatResponse)
async def interview_chat(req: ChatRequest):
    """면접 코치 기본 채팅 (직무 RAG 없이).

    mapping table:
      req.message → {"question": req.message}
      ainvoke 결과 (str) → {"reply": result}
    """
    result = await _chains["interview_chat"].ainvoke({"question": req.message})
    return {"reply": result}


# ─── POST /interview/structured — 면접 답변 구조화 평가 ──────────────

@router.post("/structured")
async def interview_structured(req: InterviewRagEvalRequest):
    """면접 답변 평가 (InterviewScore 구조화 출력)."""
    result = await _chains["interview_structured"].ainvoke({
        "question": req.question,
        "answer": req.answer,
    })
    return result.model_dump()


# ─── POST /interview/parallel — 면접 질문 + 팁 병렬 응답 ─────────────

@router.post("/parallel")
async def interview_parallel(req: ChatRequest):
    """면접 질문 생성 + 준비 팁 동시 응답.  결과: {"questions": str, "tips": str}"""
    result = await _chains["interview_parallel"].ainvoke({"question": req.message})
    return result


# ─── POST /interview/rag — 직무 RAG QA (LangGraph 품질 루프) ────────

@router.post("/rag", response_model=InterviewRagResponse)
async def interview_rag(req: InterviewRagRequest):
    """직무 문서 RAG 기반 면접 코칭 — LangGraph 품질 루프.

    - StateGraph invoke → 전체 state 반환
    - answer + sources(graph state) + quality_passed + attempts 추출
    """
    initial_state = make_interview_initial_state(req.question)

    # LangGraph는 동기 invoke — asyncio.to_thread로 event loop 블로킹 방지
    final_state = await asyncio.to_thread(
        _chains["interview_rag_graph"].invoke, initial_state
    )

    # sources는 generate 노드가 state에 직접 넣음 (Day 4 self2 패턴)
    sources = final_state.get("sources", [])
    source_items = [SourceItem(**s) if isinstance(s, dict) else s for s in sources]

    return InterviewRagResponse(
        answer=final_state.get("answer", ""),
        sources=source_items,
        quality_passed=final_state.get("quality", {}).get("passed", False),
        attempts=final_state.get("attempts", 0),
    )


# ─── POST /interview/rag/thread — thread 기반 대화 유지 ──────────────

@router.post("/rag/thread")
async def interview_rag_thread(
    req: InterviewRagRequest,
    thread_id: str = "interview:default:1",
):
    """thread_id 기반 면접 코칭 — run_interview_graph wrapper 사용.

    교안 규약 (Day 4 self2):
    - thread_id는 config["configurable"]에만 전달 (state에 넣기 금지)
    - InMemorySaver 기반 — 같은 thread_id로 재호출하면 맥락 유지
    """
    result = await asyncio.to_thread(run_interview_graph, req.question, thread_id)
    return result


# ─── POST /interview/rag/eval — 직무 RAG 기반 답변 평가 ─────────────

@router.post("/rag/eval")
async def interview_rag_eval(req: InterviewRagEvalRequest):
    """직무 문서 RAG context + 면접 답변 → 구조화 평가 (InterviewScore)."""
    from interview_rag_pipeline import get_job_retriever
    from langchain_core.documents import Document

    # 직무 문서에서 관련 chunk 검색
    retriever = get_job_retriever(k=3)
    docs = await asyncio.to_thread(retriever.invoke, req.question)

    # context 구성
    context = "\n\n".join(
        doc.page_content if isinstance(doc, Document) else str(doc)
        for doc in docs
    )

    # 구조화 평가
    result = await _chains["interview_rag_eval"].ainvoke({
        "context": context,
        "question": req.question,
        "answer": req.answer,
    })

    sources = format_sources(docs)
    return {**result.model_dump(), "sources": sources}


# ─── GET /interview/rag/stream — 직무 RAG 스트리밍 (SSE) ────────────

@router.get("/rag/stream")
async def interview_rag_stream(question: str = Query(description="면접 관련 질문")):
    """직무 문서 RAG 면접 코칭 스트리밍 (Server-Sent Events).

    SSE 프로토콜: 'data: {...}\\n\\n' 형식
    """

    async def event_generator():
        initial_state = make_interview_initial_state(question)

        # graph invoke (동기 → thread)
        final_state = await asyncio.to_thread(
            _chains["interview_rag_graph"].invoke, initial_state
        )

        answer = final_state.get("answer", "")
        sources = format_sources(final_state.get("docs", []))

        # 답변을 5단어 청크로 스트리밍
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
# TODO 1: UUID 세션 관리
# from interview_app.backend.sessions import create_session, add_message, get_history
# → day2-self2에서 연결. session_id 를 InterviewStreamRequest 에서 받아 get_history() 로 이전 이력을 꺼낸다.

# TODO 2: 예외 핸들러
# from interview_app.backend.errors import register_exception_handlers
# → backend/main.py 에서 register_exception_handlers(app) 로 등록한다.
# → RateLimitError → 429, APIError → 502 로 변환.

# TODO 3: 토큰 사용량 추적
# from interview_app.backend.usage import record_usage
# → stream 경로에서 usage 기록 시점이 제한될 수 있으므로 일반 /interview 엔드포인트에서 먼저 연결.

# TODO 4: 8주차 역할 프리셋 재사용
# from interview_app.core.roles import ROLE_PROMPTS  (8주차 roles.py 이미 있으면 import만)
# → 본 파일의 ROLE_PROMPTS 와 8주차 코드를 비교해 import 중심으로 재사용. 재작성 금지.
