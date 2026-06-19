import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field
from fastapi import HTTPException
from interview_rag import interview_chain

from backend.sessions import (
    add_message,
    clear_session,
    create_session,
    get_history,
    get_session_role,
    set_session_role,
)


load_dotenv()

router = APIRouter(prefix="/interview", tags=["interview"])

class InterviewStreamRequest(BaseModel):
    """면접 코치 `/interview/stream` 엔드포인트가 받는 요청 모델입니다."""

    question: str = Field(
        ...,
        min_length=1,
        description="면접관이 제시한 질문입니다.",
        examples=["자기소개를 해 주세요."]
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="지원자가 입력한 답변입니다.",
        examples=["안녕하세요, 저는 ..."]
    )
    role: str = Field(
        default="general",
        description="면접관 유형입니다. general · technical · hr 중 하나를 사용합니다.",
        examples=["technical"]
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
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured",
        )

    return AsyncOpenAI(api_key=api_key)

class InterviewRagRequest(BaseModel):
    question: str

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

    # 토큰 사용량 추적
    if last_chunk and last_chunk.usage and request.session_id:
        # from backend.usage import record_usage
        # record_usage(request.session_id, last_chunk.usage)
        pass

    # 세션 이력 저장: 이번 턴의 user/assistant 메시지를 다음 요청에서 사용할 수 있도록 추가
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
    """
    세션 ID 로 면접 이력을 조회합니다.

    힌트:
    - try: get_history(session_id) 로 이력을 꺼낸다.
    - except KeyError: raise HTTPException(status_code=404, detail="session not found")
    - HistoryResponse(...) 를 반환한다.
    """
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
        raise HTTPException(status_code=400, detail=f"허용되지 않은 role: {body.role}. 허용값: {ALLOWED_ROLES}")
    
    try:
        set_session_role(session_id, body.role)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    
    return RoleUpdateResponse(
        session_id=session_id,
        role=body.role,
        message=f"면접관 유형이 {body.role} 로 변경되었습니다.",
    )
    
@router.post("/rag") 
async def interview_rag_endpoint(req: InterviewRagRequest):
    # ainvoke — answer/sources 한 응답, sources는 이미 dict라 그대로 직렬화
    return await interview_chain.ainvoke({"question": req.question})

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
