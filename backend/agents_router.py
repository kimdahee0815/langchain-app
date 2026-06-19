"""
Day 3 self2 책임 메모
---------------------
- 이 파일이 담당하는 것:
    → 8주차 에이전트를 감싸고 SSE로 내보내는 백엔드 라우터
    (FastAPI 엔드포인트 정의, SSE 스트림 포맷 변환, 요청/응답 처리)

- 이 파일이 담당하지 않는 것:
    → 화면 표시, 사용자 입력 처리, AI 키 관리
    (Streamlit UI는 frontend가, OpenAI 키는 .env가, 에이전트 로직은 agents.py가 담당)

- Day 3 self1의 api_client.py와의 관계:
    → 프론트 api_client.py가 이 파일의 엔드포인트를 호출한다
    (api_client.py → POST /agents/stream → 이 파일 → agents.py)

- 8주차 파일 재사용 원칙:
    → roles.py, tools.py, agents.py는 재작성하지 않고 import만 한다.
    (기존 로직 변경 없이 라우터에서 감싸기만 함)
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agents import revise_agent, triage_agent 

router = APIRouter(prefix="/agents", tags=["agents"])

class InterviewAgentRequest(BaseModel):
    message: str
    mode: str = "single"

async def run_interview_agent_stream(message: str, mode: str):
    agent = triage_agent if mode == "multi" else revise_agent

    stream_result = Runner.run_streamed(agent, message)

    async for event in stream_result.stream_events():
        yield event
        
import json
from openai.types.responses import ResponseTextDeltaEvent
from agents import RunItemStreamEvent, Runner

async def iter_agent_events(agent_stream):
    async for event in agent_stream:
        # 1) 토큰 단위 텍스트 델타
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            payload = {"type": "token", "delta": event.data.delta}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 2) run_item 이벤트
        elif isinstance(event, RunItemStreamEvent):
            if "handoff" in event.name:
                payload = {"type": "status", "label": "handoff_detected", "detail": event.name}
            else:
                payload = {"type": "status", "label": "run_item", "detail": event.name}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

@router.post("/stream")
async def stream_interview_agent_endpoint(request: InterviewAgentRequest):
    agent_stream = run_interview_agent_stream(request.message, request.mode)
    return StreamingResponse(iter_agent_events(agent_stream), media_type="text/event-stream")