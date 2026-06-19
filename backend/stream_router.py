import json
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field

load_dotenv()

class StreamRequest(BaseModel):
  """SSE 스트리밍 응답에 사용할 사용자 요청"""
  message: str = Field(..., min_length=1)
  model: str = "gpt-5.4-nano"
  temperature: float = Field(default=0.7, ge=0.0, le=2.0)

def get_openai_client() -> AsyncOpenAI:
  """환경 변수의 API 키로 AsyncOpenAI 클라이언트를 생성"""
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY in not configured")
  
  return AsyncOpenAI(api_key=api_key)

def sse_data(payload: dict[str, str]) -> str:
  """dict payload를 SSE data 이벤트 문자열로 변환"""
  body = json.dumps(payload, ensure_ascii=False)
  return f"data: {body}\n\n"

async def event_generator(request: StreamRequest) -> AsyncIterator[str]:
  """OpenAI stream 응답을 SSE data 이벤트로 변환"""
  try:
    client = get_openai_client()
    yield sse_data({"type": "status", "label": "일반 챗봇 응답 생성 중"})
    stream = await client.chat.completions.create(
      model=request.model,
      temperature=request.temperature,
      stream=True,
      messages=[
        {"role": "system", "content": "You are a helpful customer support assistant"},
        {"role": "user", "content": request.message}
      ]
    )

    async for chunk in stream:
      delta = chunk.choices[0].delta
      token = delta.content or ""
      if not token:
        continue
      yield sse_data({"type": "token", "delta": token})
  except Exception:
    yield sse_data({"type": "status", "label": "응답 생성 중 오류가 발생했습니다."})
  finally:
    yield "data: [DONE]\n\n"

router = APIRouter(prefix="/chat", tags=["chat-stream"])

@router.post("/stream")
async def chat_stream(request: StreamRequest) -> StreamingResponse:
  """사용자 메시지에 대한 OpenAI 응답을 SSE로 스트리밍한다."""
  return StreamingResponse(
    event_generator(request),
    media_type="text/event-stream",
    # media_type="text/html",
    # media_type="text/application-json",
    headers={
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no"
    }
  )
