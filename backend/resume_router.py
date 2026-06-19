import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field

load_dotenv()

router = APIRouter(prefix="/resume", tags=["resume"])


class ResumeQuestionRequest(BaseModel):
    """이력서 기반 질문 생성 요청 모델입니다."""

    resume_text: str = Field(..., min_length=30)
    model: str = Field(default="gpt-5.4-nano")
    system_prompt: str = Field(default="")
    question_count: int = Field(default=5, ge=3, le=10)
    role_preset: str = Field(default="기술 면접")


class ResumeQuestionResponse(BaseModel):
    """이력서 기반 질문 생성 응답 모델입니다."""

    questions: list[str]
    tool_calls: list[dict[str, Any]]


def get_resume_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return AsyncOpenAI(api_key=api_key)


# ── Function Calling용 도구 ──

def extract_resume_keywords(section: str, resume_text: str) -> dict[str, Any]:
    """이력서 텍스트에서 [섹션명] 본문을 찾아 핵심 키워드를 추출한다 (간단한 빈도 기반)."""
    pattern = rf"\[{re.escape(section)}\](.*?)(?=\n\[|\Z)"
    match = re.search(pattern, resume_text, re.DOTALL)

    if not match:
        return {"section": section, "found": False, "keywords": []}

    section_text = match.group(1).strip()

    words = re.findall(r"[가-힣A-Za-z]{2,}", section_text)
    stopwords = {"그리고", "그러나", "이는", "있습니다", "했습니다", "위해", "통해", "그래서"}

    counts: dict[str, int] = {}
    for word in words:
        if word in stopwords:
            continue
        counts[word] = counts.get(word, 0) + 1

    top_keywords = sorted(counts, key=counts.get, reverse=True)[:5]

    return {"section": section, "found": True, "keywords": top_keywords}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_resume_keywords",
            "description": (
                "이력서 텍스트에서 특정 섹션(예: 직무 관련 경험, 성장 과정)의 "
                "핵심 키워드를 추출합니다. 질문을 만들기 전에 호출하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "분석할 섹션 이름 (예: 직무 관련 경험)",
                    },
                },
                "required": ["section"],
            },
        },
    }
]


@router.post("/questions", response_model=ResumeQuestionResponse)
async def generate_resume_questions(request: ResumeQuestionRequest) -> ResumeQuestionResponse:
    """이력서를 분석해 맞춤 면접 질문을 생성한다."""
    client = get_resume_openai_client()

    system_prompt = request.system_prompt or (
        f"당신은 {request.role_preset} 면접관입니다. "
        "지원자의 자기소개서를 분석해 맞춤 면접 질문을 만듭니다."
    )

    user_content = (
        f"[자기소개서]\n{request.resume_text}\n\n"
        f"위 자기소개서를 바탕으로 면접 질문 {request.question_count}개를 만들어 주세요. "
        "질문을 만들기 전에 extract_resume_keywords 도구로 '직무 관련 경험' 섹션의 키워드를 먼저 확인하세요. "
        "최종 응답은 질문만 한 줄에 하나씩, 번호나 기호 없이 작성하세요."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tool_calls_log: list[dict[str, Any]] = []

    try:
        response = await client.chat.completions.create(
            model=request.model,
            temperature=0.7,
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                }
            )

            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                section = args.get("section", "직무 관련 경험")
                result = extract_resume_keywords(section, request.resume_text)

                tool_calls_log.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": args,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = await client.chat.completions.create(
                model=request.model,
                temperature=0.7,
                messages=messages,
            )
            message = response.choices[0].message

        content = message.content or ""

    except RateLimitError:
        raise HTTPException(status_code=429, detail="rate_limited")
    except APIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    questions = [
        line.strip(" -*0123456789.").strip()
        for line in content.splitlines()
        if line.strip()
    ][: request.question_count]

    return ResumeQuestionResponse(questions=questions, tool_calls=tool_calls_log)