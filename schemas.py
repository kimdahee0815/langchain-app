"""schemas.py — Pydantic schemas (Day 1 S6, Day 5 S3)

교안 핵심 패턴:
- InterviewScore: with_structured_output 계약용 Pydantic 모델
    - Field(ge=1, le=5)로 범위 제약
    - description으로 LLM에 출력 가이드 제공
- SourceItem / RagResponse: RAG 출처 계약용 스키마
"""

from pydantic import BaseModel, Field


class InterviewScore(BaseModel):
    """면접 답변 평가 결과 — with_structured_output 계약 (Day 1 S6)"""

    score: int = Field(ge=1, le=5, description="1(매우 부족)~5(우수) 점수")
    strengths: str = Field(description="답변의 강점 1~2문장")
    improvements: str = Field(description="개선할 점 1~2문장")
    next_question: str = Field(description="이어질 후속 면접 질문 1개")


class SourceItem(BaseModel):
    """RAG 검색 결과의 출처 1건 (Day 5 S3 출처 카드 계약)"""

    source: str = Field(description="문서 출처 (파일명)")
    page: int = Field(default=0, description="페이지 번호 (0-indexed)")
    snippet: str = Field(description="원문 발췌 (120자 이내)")


class RagResponse(BaseModel):
    """RAG 응답 계약 — answer + sources (Day 5 S3-4)"""

    answer: str = Field(description="LLM이 생성한 답변")
    sources: list[SourceItem] = Field(default_factory=list, description="근거 문서 목록 (top-k)")
    quality_passed: bool = Field(default=True, description="품질 검수 통과 여부")
    attempts: int = Field(default=0, description="재검색 시도 횟수")


class ChatRequest(BaseModel):
    """채팅 요청 공통 스키마"""

    message: str = Field(description="사용자 메시지")


class ChatResponse(BaseModel):
    """채팅 응답 공통 스키마"""

    reply: str = Field(description="AI 응답")
