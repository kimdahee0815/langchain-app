"""interview_schemas.py — 면접 코치 RAG Pydantic schemas

교안 핵심 패턴:
- InterviewRagRequest: 면접 코치 RAG 요청 (question 필수)
- InterviewRagEvalRequest: 직무 RAG 기반 답변 평가 요청 (question + answer)
- InterviewRagResponse: RAG 응답 계약 — answer + sources + quality + attempts
- InterviewTip: 면접 전략/팁 응답
"""

from pydantic import BaseModel, Field

from schemas import SourceItem


class InterviewRagRequest(BaseModel):
    """면접 코치 RAG 질문 요청"""

    question: str = Field(description="면접 관련 질문 (예: '백엔드 직무 기술 면접 질문을 만들어 주세요')")


class InterviewRagEvalRequest(BaseModel):
    """직무 RAG 기반 면접 답변 평가 요청"""

    question: str = Field(description="면접 질문")
    answer: str = Field(description="지원자의 면접 답변")


class InterviewRagResponse(BaseModel):
    """면접 코치 RAG 응답 계약 — answer + sources"""

    answer: str = Field(description="면접 코치의 답변 (직무 문서 기반)")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="근거 직무 문서 목록 (top-k)"
    )
    quality_passed: bool = Field(default=True, description="품질 검수 통과 여부")
    attempts: int = Field(default=0, description="재검색 시도 횟수")


class InterviewParallelResponse(BaseModel):
    """면접 질문 생성 + 팁 병렬 응답"""

    questions: str = Field(description="생성된 면접 질문들")
    tips: str = Field(description="면접 준비 팁")
