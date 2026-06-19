from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel
from pydantic import Field

router = APIRouter(prefix="/feedback", tags=["feedback"])

# 디비에 저장되어야 한다. 인메모리 리스트로 저장 함.
feedback_store: list[dict[str, Any]] = []

class FeedbackPayload(BaseModel):
    """AI 응답 1개에 대한 사용자 피드백 요청 모델"""
    conversation_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    rating: Literal["up", "down"]
    comment: str | None = None
    metadata: dict[str, Any] | None = None
    
class FeedbackResponse(BaseModel):
    """저장된 피드백 정보를 프론트에 돌려주는 모델"""
    feedback_id: str
    conversation_id: str
    message_id: str
    rating: Literal["up", "down"]
    create_at: str

def save_feedback(payload: FeedbackPayload) -> FeedbackResponse:
    """검증된 payload를 인메모리 저장소에 추가"""
    feedback_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    saved_feedback = {
        "feedback_id": feedback_id,
        "create_at": created_at,
        **payload.model_dump()
    }
    feedback_store.append(saved_feedback)
    return FeedbackResponse(
        feedback_id=feedback_id,
        conversation_id=payload.conversation_id,
        message_id=payload.message_id,
        rating=payload.rating,
        create_at=created_at
    )
    
@router.post("", response_model=FeedbackResponse, status_code=201)
def create_feedback(payload: FeedbackPayload) -> FeedbackResponse:
    """POST /feedback 요청을 받아서 피드백을 저장"""
    return save_feedback(payload)

@router.get("")
def list_feedback() -> dict[str, Any]:
    """GET /feedback 요청을 받는 조회용 엔드포인트"""
    return {
        "count": len(feedback_store),
        "items": feedback_store
    }