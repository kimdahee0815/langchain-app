# structured_chain.py의 InterviewScore 패턴을 면접 코치 Step 1용으로 가져와요.
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class InterviewScore(BaseModel):
    """면접 답변 평가 결과 — Step 1의 출력 4가지를 그대로 필드로 옮겨요."""
    score: int = Field(ge=1, le=5, description="답변 종합 점수 (1~5)")
    strengths: str = Field(description="답변에서 잘한 점 요약")
    improvements: str = Field(description="답변에서 부족한 점과 보완 방향 요약")
    next_question: str = Field(description="이어서 물어볼 후속 면접 질문")
    
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {job_role} 직무 면접관입니다. 후보자 답변을 평가 기준에 따라 채점하세요."),
    ("human", "질문: {question}\n후보자 답변: {candidate_answer}\n위 답변을 평가하세요."),
])

model = init_chat_model("openai:gpt-4o-mini")
structured_model = model.with_structured_output(InterviewScore)

chain = prompt | structured_model 

if __name__ == "__main__":
    result = chain.invoke({
        "job_role": "B2B SaaS 기술 영업",
        "question": "도입을 망설이던 고객을 설득해서 계약까지 끌고 간 경험을 구체적으로 말해 주세요.",
        "candidate_answer": (
            "지난 분기에 PoC 단계에서 석 달째 멈춰 있던 고객사가 있었어요. "
            "운영팀 실무자 인터뷰로 실제 병목이 보안 검토 절차라는 걸 확인했고, "
            "그 절차에 맞춘 도입 일정과 ROI 시나리오를 다시 만들어 제안했습니다. "
            "2주 뒤 후속 미팅에서 연간 계약으로 전환됐어요."
        ),
    })
    print(type(result))  # 검증된 InterviewScore 객체인지 확인해요
    print(result.model_dump_json(indent=2))  # 샘플 JSON 저장용 출력 (Pydantic v2 직렬화)
