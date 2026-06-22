"""chains.py — LCEL chain factory 모음 (Day 1 S4, S6 / Day 2 S5)

교안 핵심 패턴:
- build_chat_chain(): prompt | model | StrOutputParser()  — 기본 LCEL 파이프
- build_structured_chain(): prompt | model.with_structured_output(InterviewScore)
    - StrOutputParser를 붙이면 안 됨 (Pydantic 인스턴스 → str 파서 타입 충돌)
- build_rag_chain(): RAG 답변 생성용 chain — context + question 입력
- build_parallel_chain(): RunnableParallel + RunnableBranch + with_fallbacks

route 파일(backend/app.py)에는 chain 조립 코드를 두지 않음 — factory/route 분리 원칙
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableParallel

from models import get_model
from schemas import InterviewScore


# ─── 기본 채팅 chain (Day 1 S4, Day 2 S5) ──────────────────────────

def build_chat_chain():
    """면접 코치 기본 chain: prompt | model | StrOutputParser()

    - endpoint에서는 await chain.ainvoke({"question": req.message})로 호출
    - 모듈 레벨 1회 생성, 모든 요청이 같은 chain 공유 (stateless)
    """
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 면접 코치입니다. 간결하고 실용적인 조언을 제공하세요."),
        ("human", "{question}"),
    ])
    return prompt | model | StrOutputParser()


# ─── 구조화 출력 chain (Day 1 S6) ───────────────────────────────────

def build_structured_chain():
    """면접 답변 평가 chain: with_structured_output(InterviewScore)

    - StrOutputParser를 붙이지 않음 — 결과가 InterviewScore 인스턴스
    - endpoint에서 result.model_dump()로 JSON 변환
    """
    model = get_model()
    structured_model = model.with_structured_output(InterviewScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "면접관으로서 아래 답변을 평가하세요. "
         "score(1~5), strengths, improvements, next_question을 반환하세요."),
        ("human", "면접 질문: {question}\n답변: {answer}"),
    ])
    return prompt | structured_model


# ─── RAG 답변 생성 chain (Day 3-4 통합) ─────────────────────────────

def build_rag_chain():
    """RAG 답변 생성 chain: context + question → 근거 기반 답변

    - retrieve → grade → generate 흐름에서 generate node가 사용
    - context는 검색된 chunk들의 page_content를 줄바꿈으로 결합한 문자열
    """
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "당신은 사내 문서를 기반으로 직원 질문에 답하는 AI 어시스턴트입니다.\n"
            "제공된 문서 근거만 사용하여 정확하게 답변하세요.\n"
            "근거가 부족하면 '제공된 문서에서 관련 내용을 찾기 어렵습니다'라고 솔직히 답하세요."),
        ("human",
            "## 참고 문서\n{context}\n\n"
            "## 질문\n{question}"),
    ])
    return prompt | model | StrOutputParser()


# ─── 병렬 + 분기 + Fallback chain (Day 2 S3-4) ─────────────────────

def build_parallel_chain():
    """RunnableParallel + RunnableBranch + with_fallbacks 통합 chain

    - parallel: 동일 입력으로 answer + faq를 동시 생성
    - branch: 질문 길이에 따라 상세/간결 응답 분기
    - fallback: 기본 모델 실패 시 대체 모델 시도
    """
    model = get_model()
    fallback_model = get_model()  # 동일 모델이지만 fallback 패턴 시연용

    # 상세 응답 prompt
    detailed_prompt = ChatPromptTemplate.from_messages([
        ("system", "면접 코치로서 상세하게 5문장 이내로 답변하세요."),
        ("human", "{question}"),
    ])

    # 간결 응답 prompt
    brief_prompt = ChatPromptTemplate.from_messages([
        ("system", "면접 코치로서 2문장 이내로 간결하게 답변하세요."),
        ("human", "{question}"),
    ])

    # RunnableBranch: 질문 길이 30자 기준 분기
    answer_branch = RunnableBranch(
        (lambda x: len(x.get("question", "")) > 30,
            detailed_prompt | model | StrOutputParser()),
        brief_prompt | model | StrOutputParser(),  # default
    )

    # FAQ chain
    faq_prompt = ChatPromptTemplate.from_messages([
        ("system", "이 질문과 관련된 면접 FAQ 2개를 '-' 목록으로 간단히 제시하세요."),
        ("human", "{question}"),
    ])
    faq_chain = faq_prompt | model | StrOutputParser()

    # RunnableParallel: answer + faq 동시 실행
    parallel = RunnableParallel(answer=answer_branch, faq=faq_chain)

    # with_fallbacks: 기본 parallel 실패 시 fallback
    fallback_chain = RunnableParallel(
        answer=brief_prompt | fallback_model | StrOutputParser(),
        faq=lambda x: "FAQ를 생성할 수 없습니다.",
    )

    return parallel.with_fallbacks([fallback_chain])


# ─── smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # 기본 chain smoke
    chat = build_chat_chain()
    print("[chat]", chat.invoke({"question": "자기소개는 어떻게 시작하면 좋을까요?"}))

    # 구조화 chain smoke
    structured = build_structured_chain()
    result = structured.invoke({
        "question": "자기소개를 해주세요",
        "answer": "저는 개발자입니다.",
    })
    print("[structured]", result)

    # RAG chain smoke (context 직접 전달)
    rag = build_rag_chain()
    print("[rag]", rag.invoke({
        "context": "연차 휴가는 입사 1년차에 15일이 부여됩니다.",
        "question": "연차 휴가는 며칠인가요?",
    }))

    # 병렬 chain smoke
    parallel = build_parallel_chain()
    print("[parallel]", parallel.invoke({"question": "면접에서 떨지 않으려면?"}))
