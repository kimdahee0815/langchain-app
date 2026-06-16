from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import get_model  # init_chat_model("openai:gpt-4o-mini") factory

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 기술 튜터입니다."),
    ("human", "{topic}을 초보자에게 3문장으로 설명해 주세요."),
])
model = get_model()
parser = StrOutputParser()

chain = prompt | model | parser  # RunnableSequence 조립 — 이 줄에서는 API 호출이 일어나지 않아요

if __name__ == "__main__":
    result = chain.invoke({"topic": "LCEL"})  # dict 입력, key는 template 변수명과 일치
    print(result)

# 출력: (예시 — 모델 응답이라 문구는 달라질 수 있어요)
# LCEL은 LangChain에서 prompt, model, parser를 파이프(|)로 연결해 체인을 만드는
# 표현식이에요. 각 구성 요소를 레고 블록처럼 조립하면 하나의 실행 가능한 흐름이 됩니다.
# 완성된 체인은 .invoke() 한 번으로 입력부터 최종 문자열까지 한 번에 처리해요.
