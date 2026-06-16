from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_chat_chain():
    """chain을 조립해서 리턴"""
    model = init_chat_model("openai:gpt-5.4-nano")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "당신은 친절한 면접 코치 입니다. 다음 질문에 3문장 이내로 답하세요.: {question}")
        ]
    )
    return prompt | model | StrOutputParser()