from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel

from models import get_model

def build_rag_chain():
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 시내문서를 기반으로 직원 질문에 답하는 AI 어시스턴트입니다.\n"
         "제공된 문서 근거만 사용하여 정확하게 답변하세요.\n"
         "근거가 부족하면 '제공된 문서에서 관련 내용을 찾기 어렵습니다'라고 솔직히 답변합니다."
        ),
        ("human",
         "## 참고문서\n{context}\n\n"
         "## 질문\n{question}")
    ])
    return prompt | model | StrOutputParser()

def build_chat_chain():
    pass

def build_structured_chain():
    pass

def build_parallel_chain():
    pass
