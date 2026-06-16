from dotenv  import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def get_model():
    """provider prefix가 포함된 ChatModel을 돌려준다."""
    return init_chat_model("openai:gpt-5.4-nano") # "openai": 프로바이더 표기

# print(get_model())
# print("prefix가 없는: ", init_chat_model("gpt-5.4-nano")) # prefix가 없으면 추측에 의존해서 모델 설정을 한다.
# print("prefix가 있는: ", init_chat_model("openai:gpt-5.4-nano")) # 필수!! prefix 추가. 추측을 하지 않음.
#init_chat_model("anthropic:opus-4-7") # 현재 에러가 남. provider 전용 엔진 라이브러리 설치 전! # langchain-anthropic
if __name__ == "__main__":
    model = get_model()
    response = model.invoke("LangChain을 한 문장으로 설명해줘.") # 실제 서비스 호출이 되는
    
    print(type(response))
    print(dir(response))
    print(response.content)