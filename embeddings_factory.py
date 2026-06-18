from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings() -> OpenAIEmbeddings:
    """embedding 객체를 만든다"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

if __name__=="__main__":
    embeddings = get_embeddings()
    vector = embeddings.embed_query("테스트") # 질문 1 -> 벡터 1
    
    print(len(vector)) # 벡터의 길이만 출력