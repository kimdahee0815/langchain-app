# 사용자 (브라우저 / Swagger / httpx)
#        |  POST /chat {"message": "..."}
# FastAPI  Eendpoint     -    this.route.adapter (오늘의 작업 구역)
#        v  {"question": req.message}              ^ result(str)
# LCEL chain   - prompt | model | parser          (오전에 안정화한 부품)

from fastapi import FastAPI
from pydantic import BaseModel, Field
#from openai import AsyncOpenAI
from chain import build_chat_chain
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
# client = AsyncOpenAI()
chain = build_chat_chain()

class ChatRequest(BaseModel):
    message: str # 사용자 입력이 저장되는 곳

@app.post("/chat")
async def chat(req: ChatRequest):
    # <----- LCEL REPLACE START ------>
    # response = await client.chat.completions.create(
    #     model="gpt-5.4-nano",
    #     messages=[{"role":"user", "content":req.message}]
    # )
    # result = response.choices[0].message.content
    result = await chain.ainvoke({"question":req.message})
    # <----- LCEL REPLACE END ------>
    
    return {"reply": result}