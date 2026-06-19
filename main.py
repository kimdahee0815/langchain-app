from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import interview_router, stream_router
from backend import agents_router
from backend import feedback_router
from backend import resume_router

app = FastAPI(title="Customer Support Chatbot API")
# Streamlit 개발 서버가 열리는 origin만 허용합니다.
allowed_origins= ["http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,       # 허용할 출처
    allow_credentials=True,
    allow_methods=["*"],       # 허용할 HTTP 메서드
    allow_headers=["*"],       # 허용할 헤더
)

app.include_router(agents_router.router)
app.include_router(interview_router.router)
app.include_router(feedback_router.router)
app.include_router(resume_router.router)
app.include_router(stream_router.router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
