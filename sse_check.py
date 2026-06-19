"""sse_check.py — 기존 streaming endpoint 회귀 확인 (Day 4 self2 종료기준 ③)
오늘 추가한 graph 모듈 때문에 기존 streaming이 깨지지 않았는지만 확인.
※ URL·경로·payload는 네 앱(9주차 ai-chatbot-service)의 실제 값으로 바꿔.
"""
import httpx

URL = "http://127.0.0.1:8000/chat/stream"     # ← 네 앱의 실제 streaming endpoint 경로
PAYLOAD = {"message": "면접 코치 데모 질문"}     # ← 네 앱이 받는 실제 body 형태

print(f"[sse_check] POST {URL}")
try:
    with httpx.stream("POST", URL, json=PAYLOAD, timeout=30.0) as resp:
        print("[sse_check] status:", resp.status_code)
        for line in resp.iter_lines():
            if line:
                print(line)   # data: {...} frame이 줄줄이 나오면 통과
except Exception as e:
    print("[sse_check] 실패:", type(e).__name__, e)