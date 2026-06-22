# Submission Checklist - 4구역

> freeze 완료. 이후 허용 행동: 캡처 보완·상태 기재만. 기능 추가·코드 수정 금지.

## 4구역 체크리스트

| 구역     | 점검 항목                                                    | 상태                                                                                                              |
| -------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 기능     | RAG 질문 입력 → 답변 도착                                   | 완료 —`POST /rag` (JSON) + `GET /rag/stream` (SSE) 모두 구현                                                 |
| 기능     | 검색 preview 표시                                            | 완료 —`streamlit_rag_ui.py` `st.status("검색 중...")` 구현                                                   |
| 기능     | streaming 답변 진행                                          | 완료 —`/rag/stream` 5단어 단위 SSE token 전달                                                                  |
| 기능     | 출처 card 표시                                               | 완료 —`st.expander("출처 N건")` source/page/snippet 3필드 표시                                                 |
| 기능     | session continuity (같은 세션 이어지는 대화)                 | 완료 —`InMemorySaver` + `config={"configurable": {"thread_id": ...}}`. `rag_graph.py` 2+1 thread 증거 포함 |
| evidence | Streamlit 화면 캡처 (질문→streaming→출처 한 흐름)          | 보류 — 앱 기동 후 캡처 필요 (`cap/internal-qa_stream.png`)                                                     |
| evidence | FastAPI 응답 흐름 캡처                                       | 보류 —`/rag` 또는 `/rag/stream` 응답 캡처 필요                                                               |
| evidence | LangSmith trace 식별 캡처 (또는 fallback 로그)               | 완료(fallback) —`handoff_note.md` Day3 baseline + `evidence_metrix.md` fallback route_log                    |
| evidence | 출처 card 단독 캡처                                          | 보류 —`st.expander` 펼친 화면 캡처 필요 (`cap/internal-qa_sources.png`)                                      |
| hygiene  | .env 내용·API key 3종이 어떤 캡처·노트에도 없음            | 완료 — 코드 전체 `load_dotenv()` 사용, 하드코딩 없음                                                           |
| hygiene  | 임시 로그·디버그 출력 파일이 제출물에 없음                  | 완료                                                                                                              |
| hygiene  | 깨진 import·실행 불가 파일이 제출 대상에 없음               | 완료 — 전체 import 체인 정상                                                                                     |
| handoff  | handoff note 5줄 작성 완료                                   | 완료 —`handoff_note.md` 참고                                                                                   |
| handoff  | Agent-MCP 항목이 아이디어 1줄로만 존재 (코드/설치/설계 없음) | 완료                                                                                                              |

---

## 보류 항목 보정 우선순위

1. **캡처 3종 수집** (evidence 구역 보류 3건)

   - `uv run uvicorn main:app --reload` → `uv run streamlit run streamlit_rag_ui.py`
   - RAG 모드에서 질문 입력 → preview/stream/출처 card 화면 캡처
   - 저장 위치: `cap/` 폴더
2. **thread_id 접두 규약 통일**

   - 현재 코드: `qa:demo-1` / 가이드 규약: `internal_qa:<session_id>`
   - 보정: backend 요청 직전 `f"internal_qa:{session_id}"` 합성 로직 추가 (self2 이후)

---

## 제출 파일 목록

| 파일                           | 역할                     | 상태          |
| ------------------------------ | ------------------------ | ------------- |
| `rag_pipeline.py`            | RAG 기반 인프라          | 완료          |
| `embeddings_factory.py`      | Embedding factory        | 완료          |
| `rag_chain.py`               | LCEL RAG chain           | 완료          |
| `interview_rag.py`           | 면접 RAG (CUSTOMIZE 3곳) | 완료          |
| `rag_graph.py`               | LangGraph workflow       | 완료          |
| `interview_graph.py`         | 면접 graph               | 완료          |
| `interview_graph_wrapper.py` | wrapper                  | 완료          |
| `main.py`                    | FastAPI app              | 완료          |
| `streamlit_rag_ui.py`        | Streamlit UI             | 완료          |
| `chains.py`                  | Chain factory            | 완료          |
| `README.md`                  | 실행 방법 + Q10-1 확인표 | 완료          |
| `evidence_metrix.md`         | 2행×6열 matrix          | 완료          |
| `submission_checklist.md`    | 4구역 checklist          | 완료          |
| `demo_script.md`             | 6-step 시연 스크립트     | 완료          |
| `handoff_note.md`            | 11주차 인계 메모 5줄     | 완료          |
| `pyproject.toml`             | 의존성 관리              | 완료          |
| `.env`                       | API 키 (Git 제외)        | 완료 (미제출) |
