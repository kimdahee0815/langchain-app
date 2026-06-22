# 10주차 RAG + LangGraph 통합 서비스

사내 문서 QA 봇(internal-qa)과 면접 코치(interview-coach) 2-track RAG 통합 서비스입니다.

## 기술 스택

| 구성 요소 | 파일 | 역할 |
|-----------|------|------|
| RAG 기반 인프라 | `rag_pipeline.py` | `get_retriever(k=3)`, `format_sources()` |
| Embedding factory | `embeddings_factory.py` | `get_embeddings()` — text-embedding-3-small |
| LCEL RAG chain | `rag_chain.py` | `build_rag_chain`, `ask_rag` — `{answer, sources}` fan-out |
| 면접 RAG chain | `interview_rag.py` | CUSTOMIZE 3곳 교체, `chroma_job_docs` 분리 |
| LangGraph workflow | `rag_graph.py` | `StateGraph` + conditional edges + `InMemorySaver` |
| 면접 graph | `interview_graph.py`, `interview_graph_wrapper.py` | `run_interview_graph(question, thread_id)` |
| FastAPI backend | `main.py` | `/rag`, `/rag/stream`, `/chat`, `/interview/*` |
| Streamlit UI | `streamlit_rag_ui.py` | source panel + streaming 표시 |
| Vector store (QA) | `chroma_company_docs/` | 사내 문서 index |
| Vector store (면접) | `chroma_job_docs/` | 직무 문서 index |

---

## 실행 방법

### 1. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 아래 변수명을 설정하세요 (값은 직접 입력):

```
OPENAI_API_KEY=<your-openai-api-key>
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=<your-project-name>
```

### 2. 패키지 설치

```bash
uv sync
```

주요 의존성: `langchain`, `langchain-openai`, `langchain-chroma`, `langgraph`, `fastapi`, `streamlit`, `python-dotenv`

### 3. Vector Store 구축 (최초 1회)

```bash
# 사내 QA 트랙 — chroma_company_docs 생성
uv run python rag_pipeline.py

# 면접 코치 트랙 — chroma_job_docs 생성
uv run python interview_rag.py
```

### 4. Backend 서버 실행

```bash
uv run uvicorn main:app --reload
# → http://127.0.0.1:8000
# → Swagger UI: http://127.0.0.1:8000/docs
```

### 5. Streamlit UI 실행 (별도 터미널)

```bash
uv run streamlit run streamlit_rag_ui.py
# → http://localhost:8501
```

사이드바 모드 선택으로 트랙 전환:
- **RAG 사내 문서 QA**: 사내 규정 문서 기반 질의응답
- **기본 면접 코치**: 자유 면접 질문 답변
- **면접 답변 평가**: `질문 | 답변` 형식으로 구조화 평가
- **병렬 + 분기 응답**: 답변 + FAQ 동시 생성

### 6. 면접 코치 그래프 직접 실행 (옵션)

```bash
uv run python interview_graph_wrapper.py
```

---

## Q10-1 핵심 기능 확인표

| # | 기준 | 대응 파일 | 확인 결과 |
|:-:|------|-----------|-----------|
| ① | LCEL RAG chain — `{answer, sources}` 동시 반환 | `rag_chain.py` | `RunnableParallel` fan-out: `retriever.invoke` 1회 후 answer/sources를 동일 docs에서 파생. `build_rag_chain(retriever, system_prompt)` |
| ② | 면접 RAG 전이 — CUSTOMIZE 3곳 교체 | `interview_rag.py` | `JOB_DOC_PATHS` / `PERSIST_DIR="./chroma_job_docs"` / `SYSTEM_PROMPT` 3곳 교체. `build_rag_chain` import 재사용 (복제 없음) |
| ③ | LangGraph conditional workflow | `rag_graph.py` | `StateGraph` + `add_conditional_edges("grade", should_retry, {...})` + `MAX_RETRIES=1` cap. `should_retry` → `"retry"` / `"generate"` |
| ④ | Checkpointer + thread_id 세션 분리 | `rag_graph.py`, `interview_graph_wrapper.py` | `InMemorySaver()` + `compile(checkpointer=checkpointer)` + `config={"configurable": {"thread_id": ...}}`. 2+1 thread 분리: `qa:demo-1` 2회(이어짐) + `qa:demo-2` 1회(분리) |
| ⑤ | 2-track evidence matrix + submission checklist | `evidence_metrix.md`, `submission_checklist.md` | 2행×6열 matrix 작성 완료. 4구역 checklist 작성 완료 |
