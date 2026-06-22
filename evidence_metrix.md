# Evidence Matrix - 2-track RAG 통합 서비스

> 채점 기준: trace 또는 fallback 중 1개면 trace_or_fallback 칸 인정. 무증거만 불가.

## evidence matrix (2행×6열)

| 트랙            | question                                                                  | search_preview                                          | answer_stream                                           | source_card                                             | trace_or_fallback                                                     | thread_id             |
| --------------- | ------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- | --------------------- |
| internal-qa     | "재택 근무 규정은?"                                                       | ![1782111692068](image/evidence_metrix/1782111692068.png) | ![1782111766061](image/evidence_metrix/1782111766061.png) | ![1782111804894](image/evidence_metrix/1782111804894.png) | fallback: handoff_note.md (Day3 baseline - answer/sources 6필드 포함) | `qa:demo-1`         |
| interview-coach | "이 직무에서 가장 중요한 역량을 기준으로 면접 질문 1개를 만들어 주세요." | ![1782112413217](image/evidence_metrix/1782112413217.png) | ![1782112430011](image/evidence_metrix/1782112430011.png) | ![1782112461396](image/evidence_metrix/1782112461396.png) | fallback: 아래 route_log 참고                                         | `interview:alice:1` |

---

## 트랙별 보정 메모

### internal-qa

- **search_preview**: `uv run uvicorn main:app --reload` 기동 후 Streamlit에서 질문 입력 시 `st.status("검색 중...")` 표시 확인 → 캡처 필요
- **answer_stream**: `/rag/stream` SSE endpoint 5단어 단위 token 전달 → 브라우저 화면 캡처 필요
- **source_card**: `st.expander("출처 N건")` 펼친 화면 캡처 필요
- **trace_or_fallback**: LangSmith trace 미확인 → Day3 baseline(handoff_note.md)을 fallback evidence로 사용. 6필드: project=`week10-internal-qa`, question=`휴가 신청 절차는?`, route=`/rag`, result=answer+sources 3건 반환 확인
- **thread_id 접두 불일치**: 코드(`qa:demo-1`)가 가이드 규약(`internal_qa:`) 과 다름 → 보정 우선순위 1번 (backend 요청 직전 `internal_qa:{session_id}` 합성 로직 추가 필요)

### interview-coach

- **search_preview**: `chroma_job_docs` 로드 후 `retrieve` 노드 실행 → 캡처 필요
- **answer_stream**: `/interview/rag/stream` SSE endpoint → 캡처 필요
- **source_card**: `interview_graph.py` generate 노드가 `format_sources(docs)` → `sources` state 채널로 반환, Streamlit expander 캡처 필요
- **trace_or_fallback**: 아래 fallback route_log 참고

---

## fallback route_log - interview-coach

| 필드               | 값                                                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| project_name       | week10-interview-coach                                                                                                      |
| thread_or_run_hint | `interview:alice:1` (1회차), `interview:alice:1` (2회차 — 맥락 유지), `interview:bob:1` (분리 확인)                  |
| question           | "이 직무에서 가장 중요한 역량을 기준으로 면접 질문 1개를 만들어 주세요."                                                    |
| route              | `interview_graph_wrapper.run_interview_graph` → `InterviewRagState` → retrieve→grade→generate                       |
| observed_result    | `should_retry` router: quality.passed=True(sources≥2) → "generate" 직행. answer + sources(3건) 반환. route_log 1건 누적 |
| next_input         | thread_id 접두 `internal_qa:` 규약 통일, LangSmith trace URL 수집                                                         |

---
