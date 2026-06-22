"""frontend/app.py — Streamlit 채팅 UI (Day 5 S3-S4)

교안 핵심 패턴:
- st.chat_message + st.chat_input으로 채팅 인터페이스 구현
- 3가지 모드: 기본 채팅 / 구조화 평가 / RAG QA
- RAG 모드:
    - st.status: 검색 미리보기 (진행 상태 표시)
    - st.expander: 출처 카드 (source/page/snippet 3필드)
    - SourceItem 계약 보존
- session_state로 대화 이력 관리
"""

import httpx
import streamlit as st

# ─── 페이지 설정 ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="10주차 관통예제 — AI 사내 문서 QA",
    page_icon="🤖",
    layout="wide",
)

# ─── Backend URL ─────────────────────────────────────────────────────

BACKEND_URL = "http://127.0.0.1:8000"

# ─── Session State 초기화 ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "rag"

# ─── 사이드바 ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")

    mode = st.radio(
        "모드 선택",
        ["rag", "chat", "structured", "parallel"],
        format_func=lambda x: {
            "rag": "📚 RAG 사내 문서 QA",
            "chat": "💬 기본 면접 코치",
            "structured": "📊 면접 답변 평가",
            "parallel": "🔀 병렬 + 분기 응답",
        }[x],
        index=0,
    )
    st.session_state.mode = mode

    st.divider()

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("10주차 관통예제")
    st.caption("LangChain + LangGraph + Chroma")

    # 모드별 안내
    mode_info = {
        "rag": "사내 규정 문서를 기반으로 질문에 답합니다.\n\n예시 질문:\n- 휴가 신청 절차는?\n- 출장비 한도는?\n- 재택근무 규정은?",
        "chat": "면접 코치로서 자유 질문에 답합니다.",
        "structured": "면접 답변을 점수/강점/개선점으로 구조화 평가합니다.\n\n형식: `질문 | 답변`",
        "parallel": "병렬 응답(답변 + FAQ)을 동시에 생성합니다.",
    }
    st.info(mode_info[mode])

# ─── 메인 영역 ───────────────────────────────────────────────────────

st.title("🤖 AI 사내 문서 QA 챗봇")
st.caption("10주차 관통예제 — LangChain LCEL + LangGraph + Chroma RAG")

# 기존 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📎 출처 ({len(msg['sources'])}건)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src.get('source', 'unknown')}** "
                        f"(p.{src.get('page', 0)})\n\n"
                        f"> {src.get('snippet', '')}"
                    )
                    st.divider()
        if "metadata" in msg and msg["metadata"]:
            with st.expander("🔍 상세 정보", expanded=False):
                st.json(msg["metadata"])

# ─── 채팅 입력 ───────────────────────────────────────────────────────

if user_input := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 처리
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == "rag":
                # ── RAG 모드 ─────────────────────────────────────
                with st.status("🔍 사내 문서 검색 중...", expanded=True) as status:
                    st.write("질문 분석 및 문서 검색...")

                    response = httpx.post(
                        f"{BACKEND_URL}/rag",
                        json={"message": user_input},
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.write(f"✅ 검색 완료 (근거 {len(data.get('sources', []))}건)")

                    if data.get("attempts", 0) > 0:
                        st.write(f"🔄 재검색 {data['attempts']}회 수행")

                    status.update(
                        label="검색 완료",
                        state="complete",
                        expanded=False,
                    )

                # 답변 표시
                answer = data.get("answer", "응답을 받지 못했습니다.")
                st.markdown(answer)

                # 출처 카드 (st.expander)
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📎 출처 ({len(sources)}건)", expanded=True):
                        for src in sources:
                            st.markdown(
                                f"**{src.get('source', 'unknown')}** "
                                f"(p.{src.get('page', 0)})\n\n"
                                f"> {src.get('snippet', '')}"
                            )
                            st.divider()

                # 메타데이터
                metadata = {
                    "quality_passed": data.get("quality_passed", False),
                    "attempts": data.get("attempts", 0),
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metadata": metadata,
                })

            elif st.session_state.mode == "chat":
                # ── 면접 코치 RAG 모드 (job_posting.pdf 기반) ────
                with st.status("🔍 직무 문서 검색 중...", expanded=True) as status:
                    st.write("질문 분석 및 직무 문서 검색...")

                    response = httpx.post(
                        f"{BACKEND_URL}/interview/rag",
                        json={"question": user_input},
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.write(f"✅ 검색 완료 (근거 {len(data.get('sources', []))}건)")
                    status.update(label="검색 완료", state="complete", expanded=False)

                answer = data.get("answer", "응답을 받지 못했습니다.")
                st.markdown(answer)

                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📎 직무 문서 출처 ({len(sources)}건)", expanded=False):
                        for src in sources:
                            st.markdown(
                                f"**{src.get('source', 'unknown')}** "
                                f"(p.{src.get('page', 0)})\n\n"
                                f"> {src.get('snippet', '')}"
                            )
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            elif st.session_state.mode == "structured":
                # ── 구조화 평가 모드 ─────────────────────────────
                # 입력 형식: "질문 | 답변"
                parts = user_input.split("|", 1)
                if len(parts) == 2:
                    question, answer = parts[0].strip(), parts[1].strip()
                else:
                    question = "자기소개를 해주세요"
                    answer = user_input

                response = httpx.post(
                    f"{BACKEND_URL}/chat/structured",
                    json={"question": question, "answer": answer},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                # 점수 카드 표시
                col1, col2 = st.columns([1, 3])
                with col1:
                    score = data.get("score", 0)
                    score_emoji = ["", "😟", "🤔", "😐", "😊", "🌟"][score]
                    st.metric("점수", f"{score}/5 {score_emoji}")
                with col2:
                    st.markdown(f"**💪 강점:** {data.get('strengths', '-')}")
                    st.markdown(f"**📝 개선점:** {data.get('improvements', '-')}")
                    st.markdown(f"**❓ 후속 질문:** {data.get('next_question', '-')}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"**점수:** {data.get('score', 0)}/5\n\n"
                        f"**강점:** {data.get('strengths', '-')}\n\n"
                        f"**개선점:** {data.get('improvements', '-')}\n\n"
                        f"**후속 질문:** {data.get('next_question', '-')}"
                    ),
                    "metadata": data,
                })

            elif st.session_state.mode == "parallel":
                # ── 병렬 모드 ────────────────────────────────────
                response = httpx.post(
                    f"{BACKEND_URL}/chat/parallel",
                    json={"message": user_input},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                st.markdown("### 💬 답변")
                st.markdown(data.get("answer", "-"))
                st.markdown("### ❓ 관련 FAQ")
                st.markdown(data.get("faq", "-"))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"**답변:** {data.get('answer', '-')}\n\n"
                        f"**FAQ:** {data.get('faq', '-')}"
                    ),
                })

        except httpx.ConnectError:
            st.error(
                "⚠️ Backend 서버에 연결할 수 없습니다.\n\n"
                f"`uvicorn backend.app:app --port 8000` 으로 서버를 먼저 시작해주세요."
            )
        except httpx.HTTPStatusError as e:
            st.error(f"⚠️ API 오류: {e.response.status_code}\n\n{e.response.text}")
        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")
