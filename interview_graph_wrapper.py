"""interview_graph_wrapper.py — 면접 코치 LangGraph 단일 진입점 (Day 4 self2)

수정 3곳:
    ① signature: run_interview_graph(question, thread_id)
    ② config:    {"configurable": {"thread_id": thread_id}}
    ③ output:    answer/sources 두 key 보존
graph 본체는 interview_rag_graph.py에서 그대로 import (배선 무수정).
"""

from interview_rag_graph import interview_app


def run_interview_graph(question: str, thread_id: str) -> dict:
    """면접 코치 LangGraph workflow 단일 진입점.

    - thread_id 규약: interview:{이름}:{회차} (예: interview:alice:1)
    - thread_id는 config["configurable"]에만 (state/body에 넣기 금지)
    - 반환 dict에 answer/sources 두 key 보존 (Day 5 출처 UX 입력)
    """
    config = {"configurable": {"thread_id": thread_id}}          # ② config

    # 변경분(질문)만 전달 — 나머지 state는 checkpoint에서 이어받음.
    # (첫 호출이면 노드들이 .get 기본값으로 시작하므로 안전)
    result = interview_app.invoke({"question": question}, config)

    return {                                                     # ③ output mapping
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "thread_id": thread_id,
        "attempts": result.get("attempts", 0),
        "route_log": result.get("route_log", []),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("[thread evidence] interview:alice:1 ×2  +  interview:bob:1 ×1")
    print("=" * 55)

    # ① alice 1회차
    a1 = run_interview_graph("자기소개 피드백 주세요", "interview:alice:1")
    print(f"[alice:1 #1] route={[e['decision'] for e in a1['route_log']]} | sources={len(a1['sources'])}건")
    print(f"[alice:1 #1] answer={a1['answer'][:60]} ...")

    # ② alice 2회차 — 같은 슬롯 (맥락 유지)
    a2 = run_interview_graph("방금 답변의 강점을 한 가지 더 보완해 주세요", "interview:alice:1")
    print(f"[alice:1 #2] route={[e['decision'] for e in a2['route_log']]} | sources={len(a2['sources'])}건")
    print(f"[alice:1 #2] answer={a2['answer'][:60]} ...")

    # ③ bob 1회차 — 다른 슬롯 (분리)
    b1 = run_interview_graph("자기소개 피드백 주세요", "interview:bob:1")
    print(f"[bob:1   #1] route={[e['decision'] for e in b1['route_log']]} | sources={len(b1['sources'])}건")
    print(f"[bob:1   #1] answer={b1['answer'][:60]} ...")

    # ── 슬롯 직접 확인 (이어짐 vs 분리) ──
    print("\n" + "=" * 55)
    print("[snapshot] 슬롯 직접 확인")
    print("=" * 55)
    snap_alice = interview_app.get_state({"configurable": {"thread_id": "interview:alice:1"}})
    snap_bob   = interview_app.get_state({"configurable": {"thread_id": "interview:bob:1"}})
    print("[alice:1] 최신 질문 :", snap_alice.values["question"])              # 2회차 질문
    print("[alice:1] route_log :", len(snap_alice.values["route_log"]), "건")  # 누적 → 2건
    print("[bob:1  ] 최신 질문 :", snap_bob.values["question"])                # bob 1회차 질문
    print("[bob:1  ] route_log :", len(snap_bob.values["route_log"]), "건")    # 독립 → 1건