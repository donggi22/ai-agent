import time
from langchain_openai import ChatOpenAI
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 납기 판정 오케스트레이터입니다.
영업팀의 자연어 질의를 분석하여 워크플로 유형을 결정하고, 하위 에이전트 결과를 종합해 최종 납기 판정을 내립니다.

워크플로 판별 기준:
- A: 단일 품목, 자재 충분, 금형 교체 없음 → 표준 납기 확인
- B: 자재 부족 또는 금형 교체 필요 → 조건부 납기
- C: 다수 수주 경합 또는 복잡한 금형 이동 → 에스컬레이션 가능

최종 판정은 반드시 다음 중 하나로 답하십시오: 가능 / 조건부 / 불가 / 에스컬레이션"""


def _parse_workflow(text: str) -> str:
    t = text.upper()
    if "워크플로우 C" in t or "워크플로 C" in t or "경합" in t or "이동" in t:
        return "C"
    if "워크플로우 B" in t or "워크플로 B" in t or "부족" in t or "교체" in t:
        return "B"
    return "A"


async def orchestrator_route(state: DeliveryState) -> DeliveryState:
    """Step 1: 워크플로 유형 판별"""
    start = time.time()

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"다음 영업팀 질의에서 워크플로 유형(A/B/C)을 판별하세요.\n\n질의: {state['query']}\n\n워크플로 유형만 한 글자로 답하세요.")
    ]
    response = await llm.ainvoke(messages)
    raw = response.content.strip()
    workflow = _parse_workflow(raw) if raw not in ("A", "B", "C") else raw

    entry = {
        "agent": "Orchestrator",
        "step": "워크플로 판별",
        "input": state["query"],
        "output": f"워크플로 {workflow} 선택",
        "elapsed_ms": int((time.time() - start) * 1000)
    }

    return {
        **state,
        "workflow_type": workflow,
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": state.get("error_log", []),
        "scenario_table": state.get("scenario_table", []),
    }


async def orchestrator_verdict(state: DeliveryState) -> DeliveryState:
    """Step 2: 하위 에이전트 결과 종합 → 최종 판정"""
    start = time.time()

    # 에스컬레이션 플래그 확인
    if state.get("escalation_flag"):
        verdict = "에스컬레이션"
        summary = state.get("escalation_reason", "자동 판정 불가")
    else:
        # LLM에 종합 판정 요청
        sub_results = {
            "capa": state.get("capa_result"),
            "material": state.get("material_result"),
            "quality": state.get("quality_result"),
            "mold": state.get("mold_result"),
        }
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", (
                f"하위 에이전트 분석 결과를 종합하여 납기 최종 판정을 내리세요.\n\n"
                f"워크플로: {state['workflow_type']}\n"
                f"요청수량: {state['required_qty']}개\n"
                f"납기요청일: {state['required_date']}\n"
                f"에이전트 결과:\n{sub_results}\n\n"
                f"판정(가능/조건부/불가/에스컬레이션)과 이유를 2~3문장으로 답하세요."
            ))
        ]
        response = await llm.ainvoke(messages)
        summary = response.content.strip()

        if "불가" in summary:
            verdict = "불가"
        elif "조건부" in summary:
            verdict = "조건부"
        elif "에스컬레이션" in summary:
            verdict = "에스컬레이션"
        else:
            verdict = "가능"

    scenario_table = _build_scenario_table(state, verdict)

    entry = {
        "agent": "Orchestrator",
        "step": "최종 판정",
        "input": f"하위 에이전트 결과 종합 (워크플로 {state['workflow_type']})",
        "output": f"판정: {verdict}",
        "elapsed_ms": int((time.time() - start) * 1000)
    }

    return {
        **state,
        "verdict": verdict,
        "scenario_table": scenario_table,
        "trajectory": state.get("trajectory", []) + [entry],
    }


def _build_scenario_table(state: DeliveryState, verdict: str) -> list:
    rows = [{"항목": "최종 판정", "값": verdict, "비고": ""}]

    capa = state.get("capa_result", {})
    if capa:
        d = capa.get("data", {})
        rows.append({"항목": "생산 CAPA", "값": f"{d.get('producible_qty', '-')}개", "비고": d.get("note", "")})

    mat = state.get("material_result", {})
    if mat:
        d = mat.get("data", {})
        rows.append({"항목": "자재 재고", "값": d.get("status", "-"), "비고": d.get("note", "")})

    qual = state.get("quality_result", {})
    if qual:
        d = qual.get("data", {})
        rows.append({"항목": "품질 조건", "값": "충족" if d.get("pass") else "미충족", "비고": d.get("note", "")})

    mold = state.get("mold_result", {})
    if mold:
        d = mold.get("data", {})
        # 실질 양품 수량 보정
        if d.get("initial_defect_rate") is not None:
            defect_rate = d["initial_defect_rate"]
            effective_qty = int(state["required_qty"] * (1 - defect_rate))
            rows.append({
                "항목": "실질 양품 수량",
                "값": f"{effective_qty}개",
                "비고": f"초기불량률 {defect_rate*100:.1f}% 반영"
            })

    return rows
