import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from signals import extract_mold_signals, build_chain_context
from routing import extract_intent, execute_intents

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

DEFAULT_INTENTS = {
    "A": ["call_mold_setup"],
    "B": ["call_mold_setup"],
    "C": ["call_mold_setup", "call_order_conflict"],
}

SYSTEM_PROMPT_A_B = """당신은 사출성형 공장의 금형셋업 분석 에이전트입니다.
이전 에이전트들(오케스트레이터, 생산CAPA, 자재구매, 품질조건)의 분석을 이어받아 금형 관점에서 최종 납기를 산출합니다.

핵심 공식: 실질 양품 수량 = 요청수량 × (1 – 초기불량률)

[SIGNALS TABLE]이 제공될 경우 capacity_risk, supply_risk, quality_risk를 모두 참조하여
금형 분석 결과가 전체 납기 리스크에 미치는 누적 영향을 평가하세요.

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 앞선 에이전트들의 핵심 결론(CAPA/자재/품질)을 1문장으로 직접 언급
[금형 분석] 교체시간/불량률 데이터로 실질 양품 수량과 납기를 위 맥락에서 산출하여 2문장
[최종 전달] 오케스트레이터 판정에 전달할 금형 결론 1문장"""

SYSTEM_PROMPT_C = """당신은 사출성형 공장의 금형셋업+수주경합 분석 에이전트입니다.
이전 에이전트들의 분석을 이어받아 금형 셋업과 수주 경합 여부를 함께 분석합니다.

핵심 공식: 실질 양품 수량 = 요청수량 × (1 – 초기불량률)

[SIGNALS TABLE]의 모든 시그널을 참조하여 경합 상황에서의 납기 리스크를 종합 평가하세요.
수주 경합 에스컬레이션 코드(ESC-XXX) 발생 시 즉시 보고

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 앞선 에이전트들의 핵심 결론을 1문장으로 직접 언급
[금형+경합 분석] 금형 데이터와 경합 여부를 위 맥락에서 해석하여 2문장
[최종 전달] 오케스트레이터 판정에 전달할 종합 결론 1문장"""


async def mold_agent(state: DeliveryState) -> DeliveryState:
    """금형셋업 Agent: intent 추출 → MES 조회 → LLM 분석"""
    start = time.time()
    error_log = list(state.get("error_log", []))
    escalation_flag = state.get("escalation_flag", False)
    escalation_reason = state.get("escalation_reason", None)

    workflow = state["workflow_type"]
    system_prompt = SYSTEM_PROMPT_C if workflow == "C" else SYSTEM_PROMPT_A_B

    chain_context = build_chain_context(
        state.get("agent_summaries", []),
        state.get("agent_signals", {}),
    )

    # Step 1: LLM intent 추출 (워크플로 C: 두 API 중 선택 여부 LLM이 결정)
    if workflow == "C":
        intent_context = (
            f"워크플로: C (수주 경합 가능성 있음)\n"
            f"order_id={state['order_id']}, product_code={state['product_code']}\n"
            f"금형 분석에 필요한 API를 선택하세요.\n"
            f"사용 가능: call_mold_setup, call_order_conflict\n"
            f"수주 경합이 의심되면 두 API 모두 선택하세요."
        )
    else:
        intent_context = (
            f"워크플로: {workflow}\n"
            f"order_id={state['order_id']}, product_code={state['product_code']}\n"
            f"금형 셋업 분석을 위해 어떤 MES API가 필요합니까?"
        )

    intent_result = await extract_intent(llm, intent_context)
    intents = intent_result["intents"] if intent_result["intents"] else DEFAULT_INTENTS[workflow]

    # Step 2: 결정론적 MES 실행
    mes_results = await execute_intents(intents, state)
    mold_data = mes_results.get("call_mold_setup", {})
    conflict_data = mes_results.get("call_order_conflict", {})

    recovery = None
    if "error" in mold_data:
        recovery = {"trigger": mold_data["error"], "action": "mold_data 없이 분석 진행"}
        error_log.append({"agent": "금형셋업", "api": "call_mold_setup", "error": mold_data["error"]})
    if conflict_data and "error" in conflict_data:
        error_log.append({"agent": "금형셋업", "api": "call_order_conflict", "error": conflict_data["error"]})

    # Step 3: LLM 분석
    try:
        mes_summary = f"mold={mold_data.get('data', {})}"
        if conflict_data:
            mes_summary += f", conflict={conflict_data.get('data', {})}"

        analysis_msg = await llm.ainvoke([
            SystemMessage(system_prompt),
            HumanMessage(
                f"원래 질의: \"{state['query']}\"\n\n"
                f"{chain_context}"
                f"MES 금형셋업 조회 결과: {mes_summary}\n"
                f"요청수량: {state['required_qty']}개, 납기: {state['required_date']}\n\n"
                f"시스템 프롬프트의 형식([이전 분석 인용]/[금형 분석]/[최종 전달])에 맞게 응답하세요."
            ),
        ])
        analysis = analysis_msg.content.strip()
    except Exception as e:
        recovery = {"trigger": str(e), "action": "분석 오류 메시지로 대체"}
        error_log.append({"agent": "금형셋업", "error": str(e)})
        analysis = f"오류 발생: {e}"

    # 에스컬레이션 체크
    c_data = conflict_data.get("data", {}) if conflict_data else {}
    if c_data.get("status") == "escalation":
        escalation_flag = True
        escalation_reason = c_data.get("recommended_action", "에스컬레이션 필요")

    # 실질 양품 수량
    m_data = mold_data.get("data", {})
    effective_qty = None
    if m_data.get("initial_defect_rate") is not None:
        defect_rate = m_data["initial_defect_rate"]
        effective_qty = int(state["required_qty"] * (1 - defect_rate))

    signals = extract_mold_signals(mold_data, conflict_data or {}, analysis, effective_qty)
    my_summary = f"[금형셋업] {analysis}"

    new_signals = {**state.get("agent_signals", {}), "mold_agent": signals}
    traj_entry = {
        "goal":   "금형 셋업 조회 및 실질 납기 산출" + (" + 수주 경합 확인" if "call_order_conflict" in intents else ""),
        "plan":   f"{', '.join(intents)} API 조회 → 교체시간·초기불량률 기반 실질 양품수량 산출 (confidence={intent_result['confidence']:.2f})",
        "action": {
            "tool":   intents,
            "params": {"order_id": state["order_id"], "product_code": state["product_code"], "workflow_type": state["workflow_type"]},
            "result": {"mold": m_data, "conflict": c_data, "effective_qty": effective_qty},
        },
        "state": {
            "workflow_type":  state["workflow_type"],
            "order_id":       state["order_id"],
            "product_code":   state["product_code"],
            "required_qty":   state["required_qty"],
            "required_date":  state["required_date"],
            "agent_signals":  new_signals,
            "escalation_flag": escalation_flag,
        },
        "result":   my_summary,
        "recovery": recovery,
    }

    debug_entry = {
        "agent": "금형셋업 Agent",
        "step": "금형 셋업" + (" + 수주 경합 조회" if "call_order_conflict" in intents else " 조회"),
        "mes_api": ", ".join(f"/mes/{i.replace('call_', '').replace('_', '-')}" for i in intents),
        "intent_result": intent_result,
        "intents_executed": intents,
        "mes_response": {"mold": m_data, "conflict": c_data},
        "signals": signals,
        "received_from_previous": chain_context,
        "llm_raw_response": analysis,
        "llm_output": my_summary,
        "llm_analysis": analysis,
        "effective_qty": effective_qty,
        "elapsed_ms": int((time.time() - start) * 1000),
    }

    return {
        **state,
        "mold_result": {**mold_data, "effective_qty": effective_qty},
        "escalation_flag": escalation_flag,
        "escalation_reason": escalation_reason,
        "agent_summaries": state.get("agent_summaries", []) + [my_summary],
        "agent_signals": {**state.get("agent_signals", {}), "mold_agent": signals},
        "trajectory": state.get("trajectory", []) + [traj_entry],
        "debug_trace": state.get("debug_trace", []) + [debug_entry],
        "error_log": error_log,
    }
