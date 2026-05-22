import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from signals import extract_material_signals, build_chain_context
from routing import extract_intent, execute_intents

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

DEFAULT_INTENTS = ["call_material_stock"]

SYSTEM_PROMPT = """당신은 사출성형 공장의 자재구매 분석 에이전트입니다.
이전 에이전트들(오케스트레이터, 생산CAPA)의 분석을 이어받아 자재 재고 관점에서 납기 판정에 기여합니다.

[SIGNALS TABLE]이 제공될 경우 capacity_risk 등 이전 에이전트 시그널을 반드시 참조하세요.

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 오케스트레이터와 생산CAPA의 핵심 결론을 1문장으로 직접 언급
[자재 분석] MES 재고/부족/리드타임 데이터를 위 맥락에서 해석하여 2문장
[다음 에이전트 전달] 품질/금형 에이전트가 참고해야 할 자재 결론 1문장"""


async def material_agent(state: DeliveryState) -> DeliveryState:
    """자재구매 Agent: intent 추출 → MES 조회 → LLM 분석"""
    start = time.time()
    error_log = list(state.get("error_log", []))

    chain_context = build_chain_context(
        state.get("agent_summaries", []),
        state.get("agent_signals", {}),
    )

    # Step 1: LLM intent 추출
    intent_context = (
        f"워크플로: {state['workflow_type']}, "
        f"order_id={state['order_id']}, product_code={state['product_code']}\n"
        f"자재 재고 분석을 위해 어떤 MES API가 필요합니까?"
    )
    intent_result = await extract_intent(llm, intent_context)
    intents = intent_result["intents"] if intent_result["intents"] else DEFAULT_INTENTS

    # Step 2: 결정론적 MES 실행
    mes_results = await execute_intents(intents, state)
    mes_data = mes_results.get("call_material_stock", {})
    recovery = None
    if "error" in mes_data:
        recovery = {"trigger": mes_data["error"], "action": "empty mes_data로 LLM 분석 진행"}
        error_log.append({"agent": "자재구매", "api": "call_material_stock", "error": mes_data["error"]})

    # Step 3: LLM 분석
    try:
        analysis_msg = await llm.ainvoke([
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(
                f"원래 질의: \"{state['query']}\"\n\n"
                f"{chain_context}"
                f"MES 자재재고 조회 결과: {mes_data.get('data', {})}\n"
                f"요청수량: {state['required_qty']}개, 납기: {state['required_date']}\n\n"
                f"시스템 프롬프트의 형식([이전 분석 인용]/[자재 분석]/[다음 에이전트 전달])에 맞게 응답하세요."
            ),
        ])
        analysis = analysis_msg.content.strip()
    except Exception as e:
        recovery = {"trigger": str(e), "action": "분석 오류 메시지로 대체"}
        error_log.append({"agent": "자재구매", "error": str(e)})
        analysis = f"오류 발생: {e}"

    signals = extract_material_signals(mes_data, analysis)
    my_summary = f"[자재구매] {analysis}"

    new_signals = {**state.get("agent_signals", {}), "material_agent": signals}
    traj_entry = {
        "goal":   "자재 재고 조회 및 납기 영향 분석",
        "plan":   f"call_material_stock API 조회 → 재고·부족수량·리드타임 분석 (confidence={intent_result['confidence']:.2f})",
        "action": {
            "tool":   intents[0] if intents else "none",
            "params": {"order_id": state["order_id"], "product_code": state["product_code"], "workflow_type": state["workflow_type"]},
            "result": mes_data.get("data", {}),
        },
        "state": {
            "workflow_type":  state["workflow_type"],
            "order_id":       state["order_id"],
            "product_code":   state["product_code"],
            "required_qty":   state["required_qty"],
            "required_date":  state["required_date"],
            "agent_signals":  new_signals,
            "escalation_flag": state.get("escalation_flag", False),
        },
        "result":   my_summary,
        "recovery": recovery,
    }

    debug_entry = {
        "agent": "자재구매 Agent",
        "step": "자재 재고 조회",
        "mes_api": "/mes/material-stock",
        "intent_result": intent_result,
        "intents_executed": intents,
        "mes_response": mes_data.get("data", {}),
        "signals": signals,
        "received_from_previous": chain_context,
        "llm_raw_response": analysis,
        "llm_output": my_summary,
        "llm_analysis": analysis,
        "elapsed_ms": int((time.time() - start) * 1000),
    }

    return {
        **state,
        "material_result": mes_data,
        "agent_summaries": state.get("agent_summaries", []) + [my_summary],
        "agent_signals": {**state.get("agent_signals", {}), "material_agent": signals},
        "trajectory": state.get("trajectory", []) + [traj_entry],
        "debug_trace": state.get("debug_trace", []) + [debug_entry],
        "error_log": error_log,
    }
