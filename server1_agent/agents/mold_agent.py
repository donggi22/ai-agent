import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from tools.mes_tools import get_mold_setup, get_order_conflict
from tools.mes_client import call_mold_setup, call_order_conflict

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

SYSTEM_PROMPT_A_B = """당신은 사출성형 공장의 금형셋업 분석 에이전트입니다.
이전 에이전트들(오케스트레이터, 생산CAPA, 자재구매, 품질조건)의 분석을 이어받아 금형 관점에서 최종 납기를 산출합니다.

핵심 공식: 실질 양품 수량 = 요청수량 × (1 – 초기불량률)

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 앞선 에이전트들의 핵심 결론(CAPA/자재/품질)을 1문장으로 직접 언급
[금형 분석] 교체시간/불량률 데이터로 실질 양품 수량과 납기를 위 맥락에서 산출하여 2문장
[최종 전달] 오케스트레이터 판정에 전달할 금형 결론 1문장"""

SYSTEM_PROMPT_C = """당신은 사출성형 공장의 금형셋업+수주경합 분석 에이전트입니다.
이전 에이전트들의 분석을 이어받아 금형 셋업과 수주 경합 여부를 함께 분석합니다.

핵심 공식: 실질 양품 수량 = 요청수량 × (1 – 초기불량률)
수주 경합 에스컬레이션 코드(ESC-XXX) 발생 시 즉시 보고

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 앞선 에이전트들의 핵심 결론을 1문장으로 직접 언급
[금형+경합 분석] 금형 데이터와 경합 여부를 위 맥락에서 해석하여 2문장
[최종 전달] 오케스트레이터 판정에 전달할 종합 결론 1문장"""


async def mold_agent(state: DeliveryState) -> DeliveryState:
    """금형셋업 Agent: 워크플로 C에서는 수주경합 tool도 함께 function calling"""
    start = time.time()
    error_log = list(state.get("error_log", []))
    escalation_flag = state.get("escalation_flag", False)
    escalation_reason = state.get("escalation_reason", None)

    # 워크플로 C에서는 두 개 tool 제공 → LLM이 둘 다 호출할지 결정
    if state["workflow_type"] == "C":
        tools = [get_mold_setup, get_order_conflict]
        system_prompt = SYSTEM_PROMPT_C
    else:
        tools = [get_mold_setup]
        system_prompt = SYSTEM_PROMPT_A_B

    prior_context = "\n".join(state.get("agent_summaries", []))
    chain_prefix = f"=== 이전 에이전트 분석 ===\n{prior_context}\n\n" if prior_context else ""

    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(
            f"{chain_prefix}금형 셋업 정보를 조회하고 실질 납기를 분석하세요.\n"
            f"주문: order_id={state['order_id']}, product_code={state['product_code']}\n"
            f"요청수량: {state['required_qty']}개, 납기요청일: {state['required_date']}\n"
            f"워크플로: {state['workflow_type']}"
            + ("\n수주 경합 여부도 함께 확인하세요." if state["workflow_type"] == "C" else "")
        ),
    ]

    mold_data = {}
    conflict_data = {}
    tool_calls_log = []
    analysis = ""
    llm_raw_response = ""
    function_calling_worked = False

    try:
        # Step 1: LLM이 tool call 결정 (워크플로 C면 두 개 tool 모두 호출 가능)
        ai_msg = await llm_with_tools.ainvoke(messages)

        if ai_msg.tool_calls:
            function_calling_worked = True
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls:
                if tc["name"] == "get_mold_setup":
                    tool_result = await get_mold_setup.ainvoke(tc["args"])
                    mold_data = tool_result
                elif tc["name"] == "get_order_conflict":
                    tool_result = await get_order_conflict.ainvoke(tc["args"])
                    conflict_data = tool_result
                else:
                    tool_result = {"error": f"unknown tool: {tc['name']}"}

                tool_calls_log.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": tool_result,
                })
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                ))

            final_msg = await llm.ainvoke(messages)
            analysis = final_msg.content.strip()
            llm_raw_response = final_msg.content
        else:
            # 폴백: 직접 HTTP 호출
            try:
                mold_data = await call_mold_setup(state)
            except Exception as e:
                error_log.append({"agent": "금형셋업", "api": "/mes/mold-setup", "error": str(e)})

            if state["workflow_type"] == "C":
                try:
                    conflict_data = await call_order_conflict(state)
                except Exception as e:
                    error_log.append({"agent": "금형셋업", "api": "/mes/order-conflict", "error": str(e)})

            from langchain_core.messages import HumanMessage as HM
            fallback_prompt = (
                f"원래 질의: \"{state['query']}\"\n\n"
                f"{chain_prefix}"
                f"MES 금형셋업 조회 결과: mold={mold_data.get('data', {})}"
                + (f", conflict={conflict_data.get('data', {})}" if conflict_data else "")
                + f"\n요청수량: {state['required_qty']}개, 납기: {state['required_date']}\n\n"
                f"시스템 프롬프트의 형식([이전 분석 인용]/[금형 분석]/[최종 전달])에 맞게 응답하세요."
            )
            analysis_msg = await llm.ainvoke([messages[0], HM(fallback_prompt)])
            analysis = analysis_msg.content.strip()
            llm_raw_response = analysis_msg.content

    except Exception as e:
        error_log.append({"agent": "금형셋업", "error": str(e)})
        try:
            mold_data = await call_mold_setup(state)
        except Exception:
            pass
        analysis = f"오류 발생: {e}"

    # 에스컬레이션 체크
    c_data = conflict_data.get("data", {})
    if c_data.get("status") == "escalation":
        escalation_flag = True
        escalation_reason = c_data.get("recommended_action", "에스컬레이션 필요")

    # 실질 양품 수량 보정
    m_data = mold_data.get("data", {})
    effective_qty = None
    if m_data.get("initial_defect_rate") is not None:
        defect_rate = m_data["initial_defect_rate"]
        effective_qty = int(state["required_qty"] * (1 - defect_rate))

    my_summary = f"[금형셋업] {analysis}"

    entry = {
        "agent": "금형셋업 Agent",
        "step": "금형 셋업" + (" + 수주 경합 조회" if state["workflow_type"] == "C" else " 조회"),
        "mes_api": "/mes/mold-setup" + (", /mes/order-conflict" if state["workflow_type"] == "C" else ""),
        "function_calling_worked": function_calling_worked,
        "tool_calls": tool_calls_log,
        "mes_response": {"mold": m_data, "conflict": c_data},
        "received_from_previous": prior_context,
        "llm_raw_response": llm_raw_response,
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
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
