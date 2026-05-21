import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from tools.mes_tools import get_quality_condition
from tools.mes_client import call_quality_condition

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 품질조건 분석 에이전트입니다.
이전 에이전트들(오케스트레이터, 생산CAPA, 자재구매)의 분석을 이어받아 품질 관점에서 납기 판정에 기여합니다.

응답 시 반드시 아래 형식을 따르세요:
[이전 분석 인용] 앞선 에이전트들의 핵심 결론을 1문장으로 직접 언급
[품질 분석] MES 불량률/기준 데이터를 위 맥락에서 해석하여 2문장
[다음 에이전트 전달] 금형 에이전트가 참고해야 할 품질 결론 1문장"""


async def quality_agent(state: DeliveryState) -> DeliveryState:
    """품질조건 Agent: function calling으로 MES 조회 후 분석"""
    start = time.time()
    error_log = list(state.get("error_log", []))

    prior_context = "\n".join(state.get("agent_summaries", []))
    chain_prefix = f"=== 이전 에이전트 분석 ===\n{prior_context}\n\n" if prior_context else ""

    llm_with_tools = llm.bind_tools([get_quality_condition])
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"{chain_prefix}품질 조건을 조회하고 납기 영향을 분석하세요.\n"
            f"주문: order_id={state['order_id']}, product_code={state['product_code']}\n"
            f"요청수량: {state['required_qty']}개, 납기요청일: {state['required_date']}\n"
            f"워크플로: {state['workflow_type']}"
        ),
    ]

    mes_data = {}
    tool_calls_log = []
    analysis = ""
    llm_raw_response = ""
    function_calling_worked = False

    try:
        ai_msg = await llm_with_tools.ainvoke(messages)
        llm_raw_response = ai_msg.content

        if ai_msg.tool_calls:
            function_calling_worked = True
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls:
                tool_result = await get_quality_condition.ainvoke(tc["args"])
                mes_data = tool_result
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
            mes_data = await call_quality_condition(state)
            from langchain_core.messages import HumanMessage as HM
            analysis_msg = await llm.ainvoke([
                messages[0],
                HM(
                    f"원래 질의: \"{state['query']}\"\n\n"
                    f"{chain_prefix}"
                    f"MES 품질조건 조회 결과: {mes_data.get('data', {})}\n"
                    f"요청수량: {state['required_qty']}개, 납기: {state['required_date']}\n\n"
                    f"시스템 프롬프트의 형식([이전 분석 인용]/[품질 분석]/[다음 에이전트 전달])에 맞게 응답하세요."
                )
            ])
            analysis = analysis_msg.content.strip()
            llm_raw_response = analysis_msg.content

    except Exception as e:
        error_log.append({"agent": "품질조건", "error": str(e)})
        try:
            mes_data = await call_quality_condition(state)
        except Exception as e2:
            error_log.append({"agent": "품질조건", "fallback_error": str(e2)})
        analysis = f"오류 발생: {e}"

    my_summary = f"[품질조건] {analysis}"

    entry = {
        "agent": "품질조건 Agent",
        "step": "품질 조건 조회",
        "mes_api": "/mes/quality-condition",
        "function_calling_worked": function_calling_worked,
        "tool_calls": tool_calls_log,
        "mes_response": mes_data.get("data", {}),
        "received_from_previous": prior_context,
        "llm_prompt": messages[-1].content if messages else "",
        "llm_raw_response": llm_raw_response,
        "llm_output": my_summary,
        "llm_analysis": analysis,
        "elapsed_ms": int((time.time() - start) * 1000),
    }

    return {
        **state,
        "quality_result": mes_data,
        "agent_summaries": state.get("agent_summaries", []) + [my_summary],
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
