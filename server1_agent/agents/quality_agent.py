import time
from langchain_openai import ChatOpenAI
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from tools.mes_client import call_quality_condition

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=256,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 품질 조건 분석 전문가입니다.
MES/QMS에서 조회한 불량률, 품질 기준을 분석하여
현재 품질 수준이 납기에 미치는 영향을 판단합니다. 수치 기반으로 간결하게 분석하세요."""


async def quality_agent(state: DeliveryState) -> DeliveryState:
    start = time.time()

    try:
        mes_result = await call_quality_condition(state)
        data = mes_result.get("data", {})
        error = None
    except Exception as e:
        mes_result = {}
        data = {}
        error = str(e)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", (
            f"MES 품질 조건 조회 결과를 분석하세요.\n\n"
            f"요청 수량: {state['required_qty']}개\n"
            f"MES 응답: {data}\n\n"
            f"품질 기준 충족 여부와 납기 영향을 2문장으로 답하세요."
        ))
    ]
    response = await llm.ainvoke(messages)

    entry = {
        "agent": "품질조건 Agent",
        "step": "품질 조건 조회",
        "mes_api": "/mes/quality-condition",
        "input": {"required_qty": state["required_qty"], "workflow": state["workflow_type"]},
        "mes_response": data,
        "llm_analysis": response.content.strip(),
        "elapsed_ms": int((time.time() - start) * 1000),
        "error": error,
    }

    error_log = list(state.get("error_log", []))
    if error:
        error_log.append({"agent": "품질조건", "error": error})

    return {
        **state,
        "quality_result": mes_result,
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
