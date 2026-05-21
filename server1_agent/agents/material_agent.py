import time
from langchain_openai import ChatOpenAI
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from tools.mes_client import call_material_stock

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=256,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 자재구매 및 재고 분석 전문가입니다.
MES/ERP에서 조회한 자재 재고, 부족 수량, 조달 리드타임을 분석하여
납기에 미치는 영향을 판단합니다. 수치 기반으로 간결하게 분석하세요."""


async def material_agent(state: DeliveryState) -> DeliveryState:
    start = time.time()

    try:
        mes_result = await call_material_stock(state)
        data = mes_result.get("data", {})
        error = None
    except Exception as e:
        mes_result = {}
        data = {}
        error = str(e)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", (
            f"MES 자재 재고 조회 결과를 분석하세요.\n\n"
            f"요청 수량: {state['required_qty']}개\n"
            f"납기 요청일: {state['required_date']}\n"
            f"MES 응답: {data}\n\n"
            f"자재 조달 가능 여부와 납기 영향을 2문장으로 답하세요."
        ))
    ]
    response = await llm.ainvoke(messages)

    entry = {
        "agent": "자재구매 Agent",
        "step": "자재 재고 조회",
        "mes_api": "/mes/material-stock",
        "input": {"required_qty": state["required_qty"], "workflow": state["workflow_type"]},
        "mes_response": data,
        "llm_analysis": response.content.strip(),
        "elapsed_ms": int((time.time() - start) * 1000),
        "error": error,
    }

    error_log = list(state.get("error_log", []))
    if error:
        error_log.append({"agent": "자재구매", "error": error})

    return {
        **state,
        "material_result": mes_result,
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
