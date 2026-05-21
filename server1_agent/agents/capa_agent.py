import time
from langchain_openai import ChatOpenAI
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState
from tools.mes_client import call_production_capa

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=256,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 생산 CAPA 분석 전문가입니다.
MES에서 조회한 설비 가동률, 가용시간, 생산 가능 수량을 분석하여
요청 수량의 납기 가능 여부를 판단합니다. 수치 기반으로 간결하게 분석하세요."""


async def capa_agent(state: DeliveryState) -> DeliveryState:
    start = time.time()

    # MES API 호출
    try:
        mes_result = await call_production_capa(state)
        data = mes_result.get("data", {})
        error = None
    except Exception as e:
        mes_result = {}
        data = {}
        error = str(e)

    # LLM 분석
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", (
            f"MES 생산 CAPA 조회 결과를 분석하세요.\n\n"
            f"요청 수량: {state['required_qty']}개\n"
            f"납기 요청일: {state['required_date']}\n"
            f"MES 응답: {data}\n\n"
            f"생산 가능 여부와 이유를 2문장으로 답하세요."
        ))
    ]
    response = await llm.ainvoke(messages)

    entry = {
        "agent": "생산CAPA Agent",
        "step": "생산 CAPA 조회",
        "mes_api": "/mes/production-capa",
        "input": {"required_qty": state["required_qty"], "workflow": state["workflow_type"]},
        "mes_response": data,
        "llm_analysis": response.content.strip(),
        "elapsed_ms": int((time.time() - start) * 1000),
        "error": error,
    }

    error_log = list(state.get("error_log", []))
    if error:
        error_log.append({"agent": "생산CAPA", "error": error})

    return {
        **state,
        "capa_result": mes_result,
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
