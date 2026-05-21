import time
import httpx
from langchain_openai import ChatOpenAI
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY, MES_BASE_URL
from state import DeliveryState
from tools.mes_client import call_mold_setup, call_order_conflict

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
)

SYSTEM_PROMPT = """당신은 사출성형 공장의 금형 셋업 분석 전문가입니다.
금형 교체시간, 안정화 샷 이력, 초기불량률을 기반으로 현실적 납기를 산출합니다.

핵심 공식:
- 실질 양품 수량 = 총 생산수량 × (1 – 초기불량률)
- 실제 납기 = 요청수량 ÷ 실질 양품 수량 기준으로 재산출

수치 기반으로 실질 납기를 계산하여 제시하세요."""


async def mold_agent(state: DeliveryState) -> DeliveryState:
    start = time.time()
    error_log = list(state.get("error_log", []))
    escalation_flag = state.get("escalation_flag", False)
    escalation_reason = state.get("escalation_reason", None)

    # 금형 셋업 조회
    mold_error = None
    mold_result = {}
    try:
        mold_result = await call_mold_setup(state)
        mold_data = mold_result.get("data", {})
    except httpx.HTTPStatusError as e:
        mold_data = {}
        mold_error = f"HTTP {e.response.status_code}: {e.response.text}"
        error_log.append({"agent": "금형셋업", "api": "/mes/mold-setup", "error": mold_error})
    except Exception as e:
        mold_data = {}
        mold_error = str(e)
        error_log.append({"agent": "금형셋업", "api": "/mes/mold-setup", "error": mold_error})

    # 수주 경합 조회 (워크플로 C)
    conflict_result = {}
    if state["workflow_type"] == "C":
        try:
            conflict_result = await call_order_conflict(state)
            conflict_data = conflict_result.get("data", {})
            if conflict_data.get("status") == "escalation":
                escalation_flag = True
                escalation_reason = conflict_data.get("recommended_action", "에스컬레이션 필요")
        except Exception as e:
            error_log.append({"agent": "금형셋업", "api": "/mes/order-conflict", "error": str(e)})

    # 실질 양품 수량 보정 계산
    effective_qty = None
    if mold_data.get("initial_defect_rate") is not None:
        defect_rate = mold_data["initial_defect_rate"]
        effective_qty = int(state["required_qty"] * (1 - defect_rate))

    # LLM 분석
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", (
            f"금형 셋업 분석 결과를 검토하세요.\n\n"
            f"요청 수량: {state['required_qty']}개\n"
            f"납기 요청일: {state['required_date']}\n"
            f"금형 셋업 MES 응답: {mold_data}\n"
            f"금형 오류: {mold_error or '없음'}\n"
            f"수주 경합: {conflict_result.get('data', {})}\n"
            f"실질 양품 수량 (보정): {effective_qty}개\n\n"
            f"실질 납기 가능 여부와 보정 수량 근거를 2~3문장으로 답하세요."
        ))
    ]
    response = await llm.ainvoke(messages)

    entry = {
        "agent": "금형셋업 Agent",
        "step": "금형 셋업 + 수주 경합 조회",
        "mes_api": "/mes/mold-setup, /mes/order-conflict",
        "input": {"required_qty": state["required_qty"], "workflow": state["workflow_type"]},
        "mes_response": {"mold": mold_data, "conflict": conflict_result.get("data", {})},
        "effective_qty": effective_qty,
        "llm_analysis": response.content.strip(),
        "elapsed_ms": int((time.time() - start) * 1000),
        "error": mold_error,
    }

    return {
        **state,
        "mold_result": {**mold_result, "effective_qty": effective_qty},
        "escalation_flag": escalation_flag,
        "escalation_reason": escalation_reason,
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": error_log,
    }
