"""
수동 intent-based function routing.

흐름:
  LLM (intent 추출) → _parse_intent() → execute_intents() → MES API

LLM은 "어떤 API가 필요한가"를 JSON으로 선언하고,
Python router가 결정론적으로 실행한다.

confidence < 0.6이면 default_intents로 폴백.
"""

import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools.mes_client import (
    call_production_capa,
    call_material_stock,
    call_quality_condition,
    call_mold_setup,
    call_order_conflict,
)

CONFIDENCE_THRESHOLD = 0.6

INTENT_SYSTEM = """당신은 MES API 의도 추출기입니다.
주어진 컨텍스트를 분석하여 호출해야 할 MES API 목록을 JSON으로만 출력하세요.
JSON 외 어떤 텍스트도 출력하지 마세요.

응답 형식 (반드시 이 형식만):
{"intents": ["<api_name>", ...], "confidence": <0.0~1.0>}

사용 가능한 api_name:
- call_production_capa   : 설비 가동률·생산가능수량 조회
- call_material_stock    : 자재 재고·부족·리드타임 조회
- call_quality_condition : 불량률·품질기준 충족 여부 조회
- call_mold_setup        : 금형 교체시간·초기불량률 조회
- call_order_conflict    : 수주 경합·에스컬레이션 조회 (워크플로 C 전용)"""

# API name → 실행 함수
MES_EXECUTORS = {
    "call_production_capa":   call_production_capa,
    "call_material_stock":    call_material_stock,
    "call_quality_condition": call_quality_condition,
    "call_mold_setup":        call_mold_setup,
    "call_order_conflict":    call_order_conflict,
}


def _parse_intent(text: str) -> dict | None:
    """LLM 출력에서 JSON 파싱. 실패하면 None."""
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group())
        if "intents" in result and isinstance(result["intents"], list):
            return result
    except (json.JSONDecodeError, KeyError):
        pass
    return None


async def extract_intent(llm: ChatOpenAI, context: str) -> dict:
    """
    LLM에게 필요한 MES API를 JSON으로 추출.
    confidence < CONFIDENCE_THRESHOLD 이면 intents=[] 반환 → 호출자가 default 적용.
    """
    try:
        result = await llm.ainvoke([
            SystemMessage(INTENT_SYSTEM),
            HumanMessage(context),
        ])
        raw = result.content
        parsed = _parse_intent(raw)

        if parsed is None:
            return {"intents": [], "confidence": 0.0, "raw": raw, "method": "parse_failed"}

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < CONFIDENCE_THRESHOLD:
            return {"intents": [], "confidence": confidence, "raw": raw, "method": "low_confidence"}

        return {
            "intents": parsed["intents"],
            "confidence": confidence,
            "raw": raw,
            "method": "llm_intent",
        }
    except Exception as e:
        return {"intents": [], "confidence": 0.0, "raw": str(e), "method": "error"}


async def execute_intents(intents: list[str], state: dict) -> dict[str, dict]:
    """
    intent 목록에 따라 MES API 호출.
    반환: {"call_production_capa": {mes_response}, ...}
    """
    results = {}
    for intent in intents:
        fn = MES_EXECUTORS.get(intent)
        if fn is None:
            results[intent] = {"error": f"unknown intent: {intent}"}
            continue
        try:
            results[intent] = await fn(state)
        except Exception as e:
            results[intent] = {"error": str(e)}
    return results
