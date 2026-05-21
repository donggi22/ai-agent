import time
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from state import DeliveryState

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=3000,
)


# ── Structured Output 스키마 ─────────────────────────────────

class WorkflowDecision(BaseModel):
    workflow_type: str = Field(
        description="워크플로 유형. A=표준납기확인, B=자재부족+금형교체, C=다수수주경합+금형이동"
    )
    reasoning: str = Field(description="워크플로 판별 근거 한 문장")


class VerdictOutput(BaseModel):
    verdict: str = Field(description="납기 판정: 가능, 조건부, 불가, 에스컬레이션")
    reason: str = Field(description="판정 근거 2~3문장")
    conditions: list[str] = Field(
        default_factory=list,
        description="조건부 판정 시 필요 조건 목록. 가능/불가/에스컬레이션이면 빈 리스트"
    )


ROUTE_SYSTEM = """당신은 사출성형 공장 납기 판정 오케스트레이터입니다.
영업팀 자연어 질의를 분석하여 워크플로 유형을 결정합니다.

워크플로 판별 기준:
- A: 단일 품목, 자재 충분, 금형 교체 없음 → 표준 납기 확인
- B: 자재 부족 또는 금형 교체 필요 → 조건부 납기
- C: 다수 수주 경합 또는 복잡한 금형 이동 → 에스컬레이션 가능"""

VERDICT_SYSTEM = """당신은 사출성형 공장 납기 판정 오케스트레이터입니다.
하위 에이전트(생산CAPA/자재구매/품질조건/금형셋업)의 분석 결과를 종합하여
최종 납기 판정을 내립니다.

판정 기준:
- 가능: 모든 조건 충족, 납기일 내 생산 가능
- 조건부: 일부 조건 미충족, 납기 조정 또는 추가 조치 시 가능
- 불가: 핵심 조건 미충족, 납기 불가
- 에스컬레이션: 자동 판정 불가, 인간 승인 필요

반드시 JSON 스키마에 맞게 간결하게 응답하세요. reason은 2문장 이내로."""


async def orchestrator_route(state: DeliveryState) -> DeliveryState:
    """Step 1: 자연어 질의 → 워크플로 유형 판별 (plain text LLM + 키워드 파싱)"""
    start = time.time()
    manual_wf = state.get("workflow_type") if state.get("workflow_type") in ("A", "B", "C") else None

    hint = f"\n\n(참고: 사용자가 워크플로 {manual_wf}를 선택했습니다.)" if manual_wf else ""

    try:
        result = await llm.ainvoke([
            SystemMessage(ROUTE_SYSTEM),
            HumanMessage(
                f"다음 질의를 분석하고 워크플로 유형(A/B/C)을 판별하세요. "
                f"어떤 질의든 납기 판정 맥락으로 해석하여 분석하세요.\n\n질의: {state['query']}{hint}"
            ),
        ])
        llm_raw = result.content.strip()

        # plain text에서 A/B/C 추출
        import re
        m = re.search(r'\b([ABC])\b', llm_raw)
        workflow = m.group(1) if m else (manual_wf or "A")
        if manual_wf:
            workflow = manual_wf  # 수동 선택 우선
        reasoning = llm_raw
        method = "llm_text"
    except Exception as e:
        llm_raw = f"LLM 오류: {e}"
        workflow = manual_wf or _fallback_parse_workflow(state["query"])
        reasoning = f"키워드 기반 판별 (오류: {e})"
        method = "fallback_keyword"

    orch_summary = f"[Orchestrator] 워크플로 {workflow} 판별. {reasoning[:100]}"

    entry = {
        "agent": "Orchestrator",
        "step": "워크플로 판별",
        "method": method,
        "received_context": f"질의: {state['query']}",
        "llm_raw_response": llm_raw,
        "llm_output": orch_summary,
        "elapsed_ms": int((time.time() - start) * 1000),
    }

    return {
        **state,
        "workflow_type": workflow,
        "agent_summaries": [orch_summary],
        "trajectory": state.get("trajectory", []) + [entry],
        "error_log": state.get("error_log", []),
        "scenario_table": state.get("scenario_table", []),
    }


async def orchestrator_verdict(state: DeliveryState) -> DeliveryState:
    """Step 2: 하위 에이전트 결과 종합 → 최종 판정 (plain text LLM + 키워드 파싱)"""
    start = time.time()

    summaries = state.get("agent_summaries", [])
    chain = "\n".join(s[:150] for s in summaries)

    if state.get("escalation_flag"):
        verdict = "에스컬레이션"
        reason = state.get("escalation_reason", "자동 판정 불가 - 인간 승인 필요")
        conditions = []
        llm_raw = ""
        method = "escalation_flag"
    else:
        prompt = (
            f"하위 에이전트들이 순서대로 분석한 결과를 종합하여 최종 납기 판정을 내리세요.\n\n"
            f"=== 에이전트 분석 체인 ===\n{chain}\n\n"
            f"워크플로: {state['workflow_type']}, "
            f"요청수량: {state['required_qty']}개, 납기: {state['required_date']}\n\n"
            f"답변 형식:\n"
            f"판정: <가능/조건부/불가/에스컬레이션>\n"
            f"근거: <2문장 이내>\n"
            f"조건: <조건부일 경우 필요 조건, 없으면 없음>"
        )
        try:
            result = await llm.ainvoke([
                SystemMessage(VERDICT_SYSTEM),
                HumanMessage(prompt),
            ])
            llm_raw = result.content.strip()
            verdict, reason, conditions = _parse_verdict_text(llm_raw, state)
            method = "llm_text"
        except Exception as e:
            llm_raw = f"LLM 호출 오류: {e}"
            sub_results = {
                "capa": state.get("capa_result"),
                "material": state.get("material_result"),
                "quality": state.get("quality_result"),
                "mold": state.get("mold_result"),
            }
            verdict, reason, conditions = _fallback_verdict(state, sub_results)
            method = "fallback_rule"

    scenario_table = _build_scenario_table(state, verdict)

    entry = {
        "agent": "Orchestrator",
        "step": "최종 판정",
        "method": method,
        "received_context": chain,
        "llm_raw_response": llm_raw if not state.get("escalation_flag") else "",
        "llm_output": f"판정: {verdict} — {reason}",
        "verdict": verdict,
        "conditions": conditions,
        "elapsed_ms": int((time.time() - start) * 1000),
    }

    return {
        **state,
        "verdict": verdict,
        "verdict_reason": reason,
        "verdict_conditions": conditions,
        "scenario_table": scenario_table,
        "trajectory": state.get("trajectory", []) + [entry],
    }


def _parse_verdict_text(text: str, state) -> tuple[str, str, list]:
    """LLM plain text 응답에서 판정/근거/조건 파싱"""
    # 판정 키워드 추출 (순서 중요: 불가 > 에스컬레이션 > 조건부 > 가능)
    verdict = "조건부"
    for v in ["불가", "에스컬레이션", "조건부", "가능"]:
        if v in text:
            verdict = v
            break

    # 근거 추출
    reason = text.strip()
    import re
    m = re.search(r'근거[:\s]+(.+?)(?:\n조건|\n판정|$)', text, re.DOTALL)
    if m:
        reason = m.group(1).strip()[:300]

    # 조건 추출
    conditions = []
    m2 = re.search(r'조건[:\s]+(.+?)$', text, re.DOTALL)
    if m2:
        cond_text = m2.group(1).strip()
        if cond_text and cond_text != "없음":
            conditions = [c.strip() for c in re.split(r'[\n,·•]', cond_text) if c.strip() and c.strip() != "없음"]

    return verdict, reason, conditions


def _fallback_parse_workflow(query: str) -> str:
    t = query.upper()
    if any(k in t for k in ("경합", "이동", "워크플로 C", "워크플로우 C")):
        return "C"
    if any(k in t for k in ("부족", "교체", "워크플로 B", "워크플로우 B")):
        return "B"
    return "A"


def _fallback_verdict(state: DeliveryState, sub_results: dict) -> tuple[str, str, list]:
    capa_data = (sub_results.get("capa") or {}).get("data", {})
    mat_data  = (sub_results.get("material") or {}).get("data", {})
    qual_data = (sub_results.get("quality") or {}).get("data", {})

    if capa_data.get("status") == "overloaded":
        return "불가", "설비 포화로 납기 내 생산 불가능합니다.", []
    if mat_data.get("status") == "shortage":
        days = mat_data.get("lead_time_days", 0)
        return "조건부", f"자재 부족으로 {days}일 조달 기간 필요합니다.", [f"자재 {days}일 내 조달 완료"]
    if qual_data.get("pass") is False:
        return "조건부", "품질 기준 미충족으로 추가 검토가 필요합니다.", ["품질 기준 재검토"]
    return "가능", "생산 CAPA, 자재, 품질 조건 모두 충족합니다.", []


def _build_scenario_table(state: DeliveryState, verdict: str) -> list:
    rows = [{"항목": "최종 판정", "값": verdict, "비고": ""}]

    capa = state.get("capa_result", {})
    if capa:
        d = capa.get("data", {})
        rows.append({
            "항목": "생산 CAPA",
            "값": f"{d.get('producible_qty', '-')}개",
            "비고": d.get("note", ""),
        })

    mat = state.get("material_result", {})
    if mat:
        d = mat.get("data", {})
        rows.append({
            "항목": "자재 재고",
            "값": d.get("status", "-"),
            "비고": d.get("note", ""),
        })

    qual = state.get("quality_result", {})
    if qual:
        d = qual.get("data", {})
        rows.append({
            "항목": "품질 조건",
            "값": "충족" if d.get("pass") else "미충족",
            "비고": d.get("note", ""),
        })

    mold = state.get("mold_result", {})
    if mold:
        d = mold.get("data", {})
        if d.get("initial_defect_rate") is not None:
            defect_rate = d["initial_defect_rate"]
            effective_qty = int(state["required_qty"] * (1 - defect_rate))
            rows.append({
                "항목": "실질 양품 수량",
                "값": f"{effective_qty}개",
                "비고": f"초기불량률 {defect_rate * 100:.1f}% 반영",
            })

    return rows
