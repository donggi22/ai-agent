from typing import TypedDict, Optional, List


class DeliveryState(TypedDict):
    # ── 입력 ──────────────────────────────────────────────────
    query:          str
    order_id:       str
    product_code:   str
    required_qty:   int
    required_date:  str
    workflow_type:  str                   # A / B / C

    # ── 하위 에이전트 결과 ─────────────────────────────────────
    capa_result:    Optional[dict]
    material_result: Optional[dict]
    quality_result: Optional[dict]
    mold_result:    Optional[dict]

    # ── 에이전트 간 체인 (이전 에이전트 출력 → 다음 에이전트 입력) ──
    agent_summaries: List[str]           # 각 에이전트 LLM 요약 누적 리스트

    # ── 최종 출력 ─────────────────────────────────────────────
    verdict:           str
    verdict_reason:    Optional[str]
    verdict_conditions: Optional[List[str]]
    scenario_table:    List[dict]
    escalation_flag:   bool
    escalation_reason: Optional[str]

    # ── 디버깅 ────────────────────────────────────────────────
    error_log:      List[dict]
    trajectory:     List[dict]
