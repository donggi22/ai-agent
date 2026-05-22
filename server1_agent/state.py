from typing import TypedDict, Optional, List, Dict


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

    # ── 에이전트 간 체인 ───────────────────────────────────────
    agent_summaries: List[str]           # LLM narrative 누적 (human-readable)
    agent_signals:   Dict[str, Dict]     # 구조화된 시그널 누적 (machine-readable)

    # ── 질의 유효성 ────────────────────────────────────────────
    invalid_query:  bool                   # 납기 무관 질의 감지 시 True

    # ── 최종 출력 ─────────────────────────────────────────────
    verdict:           str
    verdict_reason:    Optional[str]
    verdict_conditions: Optional[List[str]]
    scenario_table:    List[dict]
    escalation_flag:   bool
    escalation_reason: Optional[str]

    # ── 기획 Trajectory (goal/plan/action/state/result/recovery) ─
    trajectory:     List[dict]

    # ── 디버깅용 상세 로그 ────────────────────────────────────
    debug_trace:    List[dict]
    error_log:      List[dict]
