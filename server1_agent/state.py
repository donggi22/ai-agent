from typing import TypedDict, Optional, List, Any


class DeliveryState(TypedDict):
    # ── 입력 ──────────────────────────────────────────────────
    query:          str                   # 영업팀 자연어 입력
    order_id:       str
    product_code:   str
    required_qty:   int
    required_date:  str
    workflow_type:  str                   # A / B / C (Orchestrator 판별)

    # ── 하위 에이전트 결과 ─────────────────────────────────────
    capa_result:    Optional[dict]
    material_result: Optional[dict]
    quality_result: Optional[dict]
    mold_result:    Optional[dict]

    # ── 최종 출력 ─────────────────────────────────────────────
    verdict:        str                   # 가능 / 조건부 / 불가 / 에스컬레이션
    scenario_table: List[dict]            # 시나리오 비교표
    escalation_flag: bool
    escalation_reason: Optional[str]

    # ── 디버깅 ────────────────────────────────────────────────
    error_log:      List[dict]
    trajectory:     List[dict]            # UI 시각화용 실행 로그
