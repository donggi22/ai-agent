"""
에이전트별 signal 추출기 + 공통 프롬프트 빌더.

Signal 추출 원칙:
- 수치(risk 등급, rate, qty)는 MES 데이터에서 결정론적으로 계산
- LLM narrative 키워드는 보조 확증에만 사용
- risk 등급: low / medium / high
"""


# ── Signal 추출기 ──────────────────────────────────────────────

def extract_capa_signals(mes_data: dict, analysis: str) -> dict:
    d = mes_data.get("data", {})
    utilization = d.get("utilization_rate", 0.0)
    producible_qty = d.get("producible_qty", 0)
    status = d.get("status", "ok")

    if status == "overloaded" or "불가" in analysis or "포화" in analysis:
        risk = "high"
    elif utilization >= 0.85:
        risk = "medium"
    else:
        risk = "low"

    return {
        "capacity_risk": risk,
        "utilization": round(utilization, 2),
        "producible_qty": producible_qty,
        "bottleneck": risk == "high",
    }


def extract_material_signals(mes_data: dict, analysis: str) -> dict:
    d = mes_data.get("data", {})
    status = d.get("status", "ok")
    shortage_qty = d.get("shortage_qty", 0)
    lead_time_days = d.get("lead_time_days", 0)

    if status == "shortage" or shortage_qty > 0:
        risk = "high"
    elif lead_time_days > 7:
        risk = "medium"
    else:
        risk = "low"

    return {
        "supply_risk": risk,
        "shortage": shortage_qty > 0,
        "lead_time_days": lead_time_days,
        "stock_ok": status == "ok" and shortage_qty == 0,
    }


def extract_quality_signals(mes_data: dict, analysis: str) -> dict:
    d = mes_data.get("data", {})
    passed = d.get("pass", True)
    defect_rate = d.get("defect_rate", 0.0)

    if not passed:
        risk = "high"
    elif defect_rate > 0.05:
        risk = "medium"
    else:
        risk = "low"

    return {
        "quality_risk": risk,
        "defect_rate": round(defect_rate, 3),
        "standard_met": bool(passed),
        "inspection_required": not passed or defect_rate > 0.05,
    }


def extract_mold_signals(mes_data: dict, conflict_data: dict, analysis: str, effective_qty: int) -> dict:
    d = mes_data.get("data", {})
    c = conflict_data.get("data", {})
    exchange_time = d.get("exchange_time_hours", 0.0)
    initial_defect_rate = d.get("initial_defect_rate", 0.0)
    has_conflict = c.get("status") == "escalation"
    status = d.get("status", "ok")

    if has_conflict or status == "error":
        risk = "high"
    elif exchange_time > 4 or initial_defect_rate > 0.03:
        risk = "medium"
    else:
        risk = "low"

    return {
        "mold_risk": risk,
        "exchange_time_hours": round(exchange_time, 1),
        "initial_defect_rate": round(initial_defect_rate, 3),
        "effective_qty": effective_qty or 0,
        "conflict": has_conflict,
    }


# ── 프롬프트 빌더 ─────────────────────────────────────────────

def build_chain_context(agent_summaries: list, agent_signals: dict) -> str:
    """
    [SIGNALS TABLE] + [NARRATIVE CONTEXT] 형식으로 chain context 구성.
    다음 에이전트 프롬프트에 prepend하여 structured world model 제공.
    """
    if not agent_summaries and not agent_signals:
        return ""

    parts = []

    if agent_signals:
        lines = ["[SIGNALS TABLE]"]
        for agent_name, signals in agent_signals.items():
            sig_str = ", ".join(f"{k}={v}" for k, v in signals.items())
            lines.append(f"- {agent_name}: {sig_str}")
        parts.append("\n".join(lines))

    if agent_summaries:
        lines = ["[NARRATIVE CONTEXT]"]
        lines.extend(agent_summaries)
        parts.append("\n".join(lines))

    return "\n\n".join(parts) + "\n\n"
