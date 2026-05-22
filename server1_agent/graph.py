from langgraph.graph import StateGraph, END
from state import DeliveryState
from agents.orchestrator import orchestrator_route, orchestrator_verdict
from agents.capa_agent    import capa_agent
from agents.material_agent import material_agent
from agents.quality_agent  import quality_agent
from agents.mold_agent     import mold_agent


def _route_after_orchestrator(state: DeliveryState) -> str:
    """납기 무관 질의면 하위 에이전트 전부 건너뛰고 verdict로 직행"""
    if state.get("invalid_query"):
        return "orchestrator_verdict"
    return "capa_agent"


def build_graph():
    g = StateGraph(DeliveryState)

    g.add_node("orchestrator_route",   orchestrator_route)
    g.add_node("capa_agent",           capa_agent)
    g.add_node("material_agent",       material_agent)
    g.add_node("quality_agent",        quality_agent)
    g.add_node("mold_agent",           mold_agent)
    g.add_node("orchestrator_verdict", orchestrator_verdict)

    g.set_entry_point("orchestrator_route")

    # 관련성 체크 통과 여부에 따라 분기
    g.add_conditional_edges(
        "orchestrator_route",
        _route_after_orchestrator,
        {"capa_agent": "capa_agent", "orchestrator_verdict": "orchestrator_verdict"},
    )

    g.add_edge("capa_agent",           "material_agent")
    g.add_edge("material_agent",       "quality_agent")
    g.add_edge("quality_agent",        "mold_agent")
    g.add_edge("mold_agent",           "orchestrator_verdict")
    g.add_edge("orchestrator_verdict", END)

    return g.compile()


graph = build_graph()
