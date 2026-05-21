from langgraph.graph import StateGraph, END
from state import DeliveryState
from agents.orchestrator import orchestrator_route, orchestrator_verdict
from agents.capa_agent    import capa_agent
from agents.material_agent import material_agent
from agents.quality_agent  import quality_agent
from agents.mold_agent     import mold_agent


def build_graph():
    g = StateGraph(DeliveryState)

    # 오케스트레이터 + 4개 하위 에이전트 항상 순차 실행
    g.add_node("orchestrator_route",   orchestrator_route)
    g.add_node("capa_agent",           capa_agent)
    g.add_node("material_agent",       material_agent)
    g.add_node("quality_agent",        quality_agent)
    g.add_node("mold_agent",           mold_agent)
    g.add_node("orchestrator_verdict", orchestrator_verdict)

    g.set_entry_point("orchestrator_route")
    g.add_edge("orchestrator_route",   "capa_agent")
    g.add_edge("capa_agent",           "material_agent")
    g.add_edge("material_agent",       "quality_agent")
    g.add_edge("quality_agent",        "mold_agent")
    g.add_edge("mold_agent",           "orchestrator_verdict")
    g.add_edge("orchestrator_verdict", END)

    return g.compile()


graph = build_graph()
