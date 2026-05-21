from langgraph.graph import StateGraph, END
from state import DeliveryState
from agents.orchestrator import orchestrator_route, orchestrator_verdict
from agents.capa_agent   import capa_agent
from agents.material_agent import material_agent
from agents.quality_agent  import quality_agent
from agents.mold_agent     import mold_agent


# ── 조건부 라우팅 함수 ────────────────────────────────────────

def route_after_orchestrator(state: DeliveryState) -> str:
    """워크플로 유형에 따라 분기"""
    wf = state.get("workflow_type", "A")
    if wf == "A":
        return "workflow_a"
    elif wf == "B":
        return "workflow_b"
    else:
        return "workflow_c"


def check_escalation(state: DeliveryState) -> str:
    """에스컬레이션 여부 체크 → verdict로 이동"""
    return "verdict"


# ── 워크플로별 병렬 실행 래퍼 ─────────────────────────────────
# LangGraph는 순차 실행이 기본이므로, 병렬 느낌을 위해
# 각 워크플로별로 필요한 에이전트만 순서대로 실행

async def run_workflow_a(state: DeliveryState) -> DeliveryState:
    """A: 생산CAPA + 품질조건만"""
    state = await capa_agent(state)
    state = await quality_agent(state)
    return state


async def run_workflow_b(state: DeliveryState) -> DeliveryState:
    """B: 생산CAPA + 자재구매 + 금형셋업"""
    state = await capa_agent(state)
    state = await material_agent(state)
    state = await mold_agent(state)
    return state


async def run_workflow_c(state: DeliveryState) -> DeliveryState:
    """C: 전체 5개 에이전트"""
    state = await capa_agent(state)
    state = await material_agent(state)
    state = await quality_agent(state)
    state = await mold_agent(state)   # 내부에서 order_conflict도 호출
    return state


# ── 그래프 빌드 ───────────────────────────────────────────────

def build_graph():
    g = StateGraph(DeliveryState)

    # 노드 등록
    g.add_node("orchestrator_route",   orchestrator_route)
    g.add_node("workflow_a",           run_workflow_a)
    g.add_node("workflow_b",           run_workflow_b)
    g.add_node("workflow_c",           run_workflow_c)
    g.add_node("orchestrator_verdict", orchestrator_verdict)

    # 엣지
    g.set_entry_point("orchestrator_route")

    g.add_conditional_edges(
        "orchestrator_route",
        route_after_orchestrator,
        {
            "workflow_a": "workflow_a",
            "workflow_b": "workflow_b",
            "workflow_c": "workflow_c",
        }
    )

    g.add_edge("workflow_a", "orchestrator_verdict")
    g.add_edge("workflow_b", "orchestrator_verdict")
    g.add_edge("workflow_c", "orchestrator_verdict")
    g.add_edge("orchestrator_verdict", END)

    return g.compile()


graph = build_graph()
