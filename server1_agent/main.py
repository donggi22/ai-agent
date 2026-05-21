import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from graph import graph
from state import DeliveryState

app = FastAPI(title="Delivery Agent Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request 스키마 ────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:         str
    order_id:      str  = "ORD-0001"
    product_code:  str  = "P-001"
    required_qty:  int  = 3000
    required_date: str  = "2025-06-30"
    workflow_type: str  = "A"   # UI에서 수동 지정 가능, 없으면 Orchestrator 판별


# ── 헬스체크 ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "server": "Agent Service"}


# ── 일반 실행 (JSON 응답) ─────────────────────────────────────
@app.post("/run")
async def run(req: QueryRequest):
    initial_state: DeliveryState = {
        "query":          req.query,
        "order_id":       req.order_id,
        "product_code":   req.product_code,
        "required_qty":   req.required_qty,
        "required_date":  req.required_date,
        "workflow_type":  req.workflow_type,
        "capa_result":    None,
        "material_result": None,
        "quality_result": None,
        "mold_result":    None,
        "verdict":        "",
        "scenario_table": [],
        "escalation_flag": False,
        "escalation_reason": None,
        "error_log":      [],
        "trajectory":     [],
    }
    final_state = await graph.ainvoke(initial_state)
    return {
        "verdict":        final_state["verdict"],
        "workflow_type":  final_state["workflow_type"],
        "scenario_table": final_state["scenario_table"],
        "escalation_flag": final_state["escalation_flag"],
        "escalation_reason": final_state.get("escalation_reason"),
        "trajectory":     final_state["trajectory"],
        "error_log":      final_state["error_log"],
    }


# ── SSE 스트리밍 (UI trajectory 실시간 표시용) ─────────────────
@app.get("/stream")
async def stream(
    query:         str = "3000개 납기 가능한지 확인해줘",
    order_id:      str = "ORD-0001",
    product_code:  str = "P-001",
    required_qty:  int = 3000,
    required_date: str = "2025-06-30",
    workflow_type: str = "A",
):
    initial_state: DeliveryState = {
        "query":          query,
        "order_id":       order_id,
        "product_code":   product_code,
        "required_qty":   required_qty,
        "required_date":  required_date,
        "workflow_type":  workflow_type,
        "capa_result":    None,
        "material_result": None,
        "quality_result": None,
        "mold_result":    None,
        "verdict":        "",
        "scenario_table": [],
        "escalation_flag": False,
        "escalation_reason": None,
        "error_log":      [],
        "trajectory":     [],
    }

    async def event_generator():
        prev_traj_len = 0
        final_state = None

        async for chunk in graph.astream(initial_state):
            # chunk = { node_name: state_dict }
            for node_name, state in chunk.items():
                trajectory = state.get("trajectory", [])
                new_entries = trajectory[prev_traj_len:]
                prev_traj_len = len(trajectory)

                for entry in new_entries:
                    yield {
                        "event": "trajectory",
                        "data": json.dumps(entry, ensure_ascii=False)
                    }

                final_state = state

        # 최종 결과 전송
        if final_state:
            yield {
                "event": "result",
                "data": json.dumps({
                    "verdict":         final_state.get("verdict", ""),
                    "workflow_type":   final_state.get("workflow_type", ""),
                    "scenario_table":  final_state.get("scenario_table", []),
                    "escalation_flag": final_state.get("escalation_flag", False),
                    "escalation_reason": final_state.get("escalation_reason"),
                    "error_log":       final_state.get("error_log", []),
                }, ensure_ascii=False)
            }

        yield {"event": "done", "data": "완료"}

    return EventSourceResponse(event_generator())
