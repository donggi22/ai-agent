import json
import re
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph import graph
from state import DeliveryState
from config import VLLM_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from tools.mes_tools import get_production_capa

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
    required_date: str  = "2026-06-30"
    workflow_type: str  = "A"   # UI에서 수동 지정 가능, 없으면 Orchestrator 판별


# ── LLM 디버그: LLM이 살아있는지, 뭘 보내고 뭘 받는지 직접 확인 ──
@app.get("/debug/llm")
async def debug_llm(prompt: str = "사출성형 공장에서 납기 판정이 무엇인지 한 문장으로 설명해줘."):
    """LLM 호출 원문 I/O를 직접 반환. LLM이 살아있는지, 실제로 뭘 생성하는지 확인용."""
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=512,
    )
    messages = [
        SystemMessage("당신은 사출성형 공장의 납기 판정 AI입니다."),
        HumanMessage(prompt),
    ]
    start = time.time()
    try:
        response = await llm.ainvoke(messages)
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "vllm_url": VLLM_BASE_URL,
            "prompt_sent": prompt,
            "llm_response": response.content,
            "tool_calls_in_response": response.tool_calls if hasattr(response, "tool_calls") else [],
            "elapsed_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        return {
            "status": "error",
            "model": MODEL_NAME,
            "vllm_url": VLLM_BASE_URL,
            "error": str(e),
            "elapsed_ms": int((time.time() - start) * 1000),
        }


@app.get("/debug/tool-call")
async def debug_tool_call():
    """LLM이 실제로 MES tool을 호출하는지 확인. tool_calls 필드가 채워지면 function calling 동작."""
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=512,
    )
    llm_with_tool = llm.bind_tools([get_production_capa])
    messages = [
        SystemMessage("당신은 사출성형 공장 생산 CAPA 분석 전문가입니다. get_production_capa 도구를 사용하여 MES를 조회하세요."),
        HumanMessage("order_id=ORD-0001, product_code=PA-2041, required_qty=3000, required_date=2026-06-30, workflow_type=A 조건으로 생산 CAPA를 조회해줘."),
    ]
    start = time.time()
    try:
        ai_msg = await llm_with_tool.ainvoke(messages)
        return {
            "status": "ok",
            "function_calling_worked": bool(ai_msg.tool_calls),
            "tool_calls": ai_msg.tool_calls,        # LLM이 호출하기로 결정한 tool 목록
            "llm_text_response": ai_msg.content,    # tool call 없을 때 LLM이 그냥 생성한 텍스트
            "elapsed_ms": int((time.time() - start) * 1000),
            "diagnosis": (
                "✅ LLM이 tool call을 생성함 → function calling 동작"
                if ai_msg.tool_calls
                else "❌ tool_calls 비어있음 → LLM이 tool을 호출하지 않음 (모델 또는 파서 문제)"
            ),
        }
    except Exception as e:
        return {
            "status": "error",
            "function_calling_worked": False,
            "error": str(e),
            "elapsed_ms": int((time.time() - start) * 1000),
        }


# ── 헬스체크 ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "server": "Agent Service"}


# ── 일반 실행 (JSON 응답) ─────────────────────────────────────
@app.post("/run")
async def run(req: QueryRequest):
    initial_state: DeliveryState = {
        "query":              req.query,
        "order_id":           req.order_id,
        "product_code":       req.product_code,
        "required_qty":       req.required_qty,
        "required_date":      req.required_date,
        "workflow_type":      req.workflow_type,
        "capa_result":        None,
        "material_result":    None,
        "quality_result":     None,
        "mold_result":        None,
        "agent_summaries":    [],
        "agent_signals":      {},
        "debug_trace":        [],
        "verdict":            "",
        "verdict_reason":     None,
        "verdict_conditions": [],
        "scenario_table":     [],
        "invalid_query":      False,
        "escalation_flag":    False,
        "escalation_reason":  None,
        "error_log":          [],
        "trajectory":         [],
    }
    final_state = await graph.ainvoke(initial_state)
    return {
        "verdict":            final_state["verdict"],
        "verdict_reason":     final_state.get("verdict_reason"),
        "verdict_conditions": final_state.get("verdict_conditions", []),
        "invalid_query":      final_state.get("invalid_query", False),
        "workflow_type":      final_state["workflow_type"],
        "scenario_table":     final_state["scenario_table"],
        "escalation_flag":    final_state["escalation_flag"],
        "escalation_reason":  final_state.get("escalation_reason"),
        "agent_summaries":    final_state.get("agent_summaries", []),
        "agent_signals":      final_state.get("agent_signals", {}),
        "trajectory":         final_state.get("trajectory", []),
        "debug_trace":        final_state.get("debug_trace", []),
        "error_log":          final_state["error_log"],
    }


# ── SSE 스트리밍 (UI trajectory 실시간 표시용) ─────────────────
@app.get("/stream")
async def stream(
    query:         str = "3000개 납기 가능한지 확인해줘",
    order_id:      str = "ORD-0001",
    product_code:  str = "P-001",
    required_qty:  int = 3000,
    required_date: str = "2026-06-30",
    workflow_type: str = "A",
):
    initial_state: DeliveryState = {
        "query":              query,
        "order_id":           order_id,
        "product_code":       product_code,
        "required_qty":       required_qty,
        "required_date":      required_date,
        "workflow_type":      workflow_type,
        "capa_result":        None,
        "material_result":    None,
        "quality_result":     None,
        "mold_result":        None,
        "agent_summaries":    [],
        "agent_signals":      {},
        "debug_trace":        [],
        "verdict":            "",
        "verdict_reason":     None,
        "verdict_conditions": [],
        "scenario_table":     [],
        "invalid_query":      False,
        "escalation_flag":    False,
        "escalation_reason":  None,
        "error_log":          [],
        "trajectory":         [],
    }

    async def event_generator():
        prev_traj_len = 0
        prev_debug_len = 0
        final_state = None

        async for chunk in graph.astream(initial_state):
            for node_name, state in chunk.items():
                # clean schema trajectory
                trajectory = state.get("trajectory", [])
                for entry in trajectory[prev_traj_len:]:
                    yield {"event": "trajectory", "data": json.dumps(entry, ensure_ascii=False)}
                prev_traj_len = len(trajectory)

                # 상세 debug_trace
                debug_trace = state.get("debug_trace", [])
                for entry in debug_trace[prev_debug_len:]:
                    yield {"event": "debug_trace", "data": json.dumps(entry, ensure_ascii=False)}
                prev_debug_len = len(debug_trace)

                final_state = state

        # 최종 결과 전송
        if final_state:
            yield {
                "event": "result",
                "data": json.dumps({
                    "verdict":            final_state.get("verdict", ""),
                    "verdict_reason":     final_state.get("verdict_reason"),
                    "verdict_conditions": final_state.get("verdict_conditions", []),
                    "invalid_query":      final_state.get("invalid_query", False),
                    "workflow_type":      final_state.get("workflow_type", ""),
                    "scenario_table":     final_state.get("scenario_table", []),
                    "escalation_flag":    final_state.get("escalation_flag", False),
                    "escalation_reason":  final_state.get("escalation_reason"),
                    "error_log":          final_state.get("error_log", []),
                }, ensure_ascii=False)
            }

        yield {"event": "done", "data": "완료"}

    return EventSourceResponse(event_generator())
