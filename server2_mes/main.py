import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="MES Mock Server", version="1.0.0")

# 응답 데이터 로드
RESPONSES_DIR = Path(__file__).parent / "responses"

def load_response(workflow: str) -> dict:
    path = RESPONSES_DIR / f"workflow_{workflow.lower()}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"workflow {workflow} 응답 없음")
    return json.loads(path.read_text(encoding="utf-8"))


# ── Request 스키마 ────────────────────────────────────────────
class MESRequest(BaseModel):
    order_id: str
    product_code: str
    required_qty: int
    required_date: str
    workflow_type: Optional[str] = "A"   # A / B / C


# ── 헬스체크 ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "server": "MES Mock"}


# ── 생산 CAPA 조회 ────────────────────────────────────────────
@app.post("/mes/production-capa")
def production_capa(req: MESRequest):
    data = load_response(req.workflow_type)
    result = data.get("production_capa")
    if not result:
        raise HTTPException(status_code=500, detail="production_capa 데이터 없음")
    return {
        "endpoint": "production-capa",
        "order_id": req.order_id,
        "workflow_type": req.workflow_type,
        "data": result
    }


# ── 자재 재고 조회 ────────────────────────────────────────────
@app.post("/mes/material-stock")
def material_stock(req: MESRequest):
    data = load_response(req.workflow_type)
    result = data.get("material_stock")
    if not result:
        # 워크플로 A는 자재 조회 불필요 → 정상 재고 응답
        return {
            "endpoint": "material-stock",
            "order_id": req.order_id,
            "workflow_type": req.workflow_type,
            "data": {
                "status": "ok",
                "current_stock": 9999,
                "required_qty": req.required_qty,
                "shortage_qty": 0,
                "lead_time_days": 0,
                "note": "재고 충분"
            }
        }
    return {
        "endpoint": "material-stock",
        "order_id": req.order_id,
        "workflow_type": req.workflow_type,
        "data": result
    }


# ── 품질 조건 조회 ────────────────────────────────────────────
@app.post("/mes/quality-condition")
def quality_condition(req: MESRequest):
    data = load_response(req.workflow_type)
    result = data.get("quality_condition")
    if not result:
        raise HTTPException(status_code=500, detail="quality_condition 데이터 없음")
    return {
        "endpoint": "quality-condition",
        "order_id": req.order_id,
        "workflow_type": req.workflow_type,
        "data": result
    }


# ── 금형 셋업 조회 ────────────────────────────────────────────
@app.post("/mes/mold-setup")
def mold_setup(req: MESRequest):
    data = load_response(req.workflow_type)
    result = data.get("mold_setup")
    if not result:
        raise HTTPException(status_code=500, detail="mold_setup 데이터 없음")

    # 워크플로 C: 금형 이동 불가 에러 재현
    if result.get("status") == "error":
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": result["error_code"],
                "error_message": result["error_message"],
                "mold_id": result["mold_id"]
            }
        )
    return {
        "endpoint": "mold-setup",
        "order_id": req.order_id,
        "workflow_type": req.workflow_type,
        "data": result
    }


# ── 수주 경합 조회 (워크플로 C 전용) ─────────────────────────
@app.post("/mes/order-conflict")
def order_conflict(req: MESRequest):
    if req.workflow_type != "C":
        return {
            "endpoint": "order-conflict",
            "order_id": req.order_id,
            "workflow_type": req.workflow_type,
            "data": {"status": "ok", "competing_orders": [], "note": "경합 없음"}
        }

    data = load_response("C")
    result = data.get("order_conflict")
    if not result:
        raise HTTPException(status_code=500, detail="order_conflict 데이터 없음")

    # 에스컬레이션 코드 포함 응답
    return {
        "endpoint": "order-conflict",
        "order_id": req.order_id,
        "workflow_type": req.workflow_type,
        "data": result
    }
