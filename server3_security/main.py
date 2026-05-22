import os
import json
import re
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, validator
from typing import Optional

app = FastAPI(title="Security Gateway", version="1.0.0")

MES_BASE_URL = os.getenv("MES_BASE_URL", "http://mes_server:8001")
API_KEY = os.getenv("GATEWAY_API_KEY", "agent-secret-key-1234")

# 감사 로그 (인메모리, 최근 500건 유지)
_audit_logs: list = []


class MESRequest(BaseModel):
    order_id: str
    product_code: str
    required_qty: int
    required_date: str
    workflow_type: Optional[str] = "A"

    @validator("order_id")
    def validate_order_id(cls, v):
        if not re.match(r"^[A-Za-z0-9\-_]+$", v):
            raise ValueError("order_id에 허용되지 않는 문자가 포함됨")
        if len(v) > 50:
            raise ValueError("order_id 길이 초과")
        return v

    @validator("product_code")
    def validate_product_code(cls, v):
        if not re.match(r"^[A-Za-z0-9\-_]+$", v):
            raise ValueError("product_code에 허용되지 않는 문자가 포함됨")
        if len(v) > 50:
            raise ValueError("product_code 길이 초과")
        return v

    @validator("required_qty")
    def validate_qty(cls, v):
        if v <= 0 or v > 1_000_000:
            raise ValueError("required_qty는 1 ~ 1,000,000 범위여야 함")
        return v

    @validator("required_date")
    def validate_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("required_date는 YYYY-MM-DD 형식이어야 함")
        return v

    @validator("workflow_type")
    def validate_workflow(cls, v):
        if v not in ("A", "B", "C"):
            raise ValueError("workflow_type은 A, B, C 중 하나여야 함")
        return v


def _check_api_key(x_api_key: Optional[str]) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API 키가 없거나 유효하지 않음")


def _record(endpoint: str, req: MESRequest, status: int, client_ip: str) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
        "order_id": req.order_id,
        "product_code": req.product_code,
        "workflow_type": req.workflow_type,
        "required_qty": req.required_qty,
        "status": status,
        "client_ip": client_ip,
    }
    _audit_logs.append(entry)
    if len(_audit_logs) > 500:
        _audit_logs.pop(0)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


async def _proxy(endpoint: str, req: MESRequest, client_ip: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{MES_BASE_URL}/mes/{endpoint}",
                json=req.dict(),
            )
        _record(endpoint, req, resp.status_code, client_ip)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())
        return resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        _record(endpoint, req, 502, client_ip)
        raise HTTPException(status_code=502, detail=f"MES 서버 연결 실패: {e}")


# ── 헬스체크 ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "server": "Security Gateway"}


# ── 감사 로그 조회 ────────────────────────────────────────────
@app.get("/audit/logs")
def audit_logs():
    return {"count": len(_audit_logs), "logs": _audit_logs[-100:]}


# ── MES 프록시 엔드포인트 ──────────────────────────────────────
@app.post("/mes/production-capa")
async def production_capa(
    req: MESRequest, request: Request, x_api_key: Optional[str] = Header(None)
):
    _check_api_key(x_api_key)
    return await _proxy("production-capa", req, request.client.host)


@app.post("/mes/material-stock")
async def material_stock(
    req: MESRequest, request: Request, x_api_key: Optional[str] = Header(None)
):
    _check_api_key(x_api_key)
    return await _proxy("material-stock", req, request.client.host)


@app.post("/mes/quality-condition")
async def quality_condition(
    req: MESRequest, request: Request, x_api_key: Optional[str] = Header(None)
):
    _check_api_key(x_api_key)
    return await _proxy("quality-condition", req, request.client.host)


@app.post("/mes/mold-setup")
async def mold_setup(
    req: MESRequest, request: Request, x_api_key: Optional[str] = Header(None)
):
    _check_api_key(x_api_key)
    return await _proxy("mold-setup", req, request.client.host)


@app.post("/mes/order-conflict")
async def order_conflict(
    req: MESRequest, request: Request, x_api_key: Optional[str] = Header(None)
):
    _check_api_key(x_api_key)
    return await _proxy("order-conflict", req, request.client.host)
