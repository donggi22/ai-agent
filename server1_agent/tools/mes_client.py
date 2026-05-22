import httpx
from config import SECURITY_BASE_URL, GATEWAY_API_KEY

_HEADERS = {"x-api-key": GATEWAY_API_KEY}


def _build_payload(state: dict) -> dict:
    return {
        "order_id":      state["order_id"],
        "product_code":  state["product_code"],
        "required_qty":  state["required_qty"],
        "required_date": state["required_date"],
        "workflow_type": state["workflow_type"],
    }


async def call_production_capa(state: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{SECURITY_BASE_URL}/mes/production-capa",
            json=_build_payload(state),
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()


async def call_material_stock(state: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{SECURITY_BASE_URL}/mes/material-stock",
            json=_build_payload(state),
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()


async def call_quality_condition(state: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{SECURITY_BASE_URL}/mes/quality-condition",
            json=_build_payload(state),
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()


async def call_mold_setup(state: dict) -> dict:
    """422 에러는 호출자에서 잡아서 error_log에 기록"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{SECURITY_BASE_URL}/mes/mold-setup",
            json=_build_payload(state),
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()


async def call_order_conflict(state: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{SECURITY_BASE_URL}/mes/order-conflict",
            json=_build_payload(state),
            headers=_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()
