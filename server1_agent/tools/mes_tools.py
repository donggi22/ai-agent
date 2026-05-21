import httpx
from langchain_core.tools import tool
from config import MES_BASE_URL


def _payload(order_id: str, product_code: str, required_qty: int, required_date: str, workflow_type: str) -> dict:
    return {
        "order_id": order_id,
        "product_code": product_code,
        "required_qty": required_qty,
        "required_date": required_date,
        "workflow_type": workflow_type,
    }


@tool
async def get_production_capa(
    order_id: str,
    product_code: str,
    required_qty: int,
    required_date: str,
    workflow_type: str,
) -> dict:
    """MES에서 생산 CAPA를 조회합니다. 설비 가동률, 가용시간, 생산 가능 수량을 반환합니다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{MES_BASE_URL}/mes/production-capa",
            json=_payload(order_id, product_code, required_qty, required_date, workflow_type),
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def get_material_stock(
    order_id: str,
    product_code: str,
    required_qty: int,
    required_date: str,
    workflow_type: str,
) -> dict:
    """MES에서 자재 재고를 조회합니다. 현재 재고, 부족 수량, 조달 리드타임을 반환합니다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{MES_BASE_URL}/mes/material-stock",
            json=_payload(order_id, product_code, required_qty, required_date, workflow_type),
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def get_quality_condition(
    order_id: str,
    product_code: str,
    required_qty: int,
    required_date: str,
    workflow_type: str,
) -> dict:
    """MES에서 품질 조건을 조회합니다. 불량률, 기준 충족 여부를 반환합니다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{MES_BASE_URL}/mes/quality-condition",
            json=_payload(order_id, product_code, required_qty, required_date, workflow_type),
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def get_mold_setup(
    order_id: str,
    product_code: str,
    required_qty: int,
    required_date: str,
    workflow_type: str,
) -> dict:
    """MES에서 금형 셋업 정보를 조회합니다. 교체시간, 안정화 샷 수, 초기불량률을 반환합니다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{MES_BASE_URL}/mes/mold-setup",
            json=_payload(order_id, product_code, required_qty, required_date, workflow_type),
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def get_order_conflict(
    order_id: str,
    product_code: str,
    required_qty: int,
    required_date: str,
    workflow_type: str,
) -> dict:
    """MES에서 수주 경합 정보를 조회합니다. 경합 수주 목록과 에스컬레이션 여부를 반환합니다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{MES_BASE_URL}/mes/order-conflict",
            json=_payload(order_id, product_code, required_qty, required_date, workflow_type),
        )
        resp.raise_for_status()
        return resp.json()


ALL_MES_TOOLS = [
    get_production_capa,
    get_material_stock,
    get_quality_condition,
    get_mold_setup,
    get_order_conflict,
]

TOOL_MAP = {t.name: t for t in ALL_MES_TOOLS}
