"""FastAPI 入口：尺寸公差链分析。

本机启动：
    .venv/bin/uvicorn app.main:app --reload
或：
    .venv/bin/python run.py
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

import numpy as np

from . import database as db
from .engine import (
    _override_chain,
    closure_samples,
    compute_all,
    gap_probability,
    normalize_chain,
)
from .optimizer import search_cost_targets
from .scenarios import (
    apply_batch_adjust,
    apply_scenario_overrides,
    compare_with_baseline,
    rebuild_normalized,
)
from .schemas import (
    BatchAdjustRequest,
    ChainCreate,
    CostTargetRequest,
    GapProbabilityRequest,
    ScenarioCreate,
)
from .units import to_mm


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.get_engine()
    yield


app = FastAPI(
    title="尺寸公差链分析 API",
    version="1.0.0",
    description=(
        "面向机械设计与来料评审的有向尺寸链公差分析："
        "极值法 / RSS / 固定种子蒙特卡洛，单位统一、相关系数、"
        "方案分支与成本导向的公差收紧搜索。"
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ------------------------------------------------------------------ 链

@app.post("/chains", status_code=201, tags=["chains"])
def create_chain(payload: ChainCreate) -> dict:
    """创建基线尺寸链：校验、规范化（混合单位/单边公差保留原值）并计算。"""
    try:
        nc = normalize_chain(payload)
        result = compute_all(nc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    normalized = result["normalized_inputs"]
    chain_id = db.save_chain(
        name=payload.name,
        description=payload.description,
        request_data=payload.model_dump(mode="json"),
        normalized={"dimensions": normalized},
        result=result,
        mc_samples=payload.mc_samples,
        seed=payload.random_seed,
    )
    return {
        "chain_id": chain_id,
        "name": payload.name,
        "baseline_kept": True,
        "result": result,
    }


@app.get("/chains", tags=["chains"])
def list_chains() -> dict:
    return {"chains": db.list_chains()}


def _load_chain(chain_id: int):
    row = db.get_chain(chain_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"链 {chain_id} 不存在")
    return row


@app.get("/chains/{chain_id}", tags=["chains"])
def get_chain(chain_id: int) -> dict:
    row = _load_chain(chain_id)
    return {
        "chain_id": row.id,
        "name": row.name,
        "description": row.description,
        "mc_samples": row.mc_samples,
        "random_seed": row.random_seed,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "normalized": row.normalized_json,
        "result": row.result_json,
    }


@app.get("/chains/{chain_id}/traceability", tags=["chains"])
def chain_traceability(chain_id: int) -> dict:
    """追溯每项结果采用的公式、输入规范化值与原始表示。"""
    row = _load_chain(chain_id)
    res = row.result_json
    return {
        "chain_id": chain_id,
        "closure_spec_mm": res["closure_spec_mm"],
        "normalized_inputs": res["normalized_inputs"],
        "traceability": res["traceability"],
        "formulas_by_method": {
            m: res["results"][m]["formulas"]
            for m in ("worst_case", "rss", "monte_carlo")
        },
        "sensitivity_by_method": {
            m: res["results"][m]["sensitivity"]
            for m in ("worst_case", "rss", "monte_carlo")
        },
    }


# ------------------------------------------------------------ 区间概率

def _resolve_context(chain_id: int, scenario_id: int | None):
    """返回 (row/None, nc, result)；scenario_id 给定时使用方案结果。"""
    row = _load_chain(chain_id)
    if scenario_id is None:
        nc = rebuild_normalized(row.request_json)
        return None, nc, row.result_json
    sc_row = db.get_scenario(scenario_id)
    if sc_row is None or sc_row.chain_id != chain_id:
        raise HTTPException(
            status_code=404,
            detail=f"链 {chain_id} 下方案 {scenario_id} 不存在",
        )
    # 方案区间概率：以方案保存的覆盖参数重放
    nc = rebuild_normalized(row.request_json)
    ov = sc_row.overrides_json
    sigmas = np.array(ov["_resolved_sigmas_mm"], dtype=float)
    mids = np.array(ov["_resolved_mids_mm"], dtype=float)
    halfs = np.array(ov["_resolved_halfs_mm"], dtype=float)
    nc = _override_chain(nc, sigmas, mids, halfs)
    return sc_row, nc, sc_row.result_json


@app.post("/chains/{chain_id}/gap-probability", tags=["chains"])
def gap_probability_endpoint(
    chain_id: int, payload: GapProbabilityRequest,
    scenario_id: int | None = None,
) -> dict:
    """查询封闭环落在指定装配间隙区间的概率（三方法）。"""
    _, nc, result = _resolve_context(chain_id, scenario_id)
    low_mm = None if payload.lower is None else to_mm(payload.lower,
                                                      payload.unit.value)
    high_mm = None if payload.upper is None else to_mm(payload.upper,
                                                       payload.unit.value)
    closure = closure_samples(nc, result)
    answer = gap_probability(nc, result, low_mm, high_mm, closure=closure)
    answer["query"] = {
        "submitted": {"lower": payload.lower, "upper": payload.upper,
                      "unit": payload.unit.value},
        "normalized_mm": {"lower": low_mm, "upper": high_mm},
        "scenario_id": scenario_id,
    }
    return answer


# ------------------------------------------------------------ 方案分支

@app.post("/chains/{chain_id}/scenarios", status_code=201, tags=["scenarios"])
def create_scenario(chain_id: int, payload: ScenarioCreate) -> dict:
    """创建不覆盖基线的方案分支：逐尺寸改偏差/标准差后对比。"""
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    try:
        ov_nc, sigmas, mids, halfs, record = apply_scenario_overrides(
            nc, payload
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = compute_all(
        ov_nc, sigmas=sigmas, mids=mids, halfs=halfs,
        seed=payload.random_seed,
    )
    overrides_store: dict[str, Any] = {
        "human": record,
        "_resolved_sigmas_mm": [float(x) for x in sigmas],
        "_resolved_mids_mm": [float(x) for x in mids],
        "_resolved_halfs_mm": [float(x) for x in halfs],
        "random_seed": payload.random_seed,
    }
    scenario_id = db.save_scenario(
        chain_id, "branch", payload.name, payload.note,
        overrides_store, result,
    )
    return {
        "scenario_id": scenario_id,
        "chain_id": chain_id,
        "name": payload.name,
        "baseline_preserved": True,
        "overrides": record,
        "comparison": compare_with_baseline(row.result_json, result),
        "result": result,
    }


@app.post("/chains/{chain_id}/batch-adjust", status_code=201, tags=["scenarios"])
def batch_adjust(chain_id: int, payload: BatchAdjustRequest) -> dict:
    """批量缩放公差带（可选标准差），保存为方案分支并与基线对比。"""
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    try:
        ov_nc, sigmas, mids, halfs, record = apply_batch_adjust(nc, payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = compute_all(
        ov_nc, sigmas=sigmas, mids=mids, halfs=halfs,
        seed=payload.random_seed,
    )
    overrides_store = {
        "human": record,
        "_resolved_sigmas_mm": [float(x) for x in sigmas],
        "_resolved_mids_mm": [float(x) for x in mids],
        "_resolved_halfs_mm": [float(x) for x in halfs],
        "random_seed": payload.random_seed,
    }
    scenario_id = db.save_scenario(
        chain_id, "batch", payload.name, payload.note,
        overrides_store, result,
    )
    return {
        "scenario_id": scenario_id,
        "chain_id": chain_id,
        "name": payload.name,
        "baseline_preserved": True,
        "adjustment": record,
        "comparison": compare_with_baseline(row.result_json, result),
        "result": result,
    }


@app.get("/chains/{chain_id}/scenarios", tags=["scenarios"])
def list_scenarios(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id, "scenarios": db.list_scenarios(chain_id)}


@app.get("/scenarios/{scenario_id}", tags=["scenarios"])
def get_scenario(scenario_id: int) -> dict:
    row = db.get_scenario(scenario_id)
    if row is None:
        raise HTTPException(status_code=404, detail="方案不存在")
    return {
        "scenario_id": row.id,
        "chain_id": row.chain_id,
        "kind": row.kind,
        "name": row.name,
        "note": row.note,
        "overrides": row.overrides_json.get("human", row.overrides_json),
        "created_at": row.created_at.isoformat(),
        "result": row.result_json,
    }


# -------------------------------------------------------- 成本收紧搜索

@app.post("/chains/{chain_id}/cost-targets", tags=["optimization"])
def cost_targets(chain_id: int, payload: CostTargetRequest) -> dict:
    """按单位收紧成本给出达到目标超差率的候选组合（基线不变）。"""
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    if nc.closure_lsl_mm is None and nc.closure_usl_mm is None:
        raise HTTPException(
            status_code=422,
            detail="该链未声明封闭环上下限，无法定义超差率；"
                   "请先在链上设置 closure_lower/upper_limit",
        )
    unknown = set(payload.tightening_cost) - {d.id for d in nc.dimensions}
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"单位成本引用了不存在的尺寸: {sorted(unknown)}",
        )
    try:
        answer = search_cost_targets(nc, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    answer["chain_id"] = chain_id
    return answer
