"""FastAPI 入口：尺寸公差链分析。

本机启动：
    .venv/bin/uvicorn app.main:app --reload
或：
    .venv/bin/python run.py
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

import numpy as np

from . import database as db
from .engine import (
    _override_chain,
    closure_samples,
    compute_all,
    gap_probability,
    normalize_chain,
)
from .inspection import analyze_batch, baseline_comparison, validate_rows
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
    InspectionBatchCreate,
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


@app.exception_handler(RequestValidationError)
async def _sanitizing_validation_handler(request, exc: RequestValidationError):
    """422 处理器：把错误详情里的 NaN/Infinity 回显替换为字符串。

    Python json 无法编码非有限浮点（实测值非有限被拒收时 pydantic 会把
    原始 input 原样放进错误），不清洗会退化为 500。
    """

    def clean(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return repr(obj)
        if isinstance(obj, Exception):
            return str(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return JSONResponse(
        status_code=422, content={"detail": clean(exc.errors())})


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
    """返回 (sc_row|None, nc, result, replay_kwargs)。

    方案上下文用保存的覆盖参数（含 sigma_explicit 标志）重放抽样，
    保证区间概率的蒙特卡洛与方案结果完全同源。
    """
    row = _load_chain(chain_id)
    if scenario_id is None:
        nc = rebuild_normalized(row.request_json)
        return None, nc, row.result_json, {}
    sc_row = db.get_scenario(scenario_id)
    if sc_row is None or sc_row.chain_id != chain_id:
        raise HTTPException(
            status_code=404,
            detail=f"链 {chain_id} 下方案 {scenario_id} 不存在",
        )
    nc = rebuild_normalized(row.request_json)
    ov = sc_row.overrides_json
    sigmas = np.array(ov["_resolved_sigmas_mm"], dtype=float)
    mids = np.array(ov["_resolved_mids_mm"], dtype=float)
    halfs = np.array(ov["_resolved_halfs_mm"], dtype=float)
    stored_flags = ov.get("_resolved_explicit")
    explicit = (
        [d.sigma_explicit for d in nc.dimensions]
        if stored_flags is None else [bool(x) for x in stored_flags]
    )
    nc = _override_chain(nc, sigmas, mids, halfs, explicit_flags=explicit)
    replay = {"sigmas": sigmas, "mids": mids, "halfs": halfs,
              "explicit_flags": explicit}
    return sc_row, nc, sc_row.result_json, replay


@app.post("/chains/{chain_id}/gap-probability", tags=["chains"])
def gap_probability_endpoint(
    chain_id: int, payload: GapProbabilityRequest,
    scenario_id: int | None = None,
) -> dict:
    """查询封闭环落在指定装配间隙区间的概率（三方法）。"""
    _, nc, result, replay = _resolve_context(chain_id, scenario_id)
    low_mm = None if payload.lower is None else to_mm(payload.lower,
                                                      payload.unit.value)
    high_mm = None if payload.upper is None else to_mm(payload.upper,
                                                       payload.unit.value)
    closure = closure_samples(nc, result, **replay)
    answer = gap_probability(nc, result, low_mm, high_mm, closure=closure,
                             **replay)
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
        ov_nc, sigmas, mids, halfs, explicit, record = apply_scenario_overrides(
            nc, payload
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = compute_all(
        ov_nc, sigmas=sigmas, mids=mids, halfs=halfs,
        seed=payload.random_seed, explicit_flags=explicit,
    )
    overrides_store: dict[str, Any] = {
        "human": record,
        "_resolved_sigmas_mm": [float(x) for x in sigmas],
        "_resolved_mids_mm": [float(x) for x in mids],
        "_resolved_halfs_mm": [float(x) for x in halfs],
        "_resolved_explicit": [bool(x) for x in explicit],
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
        ov_nc, sigmas, mids, halfs, explicit, record = apply_batch_adjust(
            nc, payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = compute_all(
        ov_nc, sigmas=sigmas, mids=mids, halfs=halfs,
        seed=payload.random_seed, explicit_flags=explicit,
    )
    overrides_store = {
        "human": record,
        "_resolved_sigmas_mm": [float(x) for x in sigmas],
        "_resolved_mids_mm": [float(x) for x in mids],
        "_resolved_halfs_mm": [float(x) for x in halfs],
        "_resolved_explicit": [bool(x) for x in explicit],
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


# -------------------------------------------------------- 来料检验批次

def _serialized_rows(payload: InspectionBatchCreate) -> list[dict]:
    """提交行的原样快照（单位与缺测标记保留），随批次冻结入库。"""
    return [
        {
            "serial": row.serial,
            "measurements": [
                {"dimension_id": m.dimension_id, "value": m.value,
                 "unit": m.unit.value}
                for m in row.measurements
            ],
        }
        for row in payload.rows
    ]


def _batch_response(row) -> dict:
    return {
        "batch_id": row.id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "bootstrap_samples": row.bootstrap_samples,
        "random_seed": row.random_seed,
        "created_at": row.created_at.isoformat(),
        "rows": row.rows_json,
        "report": row.report_json,
        "baseline_comparison": row.comparison_json,
    }


@app.post("/chains/{chain_id}/inspection-batches", status_code=201,
          tags=["inspection"])
def create_inspection_batch(chain_id: int, payload: InspectionBatchCreate) -> dict:
    """创建来料检验批次：复核链外尺寸/重复序号/非有限值/未知单位后冻结入库。

    缺测可入库（响应 report.gaps 列出每个工件的缺口）；批次落库后不可修改，
    后续测量应另建批次。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    try:
        rows = validate_rows(nc, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report = analyze_batch(
        nc, rows, payload.bootstrap_samples, payload.random_seed)
    comparison = baseline_comparison(nc, chain_row.result_json, report)
    stored_rows = _serialized_rows(payload)
    batch_id = db.save_inspection_batch(
        chain_id, payload.name, payload.note, stored_rows, report,
        comparison, payload.bootstrap_samples, payload.random_seed,
    )
    saved = db.get_inspection_batch(batch_id)
    return _batch_response(saved)


@app.get("/chains/{chain_id}/inspection-batches", tags=["inspection"])
def list_inspection_batches(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id,
            "batches": db.list_inspection_batches(chain_id)}


@app.get("/inspection-batches/{batch_id}", tags=["inspection"])
def get_inspection_batch(batch_id: int) -> dict:
    """读取冻结批次：报告在创建时已固化，多次读取内容不变。"""
    row = db.get_inspection_batch(batch_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"检验批次 {batch_id} 不存在")
    return _batch_response(row)
