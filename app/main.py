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
from .assembly import AssemblyError, run_task
from .engine import (
    _override_chain,
    closure_samples,
    compute_all,
    gap_probability,
    normalize_chain,
)
from .inspection import analyze_batch, baseline_comparison, validate_rows
from .measurement import evaluate_batch, normalize_plan
from .optimizer import search_cost_targets
from .scenarios import (
    apply_batch_adjust,
    apply_scenario_overrides,
    compare_with_baseline,
    rebuild_normalized,
)
from .schemas import (
    AssemblyTaskCreate,
    AssemblyVersionCreate,
    BatchAdjustRequest,
    ChainCreate,
    CostTargetRequest,
    GapProbabilityRequest,
    GuardBandSpec,
    InspectionBatchCreate,
    MeasurementPlanCreate,
    ScenarioCreate,
    ThermalAnalysisCreate,
    ThermalProposalRequest,
)
from .thermal import ThermalError, analyze as thermal_analyze, build_model, search_proposals
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


# -------------------------------------------------------- 测量方案

@app.post("/chains/{chain_id}/measurement-plans", status_code=201,
          tags=["measurement"])
def create_measurement_plan(chain_id: int, payload: MeasurementPlanCreate) -> dict:
    """建立不可变测量方案：逐尺寸量具误差 + 共用量具相关项。

    统一单位为 mm 后合成标准不确定度；覆盖因子缺失、分量为负、
    相关矩阵非半正定、未覆盖链上全部尺寸时拒绝保存（422）。
    方案创建后不可修改，新版本请另建方案（历史批次引用不受影响）。
    """
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    try:
        combined = normalize_plan(nc, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    plan_id = db.save_measurement_plan(
        chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), combined,
    )
    return {
        "plan_id": plan_id,
        "chain_id": chain_id,
        "name": payload.name,
        "immutable": True,
        "plan": combined,
    }


@app.get("/chains/{chain_id}/measurement-plans", tags=["measurement"])
def list_measurement_plans(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id, "plans": db.list_measurement_plans(chain_id)}


@app.get("/measurement-plans/{plan_id}", tags=["measurement"])
def get_measurement_plan(plan_id: int) -> dict:
    row = db.get_measurement_plan(plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"测量方案 {plan_id} 不存在")
    return {
        "plan_id": row.id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "immutable": True,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "plan": row.combined_json,
    }


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
        "measurement_plan_id": row.measurement_plan_id,
        "created_at": row.created_at.isoformat(),
        "rows": row.rows_json,
        "report": row.report_json,
        "baseline_comparison": row.comparison_json,
        "measurement": row.measurement_json,
    }


@app.post("/chains/{chain_id}/inspection-batches", status_code=201,
          tags=["inspection"])
def create_inspection_batch(chain_id: int, payload: InspectionBatchCreate) -> dict:
    """创建来料检验批次：复核链外尺寸/重复序号/非有限值/未知单位后冻结入库。

    缺测可入库（响应 report.gaps 列出每个工件的缺口）；批次落库后不可修改，
    后续测量应另建批次。引用 measurement_plan_id 时：先按方案修正偏倚，
    再用 GUM 线性传播与固定种子蒙特卡洛评定各实测值与封闭环的扩展不确定度，
    按保护带给出接收/拒收/不确定判定及真值越界/合格概率；
    方案快照随批次冻结，后续新版本方案不改变本批判定。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    try:
        rows = validate_rows(nc, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    measurement = None
    plan_id = payload.measurement_plan_id
    if plan_id is not None:
        plan_row = db.get_measurement_plan(plan_id)
        if plan_row is None or plan_row.chain_id != chain_id:
            raise HTTPException(
                status_code=404,
                detail=f"链 {chain_id} 下测量方案 {plan_id} 不存在",
            )
        gb = payload.guard_band or GuardBandSpec()
        measurement = evaluate_batch(
            nc,
            plan_row.combined_json,
            rows,
            plan_meta={"plan_id": plan_row.id, "name": plan_row.name,
                       "note": plan_row.note},
            k_out=payload.output_coverage_factor,
            guard_band={
                "mode": gb.mode,
                "multiple": gb.multiple,
                "fixed_mm": (to_mm(gb.fixed, gb.unit.value)
                             if gb.fixed is not None else None),
                "submitted": gb.model_dump(mode="json"),
            },
            mc_samples=payload.measurement_mc_samples,
            seed=payload.measurement_mc_seed,
        )

    report = analyze_batch(
        nc, rows, payload.bootstrap_samples, payload.random_seed)
    comparison = baseline_comparison(nc, chain_row.result_json, report)
    stored_rows = _serialized_rows(payload)
    batch_id = db.save_inspection_batch(
        chain_id, payload.name, payload.note, stored_rows, report,
        comparison, payload.bootstrap_samples, payload.random_seed,
        measurement_plan_id=plan_id, measurement=measurement,
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


# -------------------------------------------------------- 选择性装配任务

def _load_assembly_batches(chain_id: int, batch_ids: list[int]):
    """取冻结批次并复核同链、不重复（错链 / 不存在 -> 422/404）。"""
    rows = []
    problems = []
    seen: set[int] = set()
    for bid in batch_ids:
        if bid in seen:
            continue
        seen.add(bid)
        brow = db.get_inspection_batch(bid)
        if brow is None:
            raise HTTPException(
                status_code=404, detail=f"检验批次 {bid} 不存在")
        if brow.chain_id != chain_id:
            problems.append(
                f"批次 {bid} 属于链 {brow.chain_id}，不属于当前链 {chain_id}")
        rows.append(brow)
    if problems:
        raise HTTPException(status_code=422, detail="；".join(problems))
    return rows


def _assembly_payload(version_row, task_row) -> dict:
    return {
        "task_id": version_row.task_id,
        "task_name": task_row.name,
        "task_note": task_row.note,
        "version_id": version_row.id,
        "version_no": version_row.version_no,
        "parent_version_id": version_row.parent_version_id,
        "chain_id": version_row.chain_id,
        "name": version_row.name,
        "note": version_row.note,
        "created_at": version_row.created_at.isoformat(),
        "frozen": True,
        "request": version_row.request_json,
        "result": version_row.result_json,
    }


@app.post("/chains/{chain_id}/assembly-tasks", status_code=201,
          tags=["assembly"])
def create_assembly_task(chain_id: int, payload: AssemblyTaskCreate) -> dict:
    """创建选择性装配任务（版本 1）。

    从同一基线链的多个冻结检验批次取数，按池映射构建零件池；
    批次错链、尺寸漏映射/重复映射、跨批/同批/禁配关系矛盾时拒绝（422）。
    系统按偏倚修正后的实测值计算每组封闭环间隙，传播量具不确定度，
    按保护带判定合格/不确定/不合格，再以「合格数→中心偏差→最差保护余量
    →跨批次数」字典序选互不相交组合。结果快照冻结源批次、测量方案与种子。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    batch_rows = _load_assembly_batches(chain_id, payload.batch_ids)
    try:
        result = run_task(
            nc, batch_rows, payload,
            seed=payload.random_seed,
            mc_samples=payload.measurement_mc_samples)
    except AssemblyError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc),
                    **({"diagnostics": exc.diagnostics}
                       if getattr(exc, "diagnostics", None) else {})}
        ) from exc

    task_id, version_id = db.save_assembly_first_version(
        chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    saved = db.get_assembly_version(version_id)
    task = db.get_assembly_task(task_id)
    return _assembly_payload(saved, task)


@app.get("/chains/{chain_id}/assembly-tasks", tags=["assembly"])
def list_assembly_tasks(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id, "tasks": db.list_assembly_tasks(chain_id)}


@app.get("/assembly-tasks/{task_id}/versions", tags=["assembly"])
def list_task_versions(task_id: int) -> dict:
    task = db.get_assembly_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"装配任务 {task_id} 不存在")
    versions = db.list_assembly_versions(task_id)
    return {
        "task_id": task_id,
        "chain_id": task.chain_id,
        "name": task.name,
        "versions": [
            {
                "version_id": v.id,
                "version_no": v.version_no,
                "parent_version_id": v.parent_version_id,
                "note": v.note,
                "created_at": v.created_at.isoformat(),
                "status": v.result_json["status"],
                "required_count": v.result_json["required_count"],
                "qualified_assemblies": v.result_json["qualified_assemblies"],
                "locked_assemblies": v.result_json["locked_assemblies"],
            }
            for v in versions
        ],
    }


@app.get("/assembly-versions/{version_id}", tags=["assembly"])
def get_assembly_version(version_id: int) -> dict:
    v = db.get_assembly_version(version_id)
    if v is None:
        raise HTTPException(status_code=404,
                            detail=f"装配版本 {version_id} 不存在")
    return _assembly_payload(v, db.get_assembly_task(v.task_id))


@app.post("/assembly-versions/{version_id}/rearrange", status_code=201,
          tags=["assembly"])
def rearrange_assembly(version_id: int, payload: AssemblyVersionCreate) -> dict:
    """基于已冻结版本另建新版本：锁定调用方确认的组合，重排其余实例。

    新增锁定装配在父版本已锁定集合之上累积；锁定引用必须是父版本求解出的
    合格组合（池齐全、实例唯一、满足同批/跨批/禁配规则），否则 422。
    其余实例重新求解；快照独立冻结，父版本不变。
    """
    parent = db.get_assembly_version(version_id)
    if parent is None:
        raise HTTPException(status_code=404,
                            detail=f"装配版本 {version_id} 不存在")
    chain_row = _load_chain(parent.chain_id)
    nc = rebuild_normalized(chain_row.request_json)

    presult = parent.result_json
    snap = presult["snapshot"]
    batch_ids = [b["batch_id"] for b in snap["source_batches"]]
    batch_rows = _load_assembly_batches(parent.chain_id, batch_ids)

    # 用父版本快照重建等价的任务参数（规则原样重放）
    parent_req = _ReplayedTask.from_snapshot(snap)
    # 合并父版本累积锁定 + 本次新增锁定
    locked = [[{"pool": m["pool"], "batch_id": m["batch_id"],
                "serial": m["serial"]}
               for m in a["members"]]
              for a in presult["assemblies"] if a.get("locked")]
    pool_names = [p.name for p in parent_req.pools]
    already = {frozenset((m["pool"], m["batch_id"], m["serial"])
                         for m in lock) for lock in locked}
    for spec in payload.locked_assemblies:
        members = [m for m in spec.members]
        mpools = [m.pool for m in members]
        unknown = [p for p in mpools if p not in pool_names]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"锁定装配引用了不存在的零件池: {sorted(set(unknown))}")
        if sorted(mpools) != sorted(pool_names):
            missing = sorted(set(pool_names) - set(mpools))
            extra = sorted(set(mpools) - set(pool_names))
            raise HTTPException(
                status_code=422,
                detail=f"锁定装配必须恰好覆盖全部零件池；缺失 {missing}，"
                       f"多余 {extra}")
        # 必须是父版本求解结果中的合格（accept）组合
        accepted_keys = {
            frozenset((m["pool"], m["batch_id"], m["serial"])
                      for m in a["members"])
            for a in presult["assemblies"] if a["decision"] == "accept"
        }
        key = frozenset((m.pool, m.batch_id, m.serial) for m in members)
        if key in already:
            raise HTTPException(
                status_code=422,
                detail="锁定装配与已锁定组合重复："
                       + ", ".join(f"{m.batch_id}:{m.serial}"
                                   for m in members))
        if key not in accepted_keys:
            raise HTTPException(
                status_code=422,
                detail="只能锁定父版本求解结果中判定为合格的组合；"
                       f"给定组合 {sorted(str(k) for k in key)} 不在合格集合中")
        locked.append([{"pool": m.pool, "batch_id": m.batch_id,
                        "serial": m.serial} for m in members])
        already.add(key)

    if len(locked) > parent_req.assembly_count:
        raise HTTPException(
            status_code=422,
            detail=f"锁定装配数 {len(locked)} 超过要求装配数量 "
                   f"{parent_req.assembly_count}")

    k_out = payload.output_coverage_factor or parent_req.output_coverage_factor
    gb_spec = payload.guard_band or GuardBandSpec(
        **snap["guard_band"]["submitted"])
    mc_samples = payload.measurement_mc_samples or snap["monte_carlo_samples"]
    seed = payload.random_seed if payload.random_seed is not None else snap["random_seed"]
    parent_req.output_coverage_factor = k_out
    parent_req.guard_band = gb_spec

    try:
        result = run_task(
            nc, batch_rows, parent_req, seed=seed, mc_samples=mc_samples,
            locked_assemblies=locked, parent_version_id=parent.id)
    except AssemblyError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc),
                    **({"diagnostics": exc.diagnostics}
                       if getattr(exc, "diagnostics", None) else {})}
        ) from exc

    new_no = max(v.version_no for v in db.list_assembly_versions(parent.task_id)) + 1
    new_id = db.save_assembly_version(
        parent.task_id, parent.chain_id, new_no, parent.id,
        parent.name, payload.note,
        {"note": payload.note,
         "newly_locked": [
             [{"pool": m.pool, "batch_id": m.batch_id, "serial": m.serial}
              for m in s.members] for s in payload.locked_assemblies],
         "output_coverage_factor": k_out,
         "guard_band": gb_spec.model_dump(mode="json"),
         "measurement_mc_samples": mc_samples,
         "random_seed": seed},
        result)
    saved = db.get_assembly_version(new_id)
    return _assembly_payload(saved, db.get_assembly_task(parent.task_id))


class _ReplayedTask:
    """从版本快照重建 run_task 所需的任务参数（规则原样冻结重放）。"""

    def __init__(self, name, pools, assembly_count, target_gap, cross_batch_limit,
                 same_batch_groups, forbidden_matches, output_coverage_factor,
                 guard_band):
        self.name = name
        self.pools = pools
        self.assembly_count = assembly_count
        self.target_gap = target_gap
        self.cross_batch_limit = cross_batch_limit
        self.same_batch_groups = same_batch_groups
        self.forbidden_matches = forbidden_matches
        self.output_coverage_factor = output_coverage_factor
        self.guard_band = guard_band

    @classmethod
    def from_snapshot(cls, snap: dict) -> "_ReplayedTask":
        from .schemas import (
            ForbiddenMatchSpec, PoolSpec, SameBatchGroupSpec, TargetGapSpec,
        )

        rules = snap.get("rules", {})
        return cls(
            name="replayed",
            pools=[PoolSpec(**p) for p in snap["pool_mapping"]],
            assembly_count=snap["assembly_count"],
            target_gap=TargetGapSpec(**snap["target_gap"]["submitted"]),
            cross_batch_limit=snap["cross_batch_limit"],
            same_batch_groups=[
                SameBatchGroupSpec(**g)
                for g in rules.get("submitted_same_batch_groups", [])],
            forbidden_matches=[
                ForbiddenMatchSpec(**f)
                for f in rules.get("submitted_forbidden_matches", [])],
            output_coverage_factor=snap["output_coverage_factor"],
            guard_band=GuardBandSpec(**snap["guard_band"]["submitted"]),
        )


# -------------------------------------------------------- 热分析（温度工况）

def _thermal_model_snapshot(model) -> dict:
    """归一化热模型快照（随版本冻结；重复计算据此可精确复现）。"""
    return {
        "dimension_order": model.ids,
        "reference_temperature_c": [float(x) for x in model.t0_c],
        "reference_temperature_unit_submitted": model.t0_unit,
        "alpha_per_K": [float(x) for x in model.alpha],
        "alpha_std_uncertainty_per_K": [float(x) for x in model.u_alpha],
        "alpha_unit_submitted": model.alpha_unit,
        "alpha_correlation_matrix": model.r_alpha.tolist(),
        "temperature_correlation_matrix": model.r_temp.tolist(),
        "conditions": [
            {
                "name": model.condition_names[ci],
                "note": model.condition_notes[ci],
                "temperature_unit_submitted": model.condition_temp_units[ci],
                "dimensions": [
                    {"dimension_id": model.ids[i], **model.conditions[ci][i].raw,
                     "normalized_c": {
                         "mean": model.conditions[ci][i].mean,
                         "half_width": model.conditions[ci][i].half,
                         "std_uncertainty": model.conditions[ci][i].sigma}}
                    for i in range(len(model.ids))],
            }
            for ci in range(len(model.conditions))
        ],
        "monte_carlo": {"samples": model.mc_samples, "seed": model.seed},
        "formula": "L(T) = L0 [1 + alpha (T - T0)]",
    }


def _thermal_response(row) -> dict:
    return {
        "thermal_analysis_id": row.id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "mc_samples": row.mc_samples,
        "random_seed": row.random_seed,
        "created_at": row.created_at.isoformat(),
        "baseline_preserved": True,
        "submitted_input": row.request_json,
        "model": row.model_json,
        "result": row.result_json,
    }


@app.post("/chains/{chain_id}/thermal-analyses", status_code=201,
          tags=["thermal"])
def create_thermal_analysis(chain_id: int,
                            payload: ThermalAnalysisCreate) -> dict:
    """从冻结基线链建立热分析版本：逐尺寸 T0/α/u(α) + 工况温度。

    按 L(T)=L0[1+α(T−T0)] 换算名义值与公差带，逐工况给出热态封闭环
    均值、极值边界、RSS 与固定种子蒙特卡洛超差率，拆分制造偏差 /
    膨胀系数 / 温度不确定度三类贡献，并指出最先越过规格的工况。
    漏填尺寸、温度上下界倒置、α 或温度相关矩阵非半正定时拒绝（422）。
    基线链不变，结果与种子随版本冻结（同一版本重复计算不变）。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    if nc.closure_lsl_mm is None or nc.closure_usl_mm is None:
        raise HTTPException(
            status_code=422,
            detail="热分析要求基线链同时声明封闭环下限与上限"
                   "（closure_lower_limit / closure_upper_limit），否则无法"
                   "定义热态超差率与最先越界工况",
        )
    try:
        model = build_model(nc, payload)
        result = thermal_analyze(model)
    except ThermalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    snapshot = _thermal_model_snapshot(model)
    result = {
        **result,
        "closure_spec_mm": {
            "lower_limit": nc.closure_lsl_mm,
            "upper_limit": nc.closure_usl_mm,
        },
        "traceability": {
            "baseline_chain_id": chain_id,
            "formula": "L(T) = L0 [1 + alpha (T - T0)]",
            "uncertainty_sources": [
                "制造偏差：基线分布/相关，热态按 1+αΔT 等比缩放",
                "膨胀系数：GUM 一阶 (L0ΔT)·u(α)，相关矩阵传播",
                "温度：固定温度 (L0α)·u(T) 相关传播；"
                "温度区间为硬边界（独立均匀，WC 取半宽，σ=h/√3）",
            ],
            "worst_case_convention":
                "制造半宽热态缩放；u(α)/固定 u(T) 按 ±3σ 扩展；"
                "温度区间取硬半宽",
            "monte_carlo": {
                "samples": model.mc_samples,
                "seed": model.seed,
                "substreams":
                    "SeedSequence([seed, 0x74686572, condition]).spawn(4)："
                    "制造/膨胀系数/温度/完整非线性组合",
                "combined_model":
                    "L=(L0+δ_mfg)[1+(α+Δα)(ΔT+δT)]（完整非线性，同种子复现）",
            },
            "frozen_baseline": "基线链输入、热参数候选与随机种子随版本冻结",
        },
    }
    analysis_id = db.save_thermal_analysis(
        chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), snapshot, result,
        payload.mc_samples, payload.random_seed)
    saved = db.get_thermal_analysis(analysis_id)
    return _thermal_response(saved)


@app.get("/chains/{chain_id}/thermal-analyses", tags=["thermal"])
def list_thermal_analyses(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id,
            "thermal_analyses": db.list_thermal_analyses(chain_id)}


@app.get("/thermal-analyses/{analysis_id}", tags=["thermal"])
def get_thermal_analysis(analysis_id: int) -> dict:
    """读取冻结热分析版本：结果创建时固化，重复读取/计算内容不变。"""
    row = db.get_thermal_analysis(analysis_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"热分析版本 {analysis_id} 不存在")
    return _thermal_response(row)


@app.post("/thermal-analyses/{analysis_id}/proposals", status_code=201,
          tags=["thermal"])
def create_thermal_proposal(analysis_id: int,
                            payload: ThermalProposalRequest) -> dict:
    """按候选材料 / 垫片 / 装配基准温度搜索热整改方案并排列。

    可锁定材料（locked_materials 限定尺寸只能用指定材料）；每个方案给
    候选材料系数、垫片厚度、装配基准温度与改动成本。解析粗筛后用与版本
    同源的固定种子蒙特卡洛复核，按「全工况最差超差率 → 最小规格余量 →
    成本」升序排列。候选表与种子随结果冻结。
    """
    row = db.get_thermal_analysis(analysis_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"热分析版本 {analysis_id} 不存在")
    chain_row = _load_chain(row.chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    model = build_model(
        nc, ThermalAnalysisCreate.model_validate(row.request_json))
    try:
        result = search_proposals(model, payload)
    except ThermalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = {
        **result,
        "thermal_analysis_id": analysis_id,
        "chain_id": row.chain_id,
        "ranking_note":
            "排序键（越小越优）：全工况最差固定种子 MC 超差率 → "
            "全工况最小极值法规格余量（大者优先）→ 改动总成本；"
            "零成本现状方案始终参与复核并列在其中",
        "frozen_inputs": {
            "baseline_chain_id": row.chain_id,
            "thermal_analysis_seed": row.random_seed,
            "candidate_table": payload.model_dump(mode="json"),
        },
    }
    proposal_id = db.save_thermal_proposal(
        analysis_id, row.chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    saved = db.get_thermal_proposal(proposal_id)
    return {
        "proposal_id": saved.id,
        "thermal_analysis_id": analysis_id,
        "chain_id": row.chain_id,
        "name": saved.name,
        "note": saved.note,
        "created_at": saved.created_at.isoformat(),
        "submitted_input": saved.request_json,
        "result": saved.result_json,
    }


@app.get("/thermal-analyses/{analysis_id}/proposals", tags=["thermal"])
def list_thermal_proposals(analysis_id: int) -> dict:
    row = db.get_thermal_analysis(analysis_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"热分析版本 {analysis_id} 不存在")
    return {"thermal_analysis_id": analysis_id,
            "proposals": db.list_thermal_proposals(analysis_id)}


@app.get("/thermal-proposals/{proposal_id}", tags=["thermal"])
def get_thermal_proposal(proposal_id: int) -> dict:
    """读取冻结的热整改方案搜索结果。"""
    row = db.get_thermal_proposal(proposal_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"热整改方案 {proposal_id} 不存在")
    return {
        "proposal_id": row.id,
        "thermal_analysis_id": row.thermal_analysis_id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }
