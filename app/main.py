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
from pydantic import ValidationError

from . import database as db
from .assembly import AssemblyError, run_task
from .network import (
    NetworkError,
    compute_network,
    network_from_snapshot,
    network_snapshot,
    normalize_network,
    search_network_scenarios,
)
from .process_plan import (
    ProcessPlanError,
    SolveRequest as ProcessSolveSpec,
    build_plan,
    plan_from_snapshot,
    plan_snapshot,
    solve_plan,
)
from .process_schemas import (
    ProcessPlanCreate,
    ProcessPlanVersionCreate,
    ProcessSelectRequest,
    ProcessSolveRequest,
)
from .engine import (
    _override_chain,
    closure_samples,
    compute_all,
    gap_probability,
    normalize_chain,
)
from .gage_rr import run_study
from .hole_optimizer import RemedyError, freeze_payload, search_remedies
from .hole_schemas import (
    HolePatternCreate,
    HolePatternVersionCreate,
    RemedySearchRequest,
    RemedySelectRequest,
)
from .holes import (
    HoleError,
    analyze as analyze_hole,
    build_model as build_hole_model,
    model_snapshot as hole_model_snapshot,
)
from .inspection import analyze_batch, baseline_comparison, validate_rows
from .drift import DriftError, compare_results as compare_drift_results
from .drift import run_study as run_drift_study
from .drift_schemas import (
    DriftStudyCopyRequest,
    DriftStudyCreate,
    DriftStudyFinalizeRequest,
)
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
    GageRrStudyCreate,
    GapProbabilityRequest,
    GuardBandSpec,
    InspectionBatchCreate,
    MeasurementPlanCreate,
    NetworkCreate,
    NetworkDefinition,
    NetworkScenarioSearchRequest,
    NetworkVersionCreate,
    ScenarioCreate,
    ThermalAnalysisCreate,
    ThermalProposalRequest,
)
from .thermal import ThermalError, analyze as thermal_analyze, build_model, search_proposals
from .units import to_mm
from .wear import WearError
from .wear import analyze as wear_analyze, build_model as build_wear_model
from .wear import model_snapshot as wear_model_snapshot, search_maintenance
from .wear_schemas import (
    MaintenanceSearchRequest,
    MaintenanceSelectRequest,
    WearStudyCreate,
)


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


# -------------------------------------------------------- 量具 R&R 研究

def _study_response(row) -> dict:
    return {
        "study_id": row.id,
        "chain_id": row.chain_id,
        "dimension_id": row.dimension_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "bootstrap_samples": row.bootstrap_samples,
        "random_seed": row.random_seed,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }


@app.post("/chains/{chain_id}/gage-rr-studies", status_code=201,
          tags=["gage-rr"])
def create_gage_rr_study(chain_id: int, payload: GageRrStudyCreate) -> dict:
    """对链上某一尺寸建立量具 R&R 研究（双因素随机效应 ANOVA，创建即冻结）。

    提交 零件×操作者×重复 的完整平衡交叉表（至少 2 零件 × 2 操作者 ×
    2 轮重复，每个零件由所有操作者等次数测量）、单位与过程公差；
    未知尺寸、重复单元、交叉表缺口或重复次数不齐拒绝创建（422）并指出缺口。
    系统拆分设备重复性、操作者再现性、零件×操作者交互与零件间方差
    （负方差分量截为零并保留原估计），返回总量具 R&R、方差占比、
    %Study Variation、%Tolerance、ndc，以及固定种子 bootstrap 95% 置信
    区间与主要变差来源。研究创建后不可修改，历史研究不改写。
    """
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    chain_ids = [d.id for d in nc.dimensions]
    if payload.dimension_id not in chain_ids:
        raise HTTPException(
            status_code=422,
            detail=f"尺寸 {payload.dimension_id!r} 不在基线链 {chain_id} 上"
                   f"（未知尺寸）；链上尺寸: {chain_ids}",
        )
    result = run_study(payload)
    study_id = db.save_gage_rr_study(
        chain_id, payload.dimension_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result,
        payload.bootstrap_samples, payload.random_seed)
    saved = db.get_gage_rr_study(study_id)
    return _study_response(saved)


@app.get("/chains/{chain_id}/gage-rr-studies", tags=["gage-rr"])
def list_gage_rr_studies(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id,
            "studies": db.list_gage_rr_studies(chain_id)}


@app.get("/gage-rr-studies/{study_id}", tags=["gage-rr"])
def get_gage_rr_study(study_id: int) -> dict:
    """读取冻结研究：结果创建时固化，多次读取内容不变。"""
    row = db.get_gage_rr_study(study_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"量具 R&R 研究 {study_id} 不存在")
    return _study_response(row)


# -------------------------------------------------------- 测量方案

@app.post("/chains/{chain_id}/measurement-plans", status_code=201,
          tags=["measurement"])
def create_measurement_plan(chain_id: int, payload: MeasurementPlanCreate) -> dict:
    """建立不可变测量方案：逐尺寸量具误差 + 共用量具相关项。

    统一单位为 mm 后合成标准不确定度；覆盖因子缺失、分量为负、
    相关矩阵非半正定、未覆盖链上全部尺寸时拒绝保存（422）。
    可选 gage_rr_studies 引用冻结量具 R&R 研究：该尺寸的重复性分量
    以研究的总量具标准差取代手填值（其余量具参数仍须补齐），
    研究 id 随方案快照冻结；研究错链 / 不存在 / 尺寸不匹配时拒绝。
    方案创建后不可修改，新版本请另建方案（历史批次引用不受影响）。
    """
    row = _load_chain(chain_id)
    nc = rebuild_normalized(row.request_json)
    gage_rr_refs: dict[str, dict] = {}
    for dim_id, study_id in payload.gage_rr_studies.items():
        srow = db.get_gage_rr_study(study_id)
        if srow is None or srow.chain_id != chain_id:
            raise HTTPException(
                status_code=404,
                detail=f"链 {chain_id} 下量具 R&R 研究 {study_id} 不存在",
            )
        if srow.dimension_id != dim_id:
            raise HTTPException(
                status_code=422,
                detail=f"量具 R&R 研究 {study_id} 针对尺寸 "
                       f"{srow.dimension_id!r}，不能用于尺寸 {dim_id!r}",
            )
        gage_rr_refs[dim_id] = {
            "study_id": srow.id,
            "name": srow.name,
            "total_gage_std_mm": srow.result_json["total_gage_std_mm"],
        }
    try:
        combined = normalize_plan(nc, payload, gage_rr=gage_rr_refs)
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
                    "L=(L0+δ_mfg)[1+(α+Δα)(ΔT+δT)]（完整非线性，同种子复现）；"
                    "制造偏差 δ 仅由括号内 1+αΔT 缩放一次，不重复缩放",
                "component_reject":
                    "分量样本为零均值单项偏差，其超差率在「热态名义间隙 + "
                    "单项偏差」上对照规格，而非用绝对间隙规格直接判定零均值偏差",
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


# -------------------------------------------------------- 多闭环公差网络

def _merge_network_pool(payload) -> tuple[list[dict], dict]:
    """合并共享尺寸池：冻结基线链导入（只读）+ 调用方追加尺寸。

    返回 (合并后的尺寸提交表示列表, 来源记录)；基线不存在 404，
    追加尺寸与导入尺寸 id 冲突 422。
    """
    imported: list[dict] = []
    source: dict[str, Any] = {
        "chain_id": None,
        "imported_dimension_ids": [],
        "submitted_dimension_ids": [d.id for d in payload.dimensions],
    }
    if payload.source_chain_id is not None:
        chain_row = db.get_chain(payload.source_chain_id)
        if chain_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"来源基线链 {payload.source_chain_id} 不存在",
            )
        imported = [dict(d) for d in chain_row.request_json["dimensions"]]
        source["chain_id"] = chain_row.id
        source["chain_name"] = chain_row.name
        source["imported_dimension_ids"] = [d["id"] for d in imported]

    imported_ids = {d["id"] for d in imported}
    submitted_ids = [d.id for d in payload.dimensions]
    conflict = sorted(imported_ids & set(submitted_ids))
    if conflict:
        raise HTTPException(
            status_code=422,
            detail=f"追加的共享尺寸与基线链 {payload.source_chain_id} 导入的"
                   f"尺寸 id 冲突: {conflict}；如需改值请另选 id 或改用纯"
                   "尺寸池定义",
        )
    merged = imported + [d.model_dump(mode="json") for d in payload.dimensions]
    source["pool_dimension_ids"] = source["imported_dimension_ids"] + submitted_ids
    return merged, source


def _validate_network_definition(dimensions, loops, correlations,
                                 ) -> NetworkDefinition:
    """第二阶段 Pydantic 校验（路径断开/方向不衔接/未知重复尺寸/矩阵）。"""
    try:
        return NetworkDefinition.model_validate({
            "dimensions": dimensions,
            "loops": [loop.model_dump(mode="json") for loop in loops],
            "correlations": [c.model_dump(mode="json") for c in correlations],
        })
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_clean_validation_errors(exc),
        ) from exc


def _clean_validation_errors(exc: ValidationError) -> list[dict]:
    """把 ValidationError 详情清洗为可 JSON 序列化（ctx 中的异常转字符串）。"""
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

    return clean(exc.errors(include_url=False))


def _build_network(payload, name: str, note: str):
    """合并尺寸池 → 校验 → 归一化 → 计算，返回 (net, result, snapshot, request_store)。"""
    merged, source = _merge_network_pool(payload)
    definition = _validate_network_definition(
        merged, payload.loops, payload.correlations)
    net = normalize_network(
        definition, payload.mc_samples, payload.random_seed, name=name)
    result = compute_network(net)
    snapshot = network_snapshot(net, source)
    request_store = {
        **payload.model_dump(mode="json"),
        "resolved_pool_dimension_ids": source["pool_dimension_ids"],
    }
    return net, result, snapshot, request_store, source


def _network_version_payload(net_row, version_row) -> dict:
    return {
        "network_id": net_row.id,
        "network_name": net_row.name,
        "version_id": version_row.id,
        "version_no": version_row.version_no,
        "parent_version_id": version_row.parent_version_id,
        "name": version_row.name,
        "note": version_row.note,
        "frozen": True,
        "baseline_preserved": True,
        "created_at": version_row.created_at.isoformat(),
        "source": version_row.snapshot_json["source"],
        "submitted_input": version_row.request_json,
        "snapshot": version_row.snapshot_json,
        "result": version_row.result_json,
    }


@app.post("/networks", status_code=201, tags=["networks"])
def create_network(payload: NetworkCreate) -> dict:
    """创建多闭环公差网络（版本 1）：同一套装配共享尺寸的多个功能要求。

    可从冻结基线链导入组成尺寸（source_chain_id，基线只读不改写），也可
    直接提交共享尺寸池（或两者合并，id 不得冲突）；为 2~20 个闭环设置
    有序路径、沿边方向、上下限与优先级。路径断开、方向不衔接、未知或
    重复尺寸、相关矩阵非半正定均拒绝（422）。蒙特卡洛对共享尺寸每轮只
    抽样一次，返回各闭环极值边界 / RSS 区间 / 超差率，以及闭环间协方差、
    联合合格率、同时失效组合与尺寸×闭环敏感度矩阵。版本快照冻结入库。
    """
    net, result, snapshot, request_store, _ = _build_network(
        payload, payload.name, payload.note)
    network_id, version_id = db.save_network_first_version(
        payload.name, payload.note, payload.source_chain_id,
        request_store, snapshot, result,
        payload.mc_samples, payload.random_seed)
    net_row = db.get_network(network_id)
    version_row = db.get_network_version(version_id)
    return _network_version_payload(net_row, version_row)


@app.get("/networks", tags=["networks"])
def list_networks() -> dict:
    return {"networks": db.list_networks()}


def _load_network(network_id: int):
    row = db.get_network(network_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"网络 {network_id} 不存在")
    return row


@app.get("/networks/{network_id}", tags=["networks"])
def get_network(network_id: int) -> dict:
    row = _load_network(network_id)
    versions = db.list_network_versions(network_id)
    return {
        "network_id": row.id,
        "name": row.name,
        "note": row.note,
        "source_chain_id": row.source_chain_id,
        "created_at": row.created_at.isoformat(),
        "versions": [
            {
                "version_id": v.id,
                "version_no": v.version_no,
                "parent_version_id": v.parent_version_id,
                "note": v.note,
                "mc_samples": v.mc_samples,
                "random_seed": v.random_seed,
                "loop_ids": [loop["loop_id"]
                             for loop in v.snapshot_json["loops"]],
                "joint_pass_rate_monte_carlo":
                    v.result_json["joint"]["joint_pass_rate_monte_carlo"],
                "all_loops_in_spec_worst_case":
                    v.result_json["joint"]["all_loops_in_spec_worst_case"],
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ],
    }


@app.post("/networks/{network_id}/versions", status_code=201, tags=["networks"])
def create_network_version(network_id: int,
                           payload: NetworkVersionCreate) -> dict:
    """在网络下另建独立版本（完整新定义随版本冻结，历史版本不回改）。

    parent_version_id 缺省取当前最新版本（仅记录血缘）；来源基线链只读。
    """
    net_row = _load_network(network_id)
    versions = db.list_network_versions(network_id)
    if payload.parent_version_id is not None:
        parent_ids = {v.id for v in versions}
        if payload.parent_version_id not in parent_ids:
            raise HTTPException(
                status_code=404,
                detail=f"网络 {network_id} 下父版本 "
                       f"{payload.parent_version_id} 不存在",
            )
        parent_id = payload.parent_version_id
    else:
        parent_id = versions[-1].id if versions else None

    net, result, snapshot, request_store, _ = _build_network(
        payload, net_row.name, payload.note)
    new_no = max((v.version_no for v in versions), default=0) + 1
    version_id = db.save_network_version(
        network_id, new_no, parent_id, net_row.name, payload.note,
        request_store, snapshot, result,
        payload.mc_samples, payload.random_seed)
    return _network_version_payload(net_row, db.get_network_version(version_id))


def _load_network_version(version_id: int):
    row = db.get_network_version(version_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"网络版本 {version_id} 不存在")
    return row


@app.get("/network-versions/{version_id}", tags=["networks"])
def get_network_version(version_id: int) -> dict:
    """读取冻结网络版本：快照与结果创建时固化，重复读取内容不变。"""
    version_row = _load_network_version(version_id)
    net_row = db.get_network(version_row.network_id)
    return _network_version_payload(net_row, version_row)


@app.post("/network-versions/{version_id}/scenarios", status_code=201,
          tags=["networks"])
def create_network_scenario(version_id: int,
                            payload: NetworkScenarioSearchRequest) -> dict:
    """方案分支搜索：锁定尺寸并批量调整公差/标准差，结果排序后冻结。

    未锁定尺寸在 scale_levels 网格上逐尺寸选择公差带比例（σ 策略与单链
    批量调整同口径）；beam search 以联合超差率代理剪枝，候选用共享抽样
    固定种子蒙特卡洛复核，按「全部闭环达标 → 最高优先级余量（大者优先）
    → 收紧成本（升序）」排列。版本快照与来源基线不改写。
    """
    version_row = _load_network_version(version_id)
    net = network_from_snapshot(version_row.snapshot_json)
    try:
        result = search_network_scenarios(net, payload)
    except NetworkError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scenario_id = db.save_network_scenario(
        version_row.network_id, version_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    saved = db.get_network_scenario(scenario_id)
    return _network_scenario_payload(saved)


@app.get("/network-versions/{version_id}/scenarios", tags=["networks"])
def list_network_scenarios(version_id: int) -> dict:
    _load_network_version(version_id)
    return {"version_id": version_id,
            "scenarios": db.list_network_scenarios(version_id)}


def _network_scenario_payload(row) -> dict:
    return {
        "scenario_id": row.id,
        "network_id": row.network_id,
        "version_id": row.version_id,
        "name": row.name,
        "note": row.note,
        "frozen": True,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }


@app.get("/network-scenarios/{scenario_id}", tags=["networks"])
def get_network_scenario(scenario_id: int) -> dict:
    """读取冻结的网络方案分支搜索结果。"""
    row = db.get_network_scenario(scenario_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"网络方案 {scenario_id} 不存在")
    return _network_scenario_payload(row)


# -------------------------------------------------------- 工序尺寸方案

def _design_dims_from_chain(chain_row) -> tuple[list[dict], list[str]]:
    """从冻结基线链导入独立设计尺寸（规范化 mm 快照；只读，不改写基线）。

    基线链是单一闭合环：所有边都映射到方案表面时存在一个线性相关（封闭环
    关系 Σ s_i·L_i = C0），导入时按行阶梯剔除一条可由其余尺寸线性表示的
    边（通常是反向封闭边），只保留独立设计尺寸。
    返回 (设计尺寸列表, 被剔除的闭环边 id)。
    """
    raw = []
    for d in chain_row.result_json["normalized_inputs"]:
        nm = d["normalized_mm"]
        raw.append({
            "id": d["dimension_id"],
            "start": d["edge"][0],
            "end": d["edge"][1],
            "nominal": nm["nominal"],
            "upper_deviation": nm["upper_deviation"],
            "lower_deviation": nm["lower_deviation"],
            "unit": "mm",
            "sign": int(d["sign"]),
            "source": "chain",
            "note": f"导入自基线链 {chain_row.id}（{chain_row.name}）",
        })

    all_surfaces = sorted({s for r_ in raw for s in (r_["start"], r_["end"])})
    aidx = {s: i for i, s in enumerate(all_surfaces)}

    def row_vec(r_):
        v = np.zeros(len(all_surfaces))
        v[aidx[r_["start"]]] = -r_["sign"]
        v[aidx[r_["end"]]] = r_["sign"]
        return v

    chosen: list[dict] = []
    bmat = np.zeros((0, len(all_surfaces)))
    dropped: list[str] = []
    for dd in raw:
        vv = row_vec(dd)
        rank_before = int(np.linalg.matrix_rank(bmat, tol=1e-8))
        rank_after = int(np.linalg.matrix_rank(
            np.vstack([bmat, vv]) if len(chosen) else vv[None, :], tol=1e-8))
        if rank_after == rank_before:
            dropped.append(dd["id"])
            continue
        chosen.append(dd)
        bmat = np.vstack([bmat, vv]) if len(chosen) > 1 else vv[None, :]
    if dropped:
        for dd in chosen:
            dd["note"] += f"；闭合环冗余边 {dropped} 已按线性相关剔除（封闭环）"
    return chosen, dropped


def _design_dims_inline(payload) -> list[dict]:
    dims = []
    for d in payload.design_dimensions:
        u = d.unit.value
        dims.append({
            "id": d.id,
            "start": d.start_surface,
            "end": d.end_surface,
            "nominal": to_mm(d.nominal, u),
            "upper_deviation": to_mm(d.upper_deviation, u),
            "lower_deviation": to_mm(d.lower_deviation, u),
            "unit": u,
            "sign": int(d.sign),
            "source": "inline",
            "note": d.note,
        })
    return dims


def _resolve_process_design(payload):
    """返回 (design_dims_mm, source_record, source_chain_id)。"""
    if payload.source_chain_id is not None:
        chain_row = db.get_chain(payload.source_chain_id)
        if chain_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"来源基线链 {payload.source_chain_id} 不存在")
        dims, dropped = _design_dims_from_chain(chain_row)
        source = {
            "type": "chain",
            "chain_id": chain_row.id,
            "chain_name": chain_row.name,
            "snapshot_policy": "设计尺寸随工序方案版本冻结；基线链后续更新"
                               "不改写历史方案",
            "imported_dimension_ids": [d["id"] for d in dims],
            "dropped_closure_edges": dropped,
        }
        return dims, source, chain_row.id
    dims = _design_dims_inline(payload)
    source = {"type": "inline",
              "dimension_ids": [d["id"] for d in dims]}
    return dims, source, None


def _build_process_version(payload, name: str, note: str, seed: int):
    """校验 → 构建 → 快照，返回 (plan, snapshot, request_store, source, chain_id)。"""
    # Pydantic 已做字段级校验；结构校验在 build_plan 中完成
    design_dims, source, chain_id = _resolve_process_design(payload)
    try:
        plan = build_plan(payload, design_dims)
    except ProcessPlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    snapshot = plan_snapshot(plan, source)
    request_store = {
        **payload.model_dump(mode="json"),
        "random_seed": seed,
        "mc_samples": payload.mc_samples,
    }
    return plan, snapshot, request_store, source, chain_id


def _process_version_payload(plan_row, version_row) -> dict:
    return {
        "plan_id": plan_row.id,
        "plan_name": plan_row.name,
        "plan_note": plan_row.note,
        "version_id": version_row.id,
        "version_no": version_row.version_no,
        "parent_version_id": version_row.parent_version_id,
        "name": version_row.name,
        "note": version_row.note,
        "frozen": True,
        "created_at": version_row.created_at.isoformat(),
        "mc_samples": version_row.mc_samples,
        "random_seed": version_row.random_seed,
        "source": version_row.snapshot_json["source"],
        "submitted_input": version_row.request_json,
        "snapshot": version_row.snapshot_json,
        "transfer_matrix": version_row.snapshot_json["transfer_matrix"],
    }


@app.post("/process-plans", status_code=201, tags=["process-plans"])
def create_process_plan(payload: ProcessPlanCreate) -> dict:
    """创建工序尺寸方案（一次零件加工路线独立成版，版本 1）。

    依次记录毛坯面、每道工序的定位基准/被加工面与工序尺寸名义值、公差、
    分布与制造成本；引用不可变设计尺寸链（冻结基线链只读导入或直接提交）。
    系统按有向表面关系生成工序尺寸到设计闭环的传递矩阵。工序引用尚未形成
    的表面、基准路径断开、重复约束、传递矩阵秩不足（工序自由度不受设计链
    约束）一律拒绝（422）并指出相关工序与自由度。版本快照（设计链/路线/
    矩阵/种子）写入 SQLite，来源更新不改写历史方案。
    """
    plan, snapshot, request_store, _, _ = _build_process_version(
        payload, payload.name, payload.note, payload.random_seed)
    plan_id, version_id = db.save_process_plan_first_version(
        payload.name, payload.note, payload.source_chain_id,
        request_store, snapshot, payload.mc_samples, payload.random_seed)
    plan_row = db.get_process_plan(plan_id)
    version_row = db.get_process_plan_version(version_id)
    return _process_version_payload(plan_row, version_row)


@app.get("/process-plans", tags=["process-plans"])
def list_process_plans() -> dict:
    return {"process_plans": db.list_process_plans()}


def _load_process_plan(plan_id: int):
    row = db.get_process_plan(plan_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"工序尺寸方案 {plan_id} 不存在")
    return row


@app.get("/process-plans/{plan_id}", tags=["process-plans"])
def get_process_plan(plan_id: int) -> dict:
    row = _load_process_plan(plan_id)
    versions = db.list_process_plan_versions(plan_id)
    return {
        "plan_id": row.id,
        "name": row.name,
        "note": row.note,
        "source_chain_id": row.source_chain_id,
        "created_at": row.created_at.isoformat(),
        "versions": [
            {
                "version_id": v.id,
                "version_no": v.version_no,
                "parent_version_id": v.parent_version_id,
                "name": v.name,
                "note": v.note,
                "mc_samples": v.mc_samples,
                "random_seed": v.random_seed,
                "operations": len(v.snapshot_json["edges"]),
                "closures": len(v.snapshot_json["closures"]),
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ],
    }


@app.post("/process-plans/{plan_id}/versions", status_code=201,
          tags=["process-plans"])
def create_process_plan_version(plan_id: int,
                                payload: ProcessPlanVersionCreate) -> dict:
    """在同一方案下另建独立版本（完整新加工路线随版本冻结，历史不回改）。

    parent_version_id 仅记血缘（缺省取最新版本）；来源基线链只读快照。
    """
    plan_row = _load_process_plan(plan_id)
    versions = db.list_process_plan_versions(plan_id)
    parent_id = versions[-1].id if versions else None
    seed = (payload.random_seed if payload.random_seed is not None
            else versions[-1].random_seed if versions else 20260913)
    payload_with_seed = payload.model_copy(update={"random_seed": seed})
    _, snapshot, request_store, _, chain_id = _build_process_version(
        payload_with_seed, payload.name or plan_row.name, payload.note, seed)
    new_no = max((v.version_no for v in versions), default=0) + 1
    version_id = db.save_process_plan_version(
        plan_id, new_no, parent_id, payload.name or plan_row.name,
        payload.note, request_store, snapshot, payload.mc_samples, seed)
    return _process_version_payload(plan_row,
                                    db.get_process_plan_version(version_id))


def _load_process_version(version_id: int):
    row = db.get_process_plan_version(version_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"工序方案版本 {version_id} 不存在")
    return row


@app.get("/process-plan-versions/{version_id}", tags=["process-plans"])
def get_process_plan_version(version_id: int) -> dict:
    """读取冻结工序方案版本：快照/传递矩阵创建时固化，重复读取不变。"""
    version_row = _load_process_version(version_id)
    plan_row = _load_process_plan(version_row.plan_id)
    return _process_version_payload(plan_row, version_row)


def _solve_spec_from_request(plan, payload: ProcessSolveRequest,
                             seed: int) -> ProcessSolveSpec:
    grades = {t.id: float(t.grade_factor)
              for t in payload.capability_tiers}
    grade_costs = {t.id: float(t.setup_cost)
                   for t in payload.capability_tiers}
    locked = {item.edge_id: item.nominal
              for item in payload.locked_dimensions}
    edge_ids = {e.id for e in plan.edges}
    unknown = sorted(set(locked) - edge_ids)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"锁定尺寸不在方案中: {unknown}；方案边: {plan.ids}")
    step_refs = set(payload.edge_step_sizes) | set(payload.edge_grades) \
        | set(payload.edge_grade_options)
    bad_steps = sorted(step_refs - edge_ids)
    if bad_steps:
        raise HTTPException(
            status_code=422,
            detail=f"步进/档位限定引用了方案外工序: {bad_steps}")
    return ProcessSolveSpec(
        locked=locked, grades=grades, grade_costs=grade_costs,
        default_grade=payload.default_grade or next(iter(grades)),
        edge_grades=dict(payload.edge_grades),
        edge_grade_list={k: list(v) for k, v in
                         payload.edge_grade_options.items()},
        step_sizes=[float(v) for v in payload.standard_step_sizes],
        edge_steps={k: float(v) for k, v in payload.edge_step_sizes.items()},
        mc_samples=payload.mc_samples, seed=seed,
        max_candidates=payload.max_candidates)


@app.post("/process-plan-versions/{version_id}/solve", status_code=201,
          tags=["process-plans"])
def solve_process_version(version_id: int,
                          payload: ProcessSolveRequest) -> dict:
    """锁定已定工序尺寸、反算其余名义，传播 WC/RSS/固定种子 MC 并枚举多解。

    返回名义值范围（两段单纯形 LP）、逐项代数消元过程、各闭环误差贡献，
    以及按「设计闭环达标数 → 最差余量 → 尺寸改动量 → 制造成本」排序的
    档位×步进候选。锁定后自由度仍欠约束（名义无界）时拒绝（422）并指出
    相关工序与自由度。结果随请求冻结，来源更新不改写。
    """
    version_row = _load_process_version(version_id)
    plan = plan_from_snapshot(version_row.snapshot_json)
    seed = (payload.random_seed if payload.random_seed is not None
            else version_row.random_seed)
    spec = _solve_spec_from_request(plan, payload, seed)
    try:
        result = solve_plan(plan, spec)
    except ProcessPlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["frozen_inputs"] = {
        "version_id": version_id,
        "plan_id": version_row.plan_id,
        "design_chain_snapshot": version_row.snapshot_json["design_chain_snapshot"],
        "route_snapshot": "工序路线与传递矩阵引用冻结版本快照",
        "random_seed": seed,
        "mc_samples": payload.mc_samples,
    }
    solution_id = db.save_process_solution(
        version_row.plan_id, version_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    saved = db.get_process_solution(solution_id)
    return _process_solution_payload(saved)


@app.get("/process-plan-versions/{version_id}/solutions", tags=["process-plans"])
def list_process_solutions(version_id: int) -> dict:
    _load_process_version(version_id)
    return {"version_id": version_id,
            "solutions": db.list_process_solutions(version_id)}


def _process_solution_payload(row) -> dict:
    return {
        "solution_id": row.id,
        "plan_id": row.plan_id,
        "version_id": row.version_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "selected_rank": row.selected_rank,
        "selected_note": row.selected_note,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }


@app.get("/process-solutions/{solution_id}", tags=["process-plans"])
def get_process_solution(solution_id: int) -> dict:
    """读取冻结的反算结果（选定前可重看候选；选定后结果不可改）。"""
    row = db.get_process_solution(solution_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"反算结果 {solution_id} 不存在")
    return _process_solution_payload(row)


@app.post("/process-solutions/{solution_id}/select", tags=["process-plans"])
def select_process_candidate(solution_id: int,
                             payload: ProcessSelectRequest) -> dict:
    """选定一个候选并冻结：固化设计链快照、工序路线、传递矩阵与随机种子。

    每个反算结果只能选定一次（重复选定或改选返回 422）；历史方案不改写。
    """
    row = db.get_process_solution(solution_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"反算结果 {solution_id} 不存在")
    if row.frozen:
        raise HTTPException(
            status_code=422,
            detail=f"反算结果 {solution_id} 已选定候选 rank="
                   f"{row.selected_rank} 并冻结，不能改选；请在冻结版本上"
                   "另建反算结果或新版本")
    candidates = row.result_json.get("candidates", [])
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=f"反算结果 {solution_id} 无可行候选（status="
                   f"{row.result_json.get('status')}），无法选定")
    if payload.rank > len(candidates):
        raise HTTPException(
            status_code=422,
            detail=f"rank={payload.rank} 超出候选数 {len(candidates)}")
    db.freeze_process_solution(solution_id, payload.rank, payload.note)
    saved = db.get_process_solution(solution_id)
    answer = _process_solution_payload(saved)
    chosen = next(c for c in saved.result_json["candidates"]
                  if c["rank"] == payload.rank)
    answer["selected_candidate"] = chosen
    answer["freeze_note"] = (
        "已冻结设计链快照、工序路线、传递矩阵与随机种子；来源链更新不"
        "改写本方案。后续工序指导应引用本反算结果与版本快照。")
    return answer


# -------------------------------------------------------- 孔系装配分析

def _hole_version_payload(pattern_row, version_row) -> dict:
    return {
        "pattern_id": pattern_row.id,
        "pattern_name": pattern_row.name,
        "version_id": version_row.id,
        "version_no": version_row.version_no,
        "parent_version_id": version_row.parent_version_id,
        "name": version_row.name,
        "note": version_row.note,
        "frozen": True,
        "created_at": version_row.created_at.isoformat(),
        "submitted_input": version_row.request_json,
        "snapshot": version_row.snapshot_json,
        "result": version_row.result_json,
        "mc_samples": version_row.mc_samples,
        "random_seed": version_row.random_seed,
    }


@app.post("/hole-patterns", status_code=201, tags=["holes"])
def create_hole_pattern(payload: HolePatternCreate) -> dict:
    """创建孔系装配分析（一对零件的孔 / 销 / 螺栓 + 两侧基准框架，版本 1）。

    Pydantic 逐匹配位校验名义坐标、孔径与连接件直径极限（倒置拒绝）、
    位置度与 MMC/LMC/RFS 实体条件；基准框架校验优先次序、基准要素尺寸
    与实体条件（自由度重复/欠约束、平行边线退化、位置度引用无效拒绝）；
    匹配缺失（重复 id / B 侧空缺 / 双外要素 / 双孔无螺栓）指出对象并拒绝。
    系统合并尺寸偏差、实体状态补偿公差与基准偏移，在最坏边界（VC 圆盘
    公共交集）与固定种子蒙特卡洛中求能容纳全部连接件的平移与转角，
    返回装配成功率、可行位姿范围、最先干涉匹配位与逐项余量贡献。
    规范化 mm 快照与随机种子随版本冻结写入 SQLite。
    """
    try:
        result = analyze_hole(payload)
    except HoleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model = build_hole_model(payload)
    snapshot = hole_model_snapshot(model)
    pattern_id, version_id = db.save_hole_first_version(
        payload.name, payload.note, payload.model_dump(mode="json"),
        snapshot, result, payload.mc_samples, payload.random_seed)
    return _hole_version_payload(
        db.get_hole_pattern(pattern_id), db.get_hole_version(version_id))


@app.get("/hole-patterns", tags=["holes"])
def list_hole_patterns() -> dict:
    return {"hole_patterns": db.list_hole_patterns()}


def _load_hole_version(version_id: int):
    row = db.get_hole_version(version_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"孔系版本 {version_id} 不存在")
    return row


@app.get("/hole-versions/{version_id}", tags=["holes"])
def get_hole_version(version_id: int) -> dict:
    """读取冻结孔系版本：快照与结果创建时固化，重复读取内容不变。"""
    version_row = _load_hole_version(version_id)
    pattern_row = db.get_hole_pattern(version_row.pattern_id)
    return _hole_version_payload(pattern_row, version_row)


@app.post("/hole-patterns/{pattern_id}/versions", status_code=201,
          tags=["holes"])
def create_hole_version(pattern_id: int,
                        payload: HolePatternVersionCreate) -> dict:
    """在同一孔系对象下另建独立版本（完整新定义随版本冻结，历史不回改）。

    parent_version_id 仅记血缘（缺省取最新版本）。
    """
    pattern_row = db.get_hole_pattern(pattern_id)
    if pattern_row is None:
        raise HTTPException(status_code=404,
                            detail=f"孔系对象 {pattern_id} 不存在")
    versions = db.list_hole_versions(pattern_id)
    if payload.parent_version_id is not None:
        parent_ids = {v.id for v in versions}
        if payload.parent_version_id not in parent_ids:
            raise HTTPException(
                status_code=404,
                detail=f"孔系 {pattern_id} 下父版本 "
                       f"{payload.parent_version_id} 不存在")
        parent_id = payload.parent_version_id
    else:
        parent_id = versions[-1].id if versions else None
    try:
        result = analyze_hole(payload)
    except HoleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model = build_hole_model(payload)
    snapshot = hole_model_snapshot(model)
    new_no = max((v.version_no for v in versions), default=0) + 1
    version_id = db.save_hole_version(
        pattern_id, new_no, parent_id, payload.name, payload.note,
        payload.model_dump(mode="json"), snapshot, result,
        payload.mc_samples, payload.random_seed)
    return _hole_version_payload(pattern_row,
                                 db.get_hole_version(version_id))


@app.post("/hole-versions/{version_id}/remedies", status_code=201,
          tags=["holes"])
def create_hole_remedy(version_id: int,
                       payload: RemedySearchRequest) -> dict:
    """搜索整改组合：候选钻孔尺寸 / 连接件规格 / 允许孔位修正。

    可锁定孔位（locked_mates，禁孔位修正）或连接件（locked_fasteners，
    禁换规格）；候选引用不存在匹配位、对非孔要素钻孔、候选孔径不放大、
    锁与候选冲突时拒绝（422）并指出对象。所有候选共用固定种子同源样本，
    按「失败概率 → 最大孔位改动 → 孔径放大总量」升序排列，零改动现状
    方案始终参与复核。候选表与种子随结果冻结。
    """
    version_row = _load_hole_version(version_id)
    parent_payload = HolePatternCreate.model_validate(
        version_row.request_json)
    model = build_hole_model(parent_payload)
    try:
        result = search_remedies(model, payload)
    except RemedyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["frozen_inputs"] = {
        "version_id": version_id,
        "pattern_id": version_row.pattern_id,
        "parent_random_seed": version_row.random_seed,
        "candidate_table": payload.model_dump(mode="json"),
    }
    remedy_id = db.save_hole_remedy(
        version_row.pattern_id, version_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    saved = db.get_hole_remedy(remedy_id)
    return {
        "remedy_id": saved.id,
        "pattern_id": saved.pattern_id,
        "version_id": saved.version_id,
        "name": saved.name,
        "note": saved.note,
        "created_at": saved.created_at.isoformat(),
        "submitted_input": saved.request_json,
        "result": saved.result_json,
    }


@app.get("/hole-versions/{version_id}/remedies", tags=["holes"])
def list_hole_remedies(version_id: int) -> dict:
    _load_hole_version(version_id)
    return {"version_id": version_id,
            "remedies": db.list_hole_remedies(version_id)}


@app.get("/hole-remedies/{remedy_id}", tags=["holes"])
def get_hole_remedy(remedy_id: int) -> dict:
    row = db.get_hole_remedy(remedy_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"孔系整改结果 {remedy_id} 不存在")
    return {
        "remedy_id": row.id,
        "pattern_id": row.pattern_id,
        "version_id": row.version_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "selected_rank": row.selected_rank,
        "selected_version_id": row.selected_version_id,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }


@app.post("/hole-remedies/{remedy_id}/select", tags=["holes"])
def select_hole_remedy(remedy_id: int, payload: RemedySelectRequest) -> dict:
    """采纳整改候选并冻结：按候选放大孔径 / 更换连接件创建新版本。

    新版本冻结输入孔系、匹配关系与随机种子；每个整改结果只能采纳一次
    （重复采纳或改选返回 422），历史版本不改写。
    """
    remedy_row = db.get_hole_remedy(remedy_id)
    if remedy_row is None:
        raise HTTPException(status_code=404,
                            detail=f"孔系整改结果 {remedy_id} 不存在")
    if remedy_row.frozen:
        raise HTTPException(
            status_code=422,
            detail=f"整改结果 {remedy_id} 已采纳 rank="
                   f"{remedy_row.selected_rank} 并冻结（新版本 "
                   f"{remedy_row.selected_version_id}），不能改选；"
                   "请在新版本上另建整改搜索")
    candidates = remedy_row.result_json.get("candidates", [])
    if payload.rank > len(candidates):
        raise HTTPException(
            status_code=422,
            detail=f"rank={payload.rank} 超出候选数 {len(candidates)}")
    chosen = next(c for c in candidates if c["rank"] == payload.rank)
    parent_version = db.get_hole_version(remedy_row.version_id)
    parent_payload = HolePatternCreate.model_validate(
        parent_version.request_json)
    model = build_hole_model(parent_payload)
    adopted_name = f"{remedy_row.name}-采纳"
    # 采纳版本与整改搜索同抽样规模 / 同种子，保证重算失败概率与候选一致
    remedy_samples = int(remedy_row.result_json.get("samples")
                         or parent_payload.mc_samples)
    remedy_seed = int(remedy_row.result_json["random_seed"]) \
        if remedy_row.result_json.get("random_seed") is not None \
        else parent_payload.random_seed
    new_payload = freeze_payload(
        model, chosen, adopted_name, payload.note or remedy_row.note,
        parent_payload, mc_samples=remedy_samples, seed=remedy_seed)
    result = analyze_hole(new_payload)
    new_model = build_hole_model(new_payload)
    pattern_versions = db.list_hole_versions(remedy_row.pattern_id)
    new_no = max(v.version_no for v in pattern_versions) + 1
    adopted_meta = {
        "remedy_id": remedy_row.id,
        "rank": payload.rank,
        "drill": chosen["drill"],
        "fasteners": chosen["fasteners"],
        "uses_hole_correction": chosen["uses_hole_correction"],
        "hole_correction_budget_mm": chosen["hole_correction_budget_mm"],
        "hole_correction_vectors_mm": chosen.get(
            "hole_correction_vectors_mm", []),
        "monte_carlo": {"samples": remedy_samples, "seed": remedy_seed},
        "frozen": "采纳结果冻结输入孔系、匹配关系与随机种子",
    }
    request_store = new_payload.model_dump(mode="json")
    request_store["adopted_remedy"] = adopted_meta
    result = {**result, "adopted_remedy": adopted_meta}
    new_version_id = db.save_hole_version(
        remedy_row.pattern_id, new_no, parent_version.id,
        adopted_name, payload.note, request_store,
        hole_model_snapshot(new_model), result,
        remedy_samples, remedy_seed)
    db.freeze_hole_remedy(remedy_row.id, payload.rank, payload.note,
                          new_version_id)
    saved_remedy = db.get_hole_remedy(remedy_row.id)
    answer = _hole_version_payload(
        db.get_hole_pattern(remedy_row.pattern_id),
        db.get_hole_version(new_version_id))
    answer["adopted_from"] = {
        "remedy_id": remedy_row.id,
        "rank": payload.rank,
        "selected_candidate": chosen,
        "frozen_remedy": bool(saved_remedy.frozen),
        "freeze_note": "已冻结输入孔系、匹配关系与随机种子；历史版本不改写",
    }
    return answer


# -------------------------------------------------------- 服役磨损研究

def _wear_study_response(row) -> dict:
    return {
        "study_id": row.id,
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


@app.post("/chains/{chain_id}/wear-studies", status_code=201, tags=["wear"])
def create_wear_study(chain_id: int, payload: WearStudyCreate) -> dict:
    """从冻结基线链建立服役磨损研究（独立成版，创建即冻结）。

    逐尺寸填写磨损方向（增大/减小）、分段线性累计磨损曲线（首段从
    (0,0) 出发、相邻段折点衔接）与磨损速率标准差；共用载荷导致的速率
    相关用相关矩阵声明。计算节点为严格递增的循环数序列，不得超出任何
    尺寸曲线覆盖范围；研究封闭环限值独立设置（至少给一侧）。漏填/链外
    尺寸、曲线不衔接、相关矩阵非半正定、节点越界均拒绝（422）。

    每个节点叠加初始制造散布（基线分布与相关）与累计磨损（随机速率
    模型：σ(N)=速率σ×N），给出极值界、RSS、固定种子蒙特卡洛超差率、
    首次越界循环分布与逐尺寸方差贡献。基线链不变，结果与种子随研究
    冻结（同一研究重复计算不变）。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    try:
        model = build_wear_model(nc, payload)
        result = wear_analyze(model)
    except WearError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    snapshot = wear_model_snapshot(model)
    result = {
        **result,
        "closure_spec_mm": {
            "lower_limit": model.lsl,
            "upper_limit": model.usl,
            "original": {
                "lower_limit": payload.closure_lower_limit,
                "upper_limit": payload.closure_upper_limit,
                "unit": payload.closure_unit.value,
            },
        },
        "traceability": {
            "baseline_chain_id": chain_id,
            "baseline_chain_name": chain_row.name,
            "baseline_normalized_inputs":
                chain_row.result_json["normalized_inputs"],
            "formula": "C(N) = C(0) + Σ s_i·d_i·[μ_i(N) + σ_ri·N·Z_i]",
            "wear_model": (
                "随机速率模型：同一零件整个服役期速率偏差恒定，N 循环后"
                "累计磨损标准差 = 速率σ × N；μ_i(N) 为分段线性累计磨损"
                "曲线插值；d_i=±1 为磨损方向，s_i 为闭环方向系数"),
            "uncertainty_sources": [
                "初始制造散布：基线链分布/相关原样复用（复用基线抽样器）",
                "磨损速率：σ_ri 按共用载荷相关矩阵相关传播（Cholesky）",
            ],
            "worst_case_convention":
                "制造散布取公差带线性累加 ΣT_i；磨损不确定度按 ±3σ 扩展",
            "monte_carlo": {
                "samples": model.mc_samples,
                "seed": model.seed,
                "substreams":
                    "SeedSequence([seed, 0x77656172]).spawn(2)："
                    "制造散布 / 磨损速率；所有计算节点共用同一组样本"
                    "（同一批零件），首次越界按样本逐节点扫描",
            },
            "frozen_baseline":
                "基线链规范化输入、磨损曲线、计算节点与随机种子随研究冻结；"
                "后续基线变化不改写本研究结果",
        },
    }
    study_id = db.save_wear_study(
        chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), snapshot, result,
        payload.mc_samples, payload.random_seed)
    return _wear_study_response(db.get_wear_study(study_id))


@app.get("/chains/{chain_id}/wear-studies", tags=["wear"])
def list_wear_studies(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id, "studies": db.list_wear_studies(chain_id)}


def _load_wear_study(study_id: int):
    row = db.get_wear_study(study_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"磨损研究 {study_id} 不存在")
    return row


@app.get("/wear-studies/{study_id}", tags=["wear"])
def get_wear_study(study_id: int) -> dict:
    """读取冻结磨损研究：结果创建时固化，重复读取/计算内容不变。"""
    return _wear_study_response(_load_wear_study(study_id))


def _rebuild_wear_model(study_row):
    """从冻结研究快照重建磨损模型（基线链只读，规则原样重放）。"""
    chain_row = _load_chain(study_row.chain_id)
    nc = rebuild_normalized(chain_row.request_json)
    payload = WearStudyCreate.model_validate(study_row.request_json)
    return build_wear_model(nc, payload)


def _wear_maintenance_payload(row) -> dict:
    return {
        "search_id": row.id,
        "study_id": row.study_id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "frozen": bool(row.frozen),
        "selected_rank": row.selected_rank,
        "selected_note": row.selected_note,
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
    }


@app.post("/wear-studies/{study_id}/maintenance-searches", status_code=201,
          tags=["wear"])
def create_maintenance_search(study_id: int,
                              payload: MaintenanceSearchRequest) -> dict:
    """维护编排搜索：继续使用 / 加垫片 / 更换零件方案生成与排列。

    锁定件（locked_dimensions）不可更换；replacement_costs 列出可更换件
    的单次更换成本；每次维护动作计一次停机成本；维护节点必须是研究
    计算节点的子集。系统以三种贪心策略（最低成本 / 垫片优先 / 更换
    优先）生成方案并附零动作现状方案，按判据（极值界或 RSS ±3σ 界）
    统计全周期违规数，候选共用研究同源固定种子蒙特卡洛复核，按
    「全周期违规数 → 最早越界点（晚者优先）→ 总成本」升序排列。
    候选表与种子随结果冻结。
    """
    study_row = _load_wear_study(study_id)
    model = _rebuild_wear_model(study_row)
    try:
        result = search_maintenance(model, payload)
    except WearError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    seed = (model.seed if payload.random_seed is None
            else payload.random_seed)
    result = {
        **result,
        "study_id": study_id,
        "chain_id": study_row.chain_id,
        "frozen_inputs": {
            "baseline_chain_id": study_row.chain_id,
            "wear_study_id": study_id,
            "wear_study_seed": study_row.random_seed,
            "wear_model_snapshot": "磨损曲线 / 计算节点 / 封闭环限值引用"
                                   "冻结研究快照",
            "candidate_table": payload.model_dump(mode="json"),
            "random_seed": seed,
        },
    }
    search_id = db.save_wear_maintenance(
        study_id, study_row.chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result)
    return _wear_maintenance_payload(db.get_wear_maintenance(search_id))


@app.get("/wear-studies/{study_id}/maintenance-searches", tags=["wear"])
def list_maintenance_searches(study_id: int) -> dict:
    _load_wear_study(study_id)
    return {"study_id": study_id,
            "searches": db.list_wear_maintenances(study_id)}


@app.get("/wear-maintenance-searches/{search_id}", tags=["wear"])
def get_maintenance_search(search_id: int) -> dict:
    """读取维护编排搜索结果（选定前可重看候选；选定后结果不可改）。"""
    row = db.get_wear_maintenance(search_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"维护编排结果 {search_id} 不存在")
    return _wear_maintenance_payload(row)


@app.post("/wear-maintenance-searches/{search_id}/select", tags=["wear"])
def select_maintenance_plan(search_id: int,
                            payload: MaintenanceSelectRequest) -> dict:
    """选定一个维护方案并冻结：固化来源链、磨损曲线、维护动作与随机种子。

    每个编排结果只能选定一次（重复选定或改选返回 422）；历史方案不改写，
    后续基线链变化不影响已冻结结果。
    """
    row = db.get_wear_maintenance(search_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"维护编排结果 {search_id} 不存在")
    if row.frozen:
        raise HTTPException(
            status_code=422,
            detail=f"维护编排结果 {search_id} 已选定方案 rank="
                   f"{row.selected_rank} 并冻结，不能改选；请在冻结研究上"
                   "另建编排搜索")
    candidates = row.result_json.get("candidates", [])
    if payload.rank > len(candidates):
        raise HTTPException(
            status_code=422,
            detail=f"rank={payload.rank} 超出候选数 {len(candidates)}")
    db.freeze_wear_maintenance(search_id, payload.rank, payload.note)
    saved = db.get_wear_maintenance(search_id)
    answer = _wear_maintenance_payload(saved)
    chosen = next(c for c in saved.result_json["candidates"]
                  if c["rank"] == payload.rank)
    answer["selected_candidate"] = chosen
    answer["freeze_note"] = (
        "已冻结来源基线链、磨损曲线、维护动作与随机种子；后续基线变化"
        "不改写本方案。后续维护执行应引用本编排结果与研究快照。")
    return answer


# -------------------------------------------------------- 检验批次漂移研究

def _drift_study_payload(row, *, comparison: dict | None = None) -> dict:
    return {
        "study_id": row.id,
        "chain_id": row.chain_id,
        "name": row.name,
        "note": row.note,
        "version_no": row.version_no,
        "parent_study_id": row.parent_study_id,
        "frozen": bool(row.frozen),
        "finalized_note": row.finalized_note or None,
        "finalized_at": (row.finalized_at.isoformat()
                         if row.finalized_at else None),
        "exclusions": row.exclusions_json or [],
        "created_at": row.created_at.isoformat(),
        "submitted_input": row.request_json,
        "result": row.result_json,
        **({"comparison_with_parent": comparison} if comparison else {}),
    }


def _load_drift_batches(chain_id: int, batch_ids: list[int]):
    """按给定顺序取冻结批次，复核存在与同链（重复由 Pydantic 拦截）。"""
    rows = []
    problems = []
    for bid in batch_ids:
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


def _sampled_at_iso(payload: DriftStudyCreate) -> dict[int, str]:
    return {b.batch_id: b.sampled_at.isoformat() for b in payload.batches}


@app.post("/chains/{chain_id}/drift-studies", status_code=201,
          tags=["drift"])
def create_drift_study(chain_id: int, payload: DriftStudyCreate) -> dict:
    """建立检验批次漂移研究：同一基线链的多个冻结批次按采样时刻成序。

    每个来源批次指定采样时刻（必须严格递增、不得同时刻或逆序、不得混用
    时区口径）；为关注尺寸和/或封闭环设置单侧或双侧标准化 CUSUM
    （k、h）、EWMA（λ、L）、最小样本量与报警阈值（研究级缺省 + 逐目标
    覆盖）。批次不存在 / 错链 / 重复、采样顺序错误、目标链外或无任何
    监控目标时拒绝（422/404）。
    系统逐批返回批次均值、标准误（含量具批次级系统误差）、标准化 z、
    CUSUM/EWMA 统计量与阈值余量、漂移方向、首次报警批次与估计变点；
    来源批次引用测量方案时先做偏倚修正并传播量具不确定度，封闭环另给
    各尺寸对其漂移的带符号贡献。研究创建即固化来源批次、算法参数与
    计算结果（草案，可复制排除），定稿后不得再改写。
    """
    chain_row = _load_chain(chain_id)
    nc = rebuild_normalized(chain_row.request_json)

    # 链外尺寸在引擎外先做一次显式复核（给出清晰 422）
    chain_ids = {d.id for d in nc.dimensions}
    external = sorted({d.dimension_id for d in payload.dimensions
                       if d.dimension_id not in chain_ids})
    if external:
        raise HTTPException(
            status_code=422,
            detail=f"关注尺寸不在基线链 {chain_id} 上（未知尺寸）: "
                   f"{external}；链上尺寸: {sorted(chain_ids)}")

    batch_rows = _load_drift_batches(
        chain_id, [b.batch_id for b in payload.batches])
    try:
        result = run_drift_study(
            nc, chain_row.result_json, batch_rows, payload,
            _sampled_at_iso(payload))
    except DriftError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result["frozen_inputs"] = {
        "baseline_chain_id": chain_id,
        "source_batches": [
            {"batch_id": b.batch_id, "sampled_at": b.sampled_at.isoformat(),
             "note": b.note} for b in payload.batches],
        "algorithm": "standardized tabular CUSUM + EWMA（参数逐目标冻结）",
        "measurement_policy": "引用测量方案的批次按方案偏倚修正后计算；"
                              "量具批次级系统误差 u_rest 计入标准误，"
                              "封闭环按 GUM 相关传播",
    }
    study_id = db.save_drift_study(
        chain_id, payload.name, payload.note,
        payload.model_dump(mode="json"), result,
        exclusions=[], parent_study_id=None, version_no=1)
    saved = db.get_drift_study(study_id)
    return _drift_study_payload(saved)


@app.get("/chains/{chain_id}/drift-studies", tags=["drift"])
def list_drift_studies(chain_id: int) -> dict:
    _load_chain(chain_id)
    return {"chain_id": chain_id, "studies": db.list_drift_studies(chain_id)}


def _load_drift_study(study_id: int):
    row = db.get_drift_study(study_id)
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"漂移研究 {study_id} 不存在")
    return row


@app.get("/drift-studies/{study_id}", tags=["drift"])
def get_drift_study(study_id: int) -> dict:
    """读取漂移研究快照：来源批次、参数与结果创建时固化，重复读取不变。"""
    return _drift_study_payload(_load_drift_study(study_id))


@app.post("/drift-studies/{study_id}/copy", status_code=201, tags=["drift"])
def copy_drift_study(study_id: int,
                     payload: DriftStudyCopyRequest) -> dict:
    """复制研究：可逐批注明原因排除异常批次，比较排除前后结论。

    父研究必须尚未定稿（定稿后 422）；排除对象必须是父研究来源批次且
    原因非空，排除后至少保留 2 个批次（控制图需要时间序列）。副本与
    父研究同链、同算法参数，只重放剩余批次；父研究保持不变，副本结果
    中附 comparison_with_parent（报警方向、首次报警批次、估计变点、
    最新偏移逐项对比）。副本同样创建即冻结，可继续复制或定稿。
    """
    parent = _load_drift_study(study_id)
    if parent.frozen:
        raise HTTPException(
            status_code=422,
            detail=f"漂移研究 {study_id} 已定稿冻结（来源批次、算法参数与"
                   "计算结果不可改写），不能再复制排除版本")
    chain_row = _load_chain(parent.chain_id)
    nc = rebuild_normalized(chain_row.request_json)

    req = parent.request_json
    source_specs = req["batches"]
    source_ids = [b["batch_id"] for b in source_specs]
    excluded = {e.batch_id: e.reason for e in payload.exclusions}
    unknown = sorted(set(excluded) - set(source_ids))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"排除批次 {unknown} 不在父研究 {study_id} 的来源批次中；"
                   f"父研究来源批次: {source_ids}")
    remaining_specs = [b for b in source_specs
                       if b["batch_id"] not in excluded]
    if len(remaining_specs) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"排除后仅剩 {len(remaining_specs)} 个批次："
                   "CUSUM/EWMA 至少需要 2 个按采样时刻有序的批次")

    from .drift_schemas import DriftBatchSpec

    new_payload = DriftStudyCreate(
        name=payload.name or f"{parent.name}-复制",
        note=payload.note,
        batches=[
            DriftBatchSpec(batch_id=b["batch_id"],
                           sampled_at=b["sampled_at"],
                           note=b.get("note", ""))
            for b in remaining_specs],
        dimensions=req["dimensions"],
        monitor_closure=bool(req.get("monitor_closure", False)),
        closure_overrides=req.get("closure_overrides"),
        defaults=req["defaults"],
    )
    batch_rows = _load_drift_batches(
        parent.chain_id, [b["batch_id"] for b in remaining_specs])
    try:
        result = run_drift_study(
            nc, chain_row.result_json, batch_rows, new_payload,
            _sampled_at_iso(new_payload))
    except DriftError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    exclusions_store = [
        {"batch_id": bid, "reason": reason}
        for bid, reason in excluded.items()]
    result["copied_from"] = {
        "parent_study_id": parent.id,
        "parent_name": parent.name,
        "excluded_batches": exclusions_store,
    }
    result["frozen_inputs"] = {
        **parent.result_json.get("frozen_inputs", {}),
        "copied_from_study_id": parent.id,
    }
    new_no = parent.version_no + 1
    new_id = db.save_drift_study(
        parent.chain_id, new_payload.name, payload.note,
        new_payload.model_dump(mode="json"), result,
        exclusions=exclusions_store, parent_study_id=parent.id,
        version_no=new_no)
    comparison = compare_drift_results(parent.result_json, result)
    comparison["parent_study_id"] = parent.id
    comparison["copied_study_id"] = new_id
    saved = db.get_drift_study(new_id)
    return _drift_study_payload(saved, comparison=comparison)


@app.post("/drift-studies/{study_id}/finalize", tags=["drift"])
def finalize_drift_study(study_id: int,
                         payload: DriftStudyFinalizeRequest) -> dict:
    """定稿漂移研究：冻结来源批次、算法参数与计算结果。

    定稿幂等于同一状态：已定稿再次定稿返回 422；定稿后不可复制排除、
    不可改写，后续检验数据（新批次 / 测量方案新版本）不影响本研究。
    """
    row = _load_drift_study(study_id)
    if row.frozen:
        raise HTTPException(
            status_code=422,
            detail=f"漂移研究 {study_id} 已定稿冻结，不能重复定稿；"
                   "定稿时间 "
                   + (row.finalized_at.isoformat() if row.finalized_at else ""))
    db.freeze_drift_study(study_id, payload.note)
    saved = db.get_drift_study(study_id)
    answer = _drift_study_payload(saved)
    answer["freeze_note"] = (
        "已冻结来源批次、算法参数与计算结果；后续检验数据不得改写本研究，"
        "定稿研究也不能再派生排除版本（如需排除异常批次请在定稿前复制）")
    return answer
