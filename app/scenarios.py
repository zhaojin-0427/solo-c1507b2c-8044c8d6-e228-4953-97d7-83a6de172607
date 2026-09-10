"""方案分支 / 批量调整：从持久化基线重建规范化链并应用覆盖。"""
from __future__ import annotations

import math

import numpy as np

from .engine import (
    NormalizedChain,
    _override_chain,
    _sigma_arr,
    compute_all,
    normalize_chain,
)
from .schemas import ChainCreate
from .units import to_mm


def rebuild_normalized(request_json: dict) -> NormalizedChain:
    """从存储的请求 JSON 重建规范化链（重新跑一遍输入校验）。"""
    chain_create = ChainCreate.model_validate(request_json)
    return normalize_chain(chain_create)


def _id_index(nc: NormalizedChain) -> dict[str, int]:
    return {d.id: i for i, d in enumerate(nc.dimensions)}


def apply_scenario_overrides(nc: NormalizedChain, payload) -> tuple:
    """返回 (override_nc, sigmas, mids, halfs, overrides_record)。

    tolerance_overrides 以该尺寸的原始单位给出，内部换算为 mm；
    std_dev_overrides 同理（null = 非正态时回到理论标准差，
    正态显式 null 不允许，由 schema 流程保证正态基线必有 σ）。
    """
    idx = _id_index(nc)
    halfs = np.array([d.half_width for d in nc.dimensions])
    mids = np.array([d.mid for d in nc.dimensions])
    sigmas = _sigma_arr(nc.dimensions)
    record: dict[str, dict] = {}

    for dim_id, ov in payload.tolerance_overrides.items():
        if dim_id not in idx:
            raise KeyError(f"方案引用了不存在的尺寸: {dim_id}")
        i = idx[dim_id]
        unit = nc.dimensions[i].unit
        es = to_mm(ov.upper_deviation, unit)
        ei = to_mm(ov.lower_deviation, unit)
        mids[i] = (es + ei) / 2.0
        halfs[i] = (es - ei) / 2.0
        record.setdefault(dim_id, {})["tolerance"] = {
            "upper_deviation": ov.upper_deviation,
            "lower_deviation": ov.lower_deviation,
            "unit": unit,
            "normalized_mm": {
                "upper_deviation": es,
                "lower_deviation": ei,
                "mid_shift": mids[i],
                "half_width": halfs[i],
            },
        }

    for dim_id, std in payload.std_dev_overrides.items():
        if dim_id not in idx:
            raise KeyError(f"方案引用了不存在的尺寸: {dim_id}")
        i = idx[dim_id]
        unit = nc.dimensions[i].unit
        d = nc.dimensions[i]
        if std is None:
            if d.distribution == "normal":
                raise ValueError(
                    f"尺寸 {dim_id}: 正态分布不能将标准差置空"
                )
            sigmas[i] = _theoretical(d.distribution, halfs[i])
        else:
            sigmas[i] = to_mm(std, unit)
        record.setdefault(dim_id, {})["std_dev"] = {
            "std_dev": std,
            "unit": unit,
            "normalized_mm": {"sigma": sigmas[i]},
        }

    ov_nc = _override_chain(nc, sigmas, mids, halfs)
    return ov_nc, sigmas, mids, halfs, record


def _theoretical(distribution: str, half: float) -> float:
    if distribution == "uniform":
        return half / math.sqrt(3.0)
    if distribution == "triangular":
        return half / math.sqrt(6.0)
    raise ValueError(f"分布 {distribution} 无理论标准差")


def apply_batch_adjust(nc: NormalizedChain, payload) -> tuple:
    target = payload.target_dimensions
    ids = {d.id for d in nc.dimensions}
    if target is not None:
        missing = sorted(set(target) - ids)
        if missing:
            raise KeyError(f"批量调整引用了不存在的尺寸: {missing}")
        chosen = set(target)
    else:
        chosen = set(ids)

    halfs0 = np.array([d.half_width for d in nc.dimensions])
    mids = np.array([d.mid for d in nc.dimensions])
    halfs = halfs0.copy()
    sigmas = _sigma_arr(nc.dimensions)
    record = {"dimensions": [], "tolerance_scale": payload.tolerance_scale}

    for i, d in enumerate(nc.dimensions):
        if d.id not in chosen:
            continue
        halfs[i] = halfs0[i] * payload.tolerance_scale
        if payload.std_dev_scale is not None:
            sigmas[i] = sigmas[i] * payload.std_dev_scale
        elif d.distribution != "normal":
            sigmas[i] = _theoretical(d.distribution, halfs[i])
        # 正态且未指定 std_dev_scale：保留用户 σ（结果中明确注明）
        record["dimensions"].append({
            "dimension_id": d.id,
            "old_half_width_mm": float(halfs0[i]),
            "new_half_width_mm": float(halfs[i]),
            "new_sigma_mm": float(sigmas[i]),
            "distribution": d.distribution,
            "sigma_policy": (
                "scaled" if payload.std_dev_scale is not None
                else "theoretical" if d.distribution != "normal"
                else "kept_user_value"
            ),
        })

    ov_nc = _override_chain(nc, sigmas, mids, halfs)
    return ov_nc, sigmas, mids, halfs, record


def run_scenario(nc: NormalizedChain, sigmas, mids, halfs,
                 seed: int | None) -> dict:
    return compute_all(nc, sigmas=sigmas, mids=mids, halfs=halfs, seed=seed)


def compare_with_baseline(baseline_result: dict, scenario_result: dict) -> dict:
    """逐方法对比基线与方案分支的关键指标。"""
    out: dict = {}
    for method in ("worst_case", "rss", "monte_carlo"):
        b = baseline_result["results"][method]
        sc = scenario_result["results"][method]
        item = {
            "baseline": {
                "mean_gap_mm": b["mean_gap_mm"],
                "lower_bound_mm": b["lower_bound_mm"],
                "upper_bound_mm": b["upper_bound_mm"],
                "reject_probability": b["reject_probability"],
            },
            "scenario": {
                "mean_gap_mm": sc["mean_gap_mm"],
                "lower_bound_mm": sc["lower_bound_mm"],
                "upper_bound_mm": sc["upper_bound_mm"],
                "reject_probability": sc["reject_probability"],
            },
        }
        item["delta"] = {
            "mean_gap_mm": sc["mean_gap_mm"] - b["mean_gap_mm"],
            "lower_bound_mm": sc["lower_bound_mm"] - b["lower_bound_mm"],
            "upper_bound_mm": sc["upper_bound_mm"] - b["upper_bound_mm"],
        }
        if (b.get("reject_probability") is not None
                and sc.get("reject_probability") is not None):
            item["delta"]["reject_probability"] = (
                sc["reject_probability"] - b["reject_probability"]
            )
        if "sigma_mm" in b:
            item["baseline"]["sigma_mm"] = b["sigma_mm"]
            item["scenario"]["sigma_mm"] = sc["sigma_mm"]
            item["delta"]["sigma_mm"] = sc["sigma_mm"] - b["sigma_mm"]
        out[method] = item
    return out
