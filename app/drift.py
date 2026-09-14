"""检验批次漂移研究引擎（CUSUM / EWMA；内部统一 mm）。

监控目标
========
研究把同一基线链的多个冻结来料检验批次按采样时刻排成时间序列，
对每个**关注尺寸**与可选的**封闭环**逐批计算标准化均值 z_t：

    z_t = (x̄_t − μ0) / SE_t

* 目标均值（in-control 水平）μ0：
  - 尺寸：制程中心 C_i = N_i + (ES_i+EI_i)/2（与来料检验的 Δ_i、
    RSS 制程中心假设同口径）；
  - 封闭环：基线 RSS 均值 μ_C = Σ s_i·C_i。
* 目标标准差 σ0：尺寸取基线 RSS 逐尺寸 σ_i；封闭环取基线 RSS σ_C。
* 批次均值的标准误 SE_t：
      SE_t = sqrt(σ0²/n_t + u_g,t²)
  n_t 为该批次对目标的有效样本数；u_g,t 为该批次量具方案带来的
  **批次级固定系统误差**标准不确定度（见下「量具不确定度」）。
  σ0²/n 已含单工件重复性与制造散布，量具重复性（u_rep，逐工件随机）
  不重复计入；u_g 是同一批次共用同一台量具导致的整批公共偏移，
  不随 n 缩小（保守口径）。

标准化制表 CUSUM（目标参数 k、h，均以 σ0 为单位）
    C+_t = max(0, C+_{t-1} + z_t − k)，C−_t = max(0, C−_{t-1} − z_t − k)
    C+ 越过 h：向上漂移报警；C− 越过 h：向下漂移报警。
    单侧（one_sided_up / one_sided_down）只维护并判定对应一侧。

EWMA（目标参数 λ、L）
    q_t = λ·z_t + (1−λ)·q_{t-1}，q_0 = 0
    时变控制限 ±L·sqrt(λ/(2−λ)·(1−(1−λ)^{2t}))（Lucas–Kroeger 起步限）。

阈值余量 = 阈值 − 当前统计量（距报警还差多少 σ0；≤0 表示已报警）。
变点估计：CUSUM 用本轮连续累计起点法（报警轮内首次 C+/C− 回到 0
之后的第一批；未报警时给出累积分数最后一个谷底作为探索性变点）；
EWMA 用累积标准化残差谷底法（S_t=Σ(z−q_hat)）；偏移量取变点之后
批次均值的标准化平均。所有规则确定、无随机性，结果随研究快照冻结。

量具不确定度传播（批次引用测量方案时）
======================================
偏倚修正：逐工件 x_c = x + b_i 后再算批次均值（与检验批次判定同口径）。
尺寸级批次固定系统误差：
    u_g,i = sqrt(u_cal,i² + u_bias,i² + u_rep,i²) = u_rest,i
（分辨率量化误差逐工件独立，按 √n 缩小，已由 σ0/√n 口径保守吸收；
 u_rest 为校准/偏倚/重复性等共用量具公共误差源）。
封闭环级按 GUM 线性传播（s_i 为闭环方向系数，ρ 为量具误差相关）：
    u_g,C² = Σ_iΣ_j s_i s_j ρ_ij u_rest,i u_rest,j
扩展不确定度 U_g = k·u_g（尺寸用方案覆盖因子 k_i；封闭环用批次
output_coverage_factor k_out）。量具贡献随每批结果与研究快照冻结。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .units import to_mm


class DriftError(ValueError):
    """漂移研究输入错误（API 层映射为 422）。"""


# ------------------------------------------------------------- 参数合并

_PARAM_FIELDS = ("sidedness", "cusum_k", "cusum_h",
                 "ewma_lambda", "ewma_L", "min_sample_size")


def resolve_params(defaults, overrides) -> dict[str, Any]:
    """用逐目标覆盖合并研究级缺省参数。"""
    resolved: dict[str, Any] = {
        "sidedness": defaults.sidedness.value
        if hasattr(defaults.sidedness, "value") else str(defaults.sidedness),
        "cusum_k": float(defaults.cusum_k),
        "cusum_h": float(defaults.cusum_h),
        "ewma_lambda": float(defaults.ewma_lambda),
        "ewma_L": float(defaults.ewma_L),
        "min_sample_size": int(defaults.min_sample_size),
    }
    if overrides is not None:
        for f in _PARAM_FIELDS:
            v = getattr(overrides, f, None)
            if v is None:
                continue
            resolved[f] = v.value if hasattr(v, "value") else v
    return resolved


# ------------------------------------------------------------- 批次取数

@dataclass
class _BatchData:
    """单个来源批次抽取后的 mm 数据（偏倚修正后）。"""

    batch_id: int
    batch_name: str
    sampled_at: str                       # ISO 字符串（原样保留）
    samples: dict[str, list[float]]       # dim_id -> 偏倚修正后样本（mm）
    bias_mm: dict[str, float]
    closure_complete: list[float]         # 所有尺寸齐全工件的封闭环值
    gage: dict[str, Any]                 # 量具不确定度（无方案为占位零值）
    measurement_plan_id: int | None


def _gage_block(nc, brow) -> tuple[dict[str, Any], dict[str, float]]:
    """从冻结批次的测量方案快照提取逐尺寸 u_rest / 偏倚 / 相关与封闭环传播。"""
    ids = [d.id for d in nc.dimensions]
    signs = np.array([d.sign for d in nc.dimensions], dtype=float)
    bias = {i: 0.0 for i in ids}
    u_rest = {i: 0.0 for i in ids}
    u_res = {i: 0.0 for i in ids}
    k_dim = {i: None for i in ids}
    plan_id = brow.measurement_plan_id
    combined = None
    k_out = 2.0
    if brow.measurement_json:
        combined = brow.measurement_json.get(
            "plan_snapshot", {}).get("combined")
        k_out = float(
            brow.measurement_json.get("policy", {})
            .get("output_coverage_factor", 2.0))
    if combined is not None:
        by_id = {p["dimension_id"]: p for p in combined["dimensions"]}
        for i in ids:
            p = by_id.get(i)
            if p is None:
                continue
            bias[i] = float(p["bias_correction_mm"])
            u_rest[i] = float(p["u_rest_mm"])
            u_res[i] = float(p["components_mm"]["u_resolution"])
            k_i = p.get("coverage_factor")
            if k_i is None:
                k_i = p.get("submitted", {}).get("coverage_factor")
            k_dim[i] = float(k_i) if k_i is not None else k_out
        corr = np.array(combined["correlation"]["matrix"], dtype=float)
    else:
        corr = np.eye(len(ids))

    # 封闭环 GUM 传播（只含共用量具 rest 部分；与 measurement.evaluate_batch 同式）
    wr = np.array([signs[j] * u_rest[ids[j]] for j in range(len(ids))])
    var_closure_rest = float((corr * wr[:, None] * wr[None, :]).sum())
    u_g_closure = math.sqrt(max(var_closure_rest, 0.0))
    gage = {
        "has_plan": combined is not None,
        "measurement_plan_id": plan_id,
        "u_rest_mm": {i: float(u_rest[i]) for i in ids},
        "u_resolution_mm": {i: float(u_res[i]) for i in ids},
        "bias_correction_mm": {i: float(bias[i]) for i in ids},
        "coverage_factor_dimension": {
            i: (float(k_dim[i]) if k_dim[i] is not None else None)
            for i in ids},
        "closure": {
            "u_g_mm": u_g_closure,
            "variance_mm2": var_closure_rest,
            "output_coverage_factor": k_out,
            "expanded_uncertainty_mm": k_out * u_g_closure,
        },
    }
    return gage, bias


def _extract_batches(nc, batch_rows, sampled_at: dict[int, str]) -> list[_BatchData]:
    """把冻结批次行（已确认同链、按采样时刻有序）抽取为内部数据。"""
    out: list[_BatchData] = []
    for brow in batch_rows:
        gage, bias = _gage_block(nc, brow)
        samples: dict[str, list[float]] = {d.id: [] for d in nc.dimensions}
        complete: list[float] = []
        signs = {d.id: d.sign for d in nc.dimensions}
        ids = [d.id for d in nc.dimensions]
        for row in brow.rows_json:
            by_dim = {m["dimension_id"]: m for m in row["measurements"]}
            vals_mm: dict[str, float] = {}
            for i in ids:
                m = by_dim.get(i)
                if m is None or m["value"] is None:
                    continue
                vals_mm[i] = to_mm(m["value"], m["unit"]) + bias[i]
                samples[i].append(vals_mm[i])
            if len(vals_mm) == len(ids):
                complete.append(
                    float(sum(signs[i] * vals_mm[i] for i in ids)))
        out.append(_BatchData(
            batch_id=brow.id, batch_name=brow.name,
            sampled_at=sampled_at[brow.id], samples=samples, bias_mm=bias,
            closure_complete=complete, gage=gage,
            measurement_plan_id=brow.measurement_plan_id,
        ))
    return out


# ------------------------------------------------------------- CUSUM / EWMA

def _track_sides(sidedness: str) -> tuple[bool, bool]:
    return (sidedness in ("two_sided", "one_sided_up"),
            sidedness in ("two_sided", "one_sided_down"))


def _cusum(z: np.ndarray, sidedness: str, k: float, h: float) -> dict[str, Any]:
    """标准化制表 CUSUM：逐批统计量、报警、阈值余量、变点。"""
    use_up, use_down = _track_sides(sidedness)
    cp, cn = 0.0, 0.0
    plus: list[float] = []
    minus: list[float] = []
    first_alarm_idx: int | None = None
    alarm_direction: str | None = None
    for t, zv in enumerate(z):
        if use_up:
            cp = max(0.0, cp + float(zv) - k)
        if use_down:
            cn = max(0.0, cn - float(zv) - k)
        plus.append(cp if use_up else None)
        minus.append(cn if use_down else None)
        if first_alarm_idx is None and (
                (use_up and cp > h) or (use_down and cn > h)):
            first_alarm_idx = t
            if use_up and cp > h and (not use_down or cp >= cn):
                alarm_direction = "up"
            else:
                alarm_direction = "down"

    def margin(stat) -> float | None:
        return None if stat is None else float(h - stat)

    # 变点：报警轮内累计首次归零后的第一批；未报警取累计分数最后谷底
    run_plus = np.array([0.0] + [v if v is not None else 0.0 for v in plus])
    run_minus = np.array([0.0] + [v if v is not None else 0.0 for v in minus])

    def _cp_for(seq: np.ndarray) -> tuple[int, float] | None:
        """seq 长度 T+1（seq[0]=0）。返回 (变点在 z 中的下标, 变点后标准化均值)。"""
        nz = np.nonzero(seq[1:] > 0.0)[0]
        if nz.size == 0:
            return None
        start = int(nz[0])              # 该段累计从 z[start] 起非零
        idx_after = np.arange(start, len(z))
        if idx_after.size == 0:
            return None
        return start, float(np.mean(z[idx_after]))

    cp_plus = _cp_for(run_plus) if use_up else None
    cp_minus = _cp_for(run_minus) if use_down else None

    def _peak(seq: np.ndarray) -> float:
        return float(np.max(seq))

    if first_alarm_idx is not None:
        # 以报警方向的累计段定位变点
        chosen = None
        if alarm_direction == "up" and cp_plus is not None:
            chosen = cp_plus
        elif alarm_direction == "down" and cp_minus is not None:
            chosen = cp_minus
        if chosen is None:
            chosen = (cp_plus if cp_plus is not None
                      else cp_minus if cp_minus is not None else None)
        cp_idx, cp_shift = chosen if chosen is not None else (None, None)
    elif cp_plus is not None or cp_minus is not None:
        # 未报警：探索性变点 = 两侧累计峰值更大一侧的段起点
        options = []
        if cp_plus is not None:
            options.append((_peak(run_plus), cp_plus))
        if cp_minus is not None:
            options.append((_peak(run_minus), cp_minus))
        cp_idx, cp_shift = max(options, key=lambda o: o[0])[1]
    else:
        cp_idx, cp_shift = None, None

    return {
        "chart": "tabular_cusum",
        "params": {"k": float(k), "h": float(h), "sidedness": sidedness},
        "stat_plus": plus,
        "stat_minus": minus,
        "threshold_margin_up": [margin(v) for v in plus],
        "threshold_margin_down": [margin(v) for v in minus],
        "alarmed": first_alarm_idx is not None,
        "first_alarm_batch_index": (
            first_alarm_idx + 1 if first_alarm_idx is not None else None),
        "first_alarm_direction": alarm_direction,
        "change_point": _change_point(z, cp_idx, cp_shift),
    }


def _ewma(z: np.ndarray, sidedness: str, lam: float,
          L: float) -> dict[str, Any]:
    """标准化 EWMA：q_t 与 Lucas–Kroeger 时变限、报警、余量、变点。"""
    use_up, use_down = _track_sides(sidedness)
    factor = lam / (2.0 - lam)
    q = 0.0
    qs: list[float] = []
    limits: list[float] = []
    first_alarm_idx: int | None = None
    alarm_direction: str | None = None
    for t, zv in enumerate(z, start=1):
        q = lam * float(zv) + (1.0 - lam) * q
        limit = L * math.sqrt(factor * (1.0 - (1.0 - lam) ** (2 * t)))
        qs.append(q)
        limits.append(limit)
        if first_alarm_idx is None and (
                (use_up and q > limit) or (use_down and q < -limit)):
            first_alarm_idx = t - 1
            if use_up and q > limit and (not use_down or q >= -q):
                alarm_direction = "up"
            else:
                alarm_direction = "down"

    # 离线变点：稳态均值取全序列均值，S_t=Σ(z−z̄)；
    # 向下变点取谷底，向上变点取峰前，双侧取更显著的一侧。
    zmean = float(np.mean(z)) if len(z) else 0.0
    s = np.cumsum(z - zmean)
    if use_up and use_down:
        trough_idx = int(np.argmin(s))
        peak_idx = int(np.argmax(s))
        cp_idx = peak_idx if abs(s[peak_idx]) > abs(s[trough_idx]) else trough_idx
    elif use_up:
        cp_idx = int(np.argmax(s))
    else:
        cp_idx = int(np.argmin(s))
    idx_after = np.arange(cp_idx, len(z))
    cp_shift = (float(np.mean(z[idx_after])) if len(idx_after) else None)

    def margin_up(qq, ll):
        return float(ll - qq) if use_up else None

    def margin_down(qq, ll):
        return float(ll + qq) if use_down else None

    return {
        "chart": "ewma",
        "params": {"lambda": float(lam), "L": float(L),
                   "sidedness": sidedness},
        "statistic": [float(v) for v in qs],
        "control_limit": [float(v) for v in limits],
        "threshold_margin_up": [margin_up(qs[i], limits[i])
                                for i in range(len(qs))],
        "threshold_margin_down": [margin_down(qs[i], limits[i])
                                  for i in range(len(qs))],
        "alarmed": first_alarm_idx is not None,
        "first_alarm_batch_index": (
            first_alarm_idx + 1 if first_alarm_idx is not None else None),
        "first_alarm_direction": alarm_direction,
        "change_point": _change_point(z, cp_idx, cp_shift),
    }


def _change_point(z: np.ndarray, cp_idx: int | None,
                  cp_shift: float | None) -> dict[str, Any] | None:
    if cp_idx is None:
        return None
    after = z[cp_idx:]
    return {
        "batch_index": cp_idx + 1,                   # 1 基，对应 batches 序号
        "between_batch_index": cp_idx,               # 变点位于第 cp_idx 与 cp_idx+1 批之间
        "estimated_shift_sigma": (
            float(cp_shift) if cp_shift is not None else None),
        "note": "CUSUM：本轮连续累计起点；EWMA：累积标准化残差谷底。"
                "偏移量为变点之后批次 z 值的平均（σ0 单位），探索性估计",
        "after_batch_count": int(len(after)),
    }


# ------------------------------------------------------------- 目标计算

@dataclass
class _TargetSpec:
    key: str                                # "dim:L1" / "closure"
    label: str
    kind: str                               # dimension / closure
    dimension_id: str | None
    params: dict[str, Any] = field(default_factory=dict)


def _evaluate_target(spec: _TargetSpec, batches: list[_BatchData],
                     target_cfg: dict[str, Any]) -> dict[str, Any]:
    """逐目标：批次均值/标准误/z → CUSUM / EWMA → 汇总。"""
    min_n = int(spec.params["min_sample_size"])
    mu0 = float(target_cfg["mu0_mm"])
    sigma0 = float(target_cfg["sigma0_mm"])
    lsl = target_cfg.get("lsl_mm")
    usl = target_cfg.get("usl_mm")

    entries: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    z_values: list[float] = []
    for b in batches:
        if spec.kind == "dimension":
            vals = b.samples[spec.dimension_id]
            n = len(vals)
            mean = float(np.mean(vals)) if n else None
            sample_std = (float(np.std(vals, ddof=1))
                          if n >= 2 else None)
            u_g = b.gage["u_rest_mm"][spec.dimension_id]
            k_i = b.gage["coverage_factor_dimension"][spec.dimension_id]
            gage_block = {
                "measurement_plan_id": b.measurement_plan_id,
                "u_g_mm": float(u_g),
                "coverage_factor": k_i,
                "expanded_uncertainty_mm": (
                    float(k_i * u_g) if k_i is not None else None),
            }
        else:
            vals = b.closure_complete
            n = len(vals)
            mean = float(np.mean(vals)) if n else None
            sample_std = (float(np.std(vals, ddof=1))
                          if n >= 2 else None)
            cg = b.gage["closure"]
            gage_block = {
                "measurement_plan_id": b.measurement_plan_id,
                "u_g_mm": cg["u_g_mm"],
                "coverage_factor": cg["output_coverage_factor"],
                "expanded_uncertainty_mm": cg["expanded_uncertainty_mm"],
            }

        if n < min_n:
            insufficient.append({
                "batch_id": b.batch_id, "batch_name": b.batch_name,
                "sampled_at": b.sampled_at, "sample_count": n,
                "min_sample_size": min_n,
                "reason": f"有效样本 {n} < 最小样本量 {min_n}，该批不参与控制图",
            })
            continue
        se_sampling_sq = sigma0 ** 2 / n
        u_g = float(gage_block["u_g_mm"])
        se = math.sqrt(se_sampling_sq + u_g ** 2)
        shift = mean - mu0
        if sigma0 <= 0.0:
            z = None
            z_note = "目标基线标准差 σ0=0（固定尺寸/固定间隙）：" \
                     "无法标准化，不进行控制图作图"
        elif se <= 0.0:
            z = 0.0 if abs(shift) <= 1e-12 else None
            z_note = None if z is not None else \
                "标准误为 0 且批次均值偏离 μ0：z 无定义，不参与作图"
        else:
            z = float(shift / se)
            z_note = None
        if z is None:
            insufficient.append({
                "batch_id": b.batch_id, "batch_name": b.batch_name,
                "sampled_at": b.sampled_at, "sample_count": n,
                "min_sample_size": min_n,
                "reason": z_note,
            })
            continue
        z_values.append(z)
        entries.append({
            "sequence_no": len(entries) + 1,
            "batch_id": b.batch_id,
            "batch_name": b.batch_name,
            "sampled_at": b.sampled_at,
            "sample_count": n,
            "mean_mm": mean,
            "sample_std_mm": sample_std,
            "shift_from_in_control_mm": float(shift),
            "se_sampling_mm": float(math.sqrt(se_sampling_sq)),
            "u_gage_batch_mm": u_g,
            "standard_error_mm": float(se),
            "z": float(z),
            "gage_uncertainty": gage_block,
        })

    charts_unavailable = None
    cusum = ewma = None
    if len(entries) < 2:
        charts_unavailable = (
            f"满足最小样本量的批次仅 {len(entries)} 个（至少需要 2 个）："
            "CUSUM / EWMA 需要时间序列，本目标只给逐批统计，不作图、不判报警")
    else:
        z_arr = np.array(z_values, dtype=float)
        cusum = _cusum(z_arr, spec.params["sidedness"],
                       spec.params["cusum_k"], spec.params["cusum_h"])
        ewma = _ewma(z_arr, spec.params["sidedness"],
                     spec.params["ewma_lambda"], spec.params["ewma_L"])

    def _conclusion(chart: dict[str, Any] | None) -> dict[str, Any]:
        if chart is None:
            return {"alarmed": False, "first_alarm_batch": None,
                    "first_alarm_direction": None,
                    "change_point": None,
                    "latest_drift_direction": "undetermined",
                    "latest_shift_sigma": (
                        entries[-1]["z"] if entries else None)}
        ci = chart["first_alarm_batch_index"]
        alarm_batch = None
        if ci is not None:
            e = entries[ci - 1]
            alarm_batch = {
                "batch_id": e["batch_id"], "batch_name": e["batch_name"],
                "sampled_at": e["sampled_at"], "sequence_no": e["sequence_no"]}
        latest_z = entries[-1]["z"] if entries else 0.0
        return {
            "alarmed": chart["alarmed"],
            "first_alarm_batch": alarm_batch,
            "first_alarm_direction": chart["first_alarm_direction"],
            "change_point": chart["change_point"],
            "latest_drift_direction": (
                "up" if latest_z > 0 else "down" if latest_z < 0
                else "none"),
            "latest_shift_sigma": float(latest_z),
        }

    result: dict[str, Any] = {
        "target": spec.key,
        "label": spec.label,
        "kind": spec.kind,
        "dimension_id": spec.dimension_id,
        "in_control": {
            "mean_mm": mu0,
            "sigma0_mm": sigma0,
            "spec_mm": {"lsl": lsl, "usl": usl},
            "basis": (
                "尺寸：μ0=N+(ES+EI)/2 制程中心，σ0 为基线 RSS 逐尺寸 σ；"
                "封闭环：μ0、σ0 取基线 RSS 封闭环均值与标准差"),
        },
        "params": spec.params,
        "standard_error_formula": "SE_t = sqrt(σ0²/n_t + u_g,t²)；"
                                  "z_t = (x̄_t − μ0)/SE_t",
        "batches_total": len(batches),
        "batches_used": len(entries),
        "insufficient_batches": insufficient,
        "batch_results": entries,
        "cusum": cusum,
        "ewma": ewma,
        "charts_unavailable_reason": charts_unavailable,
        "conclusion_cusum": _conclusion(cusum),
        "conclusion_ewma": _conclusion(ewma),
    }
    result["conclusion"] = _overall_conclusion(result)
    return result


def _overall_conclusion(target_result: dict[str, Any]) -> dict[str, Any]:
    """汇总两种控制图：任一报警即报警，取最先报警批次。"""
    conclusions = [("cusum", target_result["conclusion_cusum"]),
                   ("ewma", target_result["conclusion_ewma"])]
    alarming = [(name, c) for name, c in conclusions if c["alarmed"]]
    if not alarming:
        return {
            "alarmed": False,
            "first_alarm_batch": None,
            "first_alarm_chart": None,
            "first_alarm_direction": None,
            "change_points": {
                name: c["change_point"] for name, c in conclusions},
            "drift_direction": conclusions[0][1]["latest_drift_direction"],
            "latest_shift_sigma": conclusions[0][1]["latest_shift_sigma"],
        }

    def seq_of(item):
        c = item[1]
        return c["first_alarm_batch"]["sequence_no"]

    name, first = min(alarming, key=seq_of)
    return {
        "alarmed": True,
        "first_alarm_batch": first["first_alarm_batch"],
        "first_alarm_chart": name,
        "first_alarm_direction": first["first_alarm_direction"],
        "change_points": {n: c["change_point"] for n, c in conclusions},
        "drift_direction": first["first_alarm_direction"],
        "latest_shift_sigma": conclusions[0][1]["latest_shift_sigma"],
    }


# ------------------------------------------------------------- 研究总装

def run_study(nc, baseline_result: dict[str, Any], batch_rows, payload,
              sampled_at: dict[int, str]) -> dict[str, Any]:
    """对同一基线链的多个冻结批次执行漂移研究。

    batch_rows 必须已由 API 层确认存在、同链、不重复，且顺序与
    payload.batches（采样严格递增）一致。
    """
    batches = _extract_batches(nc, batch_rows, sampled_at)

    # 基线目标参数
    centers = {d.id: d.nominal + d.mid for d in nc.dimensions}
    sigmas_dim = {d.id: d.sigma for d in nc.dimensions}
    signs = {d.id: d.sign for d in nc.dimensions}
    rss = baseline_result["results"]["rss"]
    closure_mu0 = float(rss["mean_gap_mm"])
    closure_sigma0 = float(rss["sigma_mm"])

    target_specs: list[_TargetSpec] = []
    target_cfgs: dict[str, dict[str, Any]] = {}
    chain_ids = {d.id for d in nc.dimensions}
    for dt in payload.dimensions:
        if dt.dimension_id not in chain_ids:
            raise DriftError(
                f"关注尺寸 {dt.dimension_id!r} 不在基线链上（未知尺寸）；"
                f"链上尺寸: {sorted(chain_ids)}")
        key = f"dim:{dt.dimension_id}"
        d = next(x for x in nc.dimensions if x.id == dt.dimension_id)
        target_specs.append(_TargetSpec(
            key=key, label=f"尺寸 {dt.dimension_id}", kind="dimension",
            dimension_id=dt.dimension_id,
            params=resolve_params(payload.defaults, dt.overrides)))
        target_cfgs[key] = {
            "mu0_mm": centers[dt.dimension_id],
            "sigma0_mm": sigmas_dim[dt.dimension_id],
            "lsl_mm": d.nominal + d.lower_dev,
            "usl_mm": d.nominal + d.upper_dev,
        }
    if payload.monitor_closure:
        key = "closure"
        target_specs.append(_TargetSpec(
            key=key, label="封闭环", kind="closure", dimension_id=None,
            params=resolve_params(
                payload.defaults, payload.closure_overrides)))
        target_cfgs[key] = {
            "mu0_mm": closure_mu0,
            "sigma0_mm": closure_sigma0,
            "lsl_mm": nc.closure_lsl_mm,
            "usl_mm": nc.closure_usl_mm,
        }

    target_results = [
        _evaluate_target(spec, batches, target_cfgs[spec.key])
        for spec in target_specs
    ]

    # ---- 封闭环：各尺寸对封闭环漂移的贡献（末批相对首批）
    closure_contributions = None
    if payload.monitor_closure and batches:
        first_means = {}
        last_means = {}
        for d in nc.dimensions:
            seq = [float(np.mean(b.samples[d.id])) for b in batches
                   if b.samples[d.id]]
            if seq:
                first_means[d.id] = seq[0]
                last_means[d.id] = seq[-1]
        contribs = []
        total_signed = 0.0
        for d in nc.dimensions:
            if d.id in first_means:
                delta = last_means[d.id] - first_means[d.id]
                signed = signs[d.id] * delta
                total_signed += signed
            else:
                delta = signed = 0.0
            contribs.append({
                "dimension_id": d.id,
                "sign": signs[d.id],
                "first_batch_mean_mm": first_means.get(d.id),
                "last_batch_mean_mm": last_means.get(d.id),
                "mean_shift_mm": float(delta),
                "closure_drift_contribution_mm": float(signed),
            })
        for c in contribs:
            denom = abs(total_signed)
            c["share_of_observed_closure_drift"] = (
                c["closure_drift_contribution_mm"] / denom
                if denom > 0 else None)
        closure_first = [float(np.mean(b.closure_complete))
                         for b in batches if b.closure_complete]
        closure_contributions = {
            "baseline_batch_id": batches[0].batch_id,
            "latest_batch_id": batches[-1].batch_id,
            "observed_closure_drift_mm": (
                closure_first[-1] - closure_first[0]
                if len(closure_first) >= 2 else None),
            "signed_contributions_total_mm": float(total_signed),
            "dimensions": contribs,
            "note": "贡献 = s_i·(x̄_i,末批 − x̄_i,首批)（偏倚修正后批次均值，"
                    "mm）；占比按各尺寸带符号贡献绝对值之和归一",
        }

    warnings = []
    for tr in target_results:
        if tr["charts_unavailable_reason"]:
            warnings.append(f"{tr['label']}: {tr['charts_unavailable_reason']}")
        for ib in tr["insufficient_batches"]:
            if "最小样本量" in ib["reason"]:
                warnings.append(
                    f"{tr['label']}: 批次 {ib['batch_id']}（{ib['batch_name']}）"
                    f" {ib['reason']}")
        cfg = target_cfgs[tr["target"]]
        if cfg["sigma0_mm"] <= 0.0:
            warnings.append(
                f"{tr['label']}: 基线 σ0=0（固定尺寸/间隙），控制图不可标准化")
    if any(b.measurement_plan_id is not None for b in batches) and not all(
            b.measurement_plan_id is not None for b in batches):
        warnings.append(
            "部分来源批次引用了测量方案、部分未引用：未引用批次按零偏倚、"
            "零量具系统误差处理，跨批比较时请注意量具口径不一致")

    return {
        "status": "completed",
        "sequence_policy": "批次按采样时刻严格递增排序，控制图顺序即采样顺序",
        "source_batches": [
            {
                "sequence_no": i + 1,
                "batch_id": b.batch_id,
                "batch_name": b.batch_name,
                "sampled_at": b.sampled_at,
                "measurement_plan_id": b.measurement_plan_id,
                "sample_counts": {
                    d.id: len(b.samples[d.id]) for d in nc.dimensions},
                "closure_complete_count": len(b.closure_complete),
                "gage_uncertainty": {
                    "has_plan": b.gage["has_plan"],
                    "closure_u_g_mm": b.gage["closure"]["u_g_mm"],
                    "closure_expanded_uncertainty_mm":
                        b.gage["closure"]["expanded_uncertainty_mm"],
                    "dimension_u_g_mm": b.gage["u_rest_mm"],
                    "dimension_expanded_uncertainty_mm": {
                        d.id: (
                            b.gage["coverage_factor_dimension"][d.id]
                            * b.gage["u_rest_mm"][d.id]
                            if b.gage["coverage_factor_dimension"][d.id]
                            is not None else None)
                        for d in nc.dimensions},
                },
            }
            for i, b in enumerate(batches)
        ],
        "targets": target_results,
        "closure_drift_contributions": closure_contributions,
        "warnings": warnings,
        "formulas": FORMULAS,
        "conclusion": _study_conclusion(target_results),
    }


def _study_conclusion(target_results: list[dict[str, Any]]) -> dict[str, Any]:
    """研究级总结：所有目标的首次报警批次中取最早者。"""
    alarming = []
    for tr in target_results:
        c = tr["conclusion"]
        if c["alarmed"]:
            alarming.append((tr, c))
    if not alarming:
        return {
            "any_alarm": False,
            "first_alarm": None,
            "alarmed_targets": [],
            "target_directions": {
                tr["target"]: tr["conclusion"]["drift_direction"]
                for tr in target_results},
        }
    tr, c = min(alarming,
                key=lambda x: x[1]["first_alarm_batch"]["sequence_no"])
    return {
        "any_alarm": True,
        "first_alarm": {
            "target": tr["target"],
            "label": tr["label"],
            "kind": tr["kind"],
            "chart": c["first_alarm_chart"],
            "direction": c["first_alarm_direction"],
            "batch": c["first_alarm_batch"],
        },
        "alarmed_targets": [
            {"target": t["target"], "label": t["label"],
             "chart": cc["first_alarm_chart"],
             "direction": cc["first_alarm_direction"],
             "batch": cc["first_alarm_batch"]}
            for t, cc in ((t, t["conclusion"]) for t in target_results)
            if cc["alarmed"]],
        "target_directions": {
            t["target"]: t["conclusion"]["drift_direction"]
            for t in target_results},
    }


FORMULAS = [
    "目标均值 μ0：尺寸取制程中心 N+(ES+EI)/2；封闭环取基线 RSS 均值",
    "目标标准差 σ0：尺寸取基线 RSS 逐尺寸 σ；封闭环取基线 RSS σ_C",
    "批次均值标准误 SE_t = sqrt(σ0²/n_t + u_g,t²)；z_t = (x̄_t − μ0)/SE_t",
    "u_g,t 为批次量具方案的批次级固定系统误差（校准/偏倚/重复性等"
    "共用量具公共误差源 u_rest），不随 n 缩小；逐工件随机的分辨率"
    "量化误差按 √n 缩小，已由 σ0/√n 口径保守吸收",
    "CUSUM：C+_t=max(0,C+_{t-1}+z_t−k)，C−_t=max(0,C−_{t-1}−z_t−k)，"
    "越过 h 报警；单侧只维护对应一侧",
    "EWMA：q_t=λ·z_t+(1−λ)·q_{t-1}，q_0=0；时变限 "
    "±L·sqrt(λ/(2−λ)·(1−(1−λ)^{2t}))",
    "阈值余量：CUSUM 为 h−C±；EWMA 为 限−|q|（≤0 即已报警）",
    "变点：CUSUM 取报警轮连续累计起点；EWMA 取累积标准化残差谷底；"
    "偏移量为变点后批次 z 值平均（σ0 单位）",
    "封闭环量具传播：u_g,C²=Σ_iΣ_j s_i s_j ρ_ij u_rest,i u_rest,j；"
    "扩展不确定度尺寸用 k_i、封闭环用 k_out",
    "各尺寸封闭环漂移贡献：s_i·(x̄_i,末批 − x̄_i,首批)（偏倚修正后均值）",
]


# ------------------------------------------------------------- 版本比较

def compare_results(base_result: dict[str, Any],
                    new_result: dict[str, Any]) -> dict[str, Any]:
    """比较两份研究结果（通常为排除异常批次前后）的逐目标结论。"""
    base_by = {t["target"]: t for t in base_result["targets"]}
    new_by = {t["target"]: t for t in new_result["targets"]}
    targets = []
    for key in base_by:
        b, n = base_by[key], new_by.get(key)
        if n is None:
            continue
        targets.append({
            "target": key,
            "label": b["label"],
            "kind": b["kind"],
            "before": _conclusion_summary(b),
            "after": _conclusion_summary(n),
            "changed": _conclusion_summary(b) != _conclusion_summary(n),
            "alarm_disappeared": (
                b["conclusion"]["alarmed"]
                and not n["conclusion"]["alarmed"]),
            "alarm_newly_appeared": (
                not b["conclusion"]["alarmed"]
                and n["conclusion"]["alarmed"]),
            "first_alarm_batch_changed": (
                (b["conclusion"].get("first_alarm_batch") or {}).get("batch_id")
                != (n["conclusion"].get("first_alarm_batch") or {}).get(
                    "batch_id")),
            "change_point_batch_changed": (
                ((b["conclusion_cusum"].get("change_point") or {})
                 .get("batch_index"))
                != ((n["conclusion_cusum"].get("change_point") or {})
                    .get("batch_index"))),
            "latest_shift_sigma_before":
                b["conclusion"]["latest_shift_sigma"],
            "latest_shift_sigma_after":
                n["conclusion"]["latest_shift_sigma"],
            "batches_used_before": b["batches_used"],
            "batches_used_after": n["batches_used"],
        })

    def alarm_id(res):
        c = res["conclusion"]
        return (c["first_alarm"]["batch"]["sequence_no"]
                if c["any_alarm"] else None)

    before_id, after_id = alarm_id(base_result), alarm_id(new_result)
    overall_changed = any(t["changed"] for t in targets)
    return {
        "targets": targets,
        "overall": {
            "any_alarm_before": base_result["conclusion"]["any_alarm"],
            "any_alarm_after": new_result["conclusion"]["any_alarm"],
            "first_alarm_before": base_result["conclusion"]["first_alarm"],
            "first_alarm_after": new_result["conclusion"]["first_alarm"],
            "first_alarm_sequence_before": before_id,
            "first_alarm_sequence_after": after_id,
            "conclusion_changed": overall_changed,
        },
        "summary_note": (
            "比较排除异常批次前（父研究）后（复制研究）的报警方向、"
            "首次报警批次、估计变点与最新偏移；报警消失说明被排除批次"
            "主导了原结论"
        ),
    }


def _conclusion_summary(t: dict[str, Any]) -> dict[str, Any]:
    """可比较的结论指纹（顺序无关的字段子集）。"""
    c = t["conclusion"]
    cp_cusum = t["conclusion_cusum"].get("change_point")
    return {
        "alarmed": c["alarmed"],
        "first_alarm_batch_id": (
            c["first_alarm_batch"]["batch_id"]
            if c["first_alarm_batch"] else None),
        "first_alarm_direction": c["first_alarm_direction"],
        "change_point_batch_index": (
            cp_cusum["batch_index"] if cp_cusum else None),
        "drift_direction": c["drift_direction"],
    }
