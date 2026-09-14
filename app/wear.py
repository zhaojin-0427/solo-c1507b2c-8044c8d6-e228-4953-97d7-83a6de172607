"""服役磨损研究引擎（内部统一 mm / 循环数）。

磨损模型（每个尺寸 i）
======================
基线链给出初始制造散布（分布 / 相关与基线同一口径）；服役后尺寸随
循环数磨损：

    L_i(N) = L_i(0) + d_i · W_i(N)
    W_i(N) = μ_i(N) + σ_ri · N · Z_i

* d_i ∈ {+1,-1}：磨损后尺寸增大 / 减小；
* μ_i(N)：分段线性累计磨损曲线（折点线性插值，mm）；
* σ_ri：磨损速率标准差（mm/循环）——随机速率模型：同一零件整个服役期
  速率偏差恒定，N 循环后累计磨损标准差为 σ_ri·N（N=0 时无磨损不确定性）；
* Z_i ~ N(0,1)：跨尺寸按共用载荷相关矩阵 R_wear 相关（Cholesky 抽样）。

封闭环  C(N) = Σ s_i·L_i(N) = C(0) + Σ c_i·W_i(N)，c_i = s_i·d_i。
制造散布与磨损速率相互独立，方差直接相加：

    σ²_C(N) = σ²_mfg + N² · Σ_ij c_i c_j ρ^w_ij σ_ri σ_rj

极值法（WC）：均值 ± [ΣT_i + 3·σ_wear,C(N)]（制造取公差带线性累加，
磨损按 ±3σ 扩展，与热分析的标准不确定度扩展同口径）。
RSS：μ_C(N) ± 3σ_C(N)，超差率按正态近似。
蒙特卡洛：固定种子两个子流（SeedSequence([seed, 0x77656172]).spawn(2)）
——制造散布复用基线链抽样器，磨损速率按相关正态抽样；所有计算节点
共用同一组样本（同一批零件），首次越界循环按样本逐节点扫描得到。

维护编排
========
维护动作在节点循环数处生效（该节点即按维护后状态评估）：
* 加垫片：封闭环均值永久平移 sign×厚度，厚度不确定度自加入节点起计入；
* 更换零件：该尺寸累计磨损清零、自更换节点重新计循环（新零件新的
  速率抽样），更换成本与停机成本各计一次；
* 继续使用：无动作。

方案由三种贪心策略生成（最低成本优先 / 垫片优先 / 更换优先）外加
零成本现状方案，按「全周期违规数 → 最早越界点（晚者优先）→ 总成本」
升序排列。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import (
    NormalizedChain,
    _cholesky_psd,
    _s_arr,
    _sigma_arr,
    mean_gap,
    nominal_gap,
    normal_cdf,
    sample_dimensions,
)
from .units import to_mm
from .wear_schemas import WearDirection

Z_STAT = 3.0  # 统计界限 / 标准不确定度扩展倍数（与全站口径一致）

_WEAR_TAG = 0x77656172   # "wear" 子流标签
_REPL_TAG = 0x72706C     # "rpl" 更换件新速率子流标签


class WearError(ValueError):
    """磨损研究 / 维护编排输入错误（API 层映射为 422）。"""


# ------------------------------------------------------------------ 数据结构

@dataclass
class WearDim:
    """归一化后的单尺寸磨损参数（mm / 循环）。"""

    id: str
    sign: int                       # 闭环方向系数 s_i（含测量方向）
    wear_dir: int                   # +1 增大 / -1 减小
    curve_cycles: np.ndarray        # 折点循环数（严格递增，起点 0）
    curve_wear_mm: np.ndarray       # 折点累计磨损 mm（单调不减，起点 0）
    rate_sigma: float               # 磨损速率标准差 mm/循环
    raw: dict = field(default_factory=dict)


@dataclass
class WearModel:
    """归一化磨损研究模型。"""

    nc: NormalizedChain
    ids: list[str]
    dims: list[WearDim]
    r_wear: np.ndarray              # 磨损速率相关矩阵
    nodes: np.ndarray               # 计算节点（循环数，严格递增）
    lsl: float | None
    usl: float | None
    mc_samples: int
    seed: int


# ------------------------------------------------------------------ 归一化

def build_model(nc: NormalizedChain, payload) -> WearModel:
    """校验尺寸覆盖关系并把 API 请求归一化为 WearModel。"""
    chain_ids = [d.id for d in nc.dimensions]
    chain_set = set(chain_ids)
    given = [d.dimension_id for d in payload.dimensions]
    given_set = set(given)
    missing = sorted(chain_set - given_set)
    extra = sorted(given_set - chain_set)
    if missing:
        raise WearError(f"漏填尺寸的磨损参数: {missing}")
    if extra:
        raise WearError(f"磨损参数引用了基线链外尺寸: {extra}")

    spec_by_id = {d.dimension_id: d for d in payload.dimensions}
    sign_by_id = {d.id: d.sign for d in nc.dimensions}
    dims: list[WearDim] = []
    for dim_id in chain_ids:
        spec = spec_by_id[dim_id]
        u = spec.curve_unit.value
        dims.append(WearDim(
            id=dim_id,
            sign=sign_by_id[dim_id],
            wear_dir=1 if spec.wear_direction == WearDirection.INCREASE else -1,
            curve_cycles=np.array(
                [spec.wear_curve[0].start_cycles]
                + [seg.end_cycles for seg in spec.wear_curve],
                dtype=float),
            curve_wear_mm=np.array(
                [to_mm(spec.wear_curve[0].start_wear, u)]
                + [to_mm(seg.end_wear, u) for seg in spec.wear_curve],
                dtype=float),
            rate_sigma=to_mm(spec.rate_std_dev, u),
            raw={
                "wear_direction": spec.wear_direction.value,
                "curve_unit": u,
                "rate_std_dev": spec.rate_std_dev,
                "segments": [
                    {"start_cycles": seg.start_cycles,
                     "end_cycles": seg.end_cycles,
                     "start_wear": seg.start_wear,
                     "end_wear": seg.end_wear}
                    for seg in spec.wear_curve],
            },
        ))

    idx = {name: i for i, name in enumerate(chain_ids)}
    n = len(chain_ids)
    r_wear = np.eye(n)
    for c in payload.wear_rate_correlations:
        i, j = idx[c.dim_a], idx[c.dim_b]
        r_wear[i, j] = r_wear[j, i] = c.rho

    cu = payload.closure_unit.value
    lsl = (to_mm(payload.closure_lower_limit, cu)
           if payload.closure_lower_limit is not None else None)
    usl = (to_mm(payload.closure_upper_limit, cu)
           if payload.closure_upper_limit is not None else None)

    return WearModel(
        nc=nc, ids=chain_ids, dims=dims, r_wear=r_wear,
        nodes=np.array(payload.evaluation_cycles, dtype=float),
        lsl=lsl, usl=usl,
        mc_samples=payload.mc_samples, seed=payload.random_seed)


def _coef(model: WearModel) -> np.ndarray:
    """封闭环磨损系数 c_i = s_i·d_i。"""
    return np.array([d.sign * d.wear_dir for d in model.dims], dtype=float)


def _rate_arr(model: WearModel) -> np.ndarray:
    return np.array([d.rate_sigma for d in model.dims], dtype=float)


def _mean_wear_matrix(model: WearModel,
                      nodes: np.ndarray | None = None) -> np.ndarray:
    """各尺寸在计算节点处的平均累计磨损 μ_i(N)，形状 (节点数, k)。"""
    nodes = model.nodes if nodes is None else nodes
    return np.stack([
        np.interp(nodes, d.curve_cycles, d.curve_wear_mm)
        for d in model.dims
    ], axis=1)


def _mfg_stats(nc: NormalizedChain):
    """基线制造散布：均值间隙、名义间隙、封闭环方差与逐尺寸贡献、公差半宽和。"""
    nd = nc.dimensions
    s = _s_arr(nd)
    sigma = _sigma_arr(nd)
    weighted = s * sigma
    cov = nc.corr * weighted[:, None] * weighted[None, :]
    var = float(cov.sum())
    contrib = cov.sum(axis=1)
    half = float(sum(d.half_width for d in nd))
    return mean_gap(nc), nominal_gap(nc), var, contrib, half


def _wear_variance(r_wear: np.ndarray, coef: np.ndarray,
                   sw: np.ndarray) -> tuple[float, np.ndarray]:
    """磨损速率不确定度传播的封闭环方差与逐尺寸贡献行。

    sw_i 为各尺寸在评估点的累计磨损标准差（σ_ri × 有效循环数）。
    """
    w = coef * sw
    cov = r_wear * w[:, None] * w[None, :]
    return float(cov.sum()), cov.sum(axis=1)


def _reject_normal(mean: float, sigma: float,
                   lsl: float | None, usl: float | None) -> float | None:
    if lsl is None and usl is None:
        return None
    if sigma <= 0:
        return 1.0 if ((lsl is not None and mean < lsl)
                       or (usl is not None and mean > usl)) else 0.0
    p = 0.0
    if lsl is not None:
        p += float(normal_cdf((lsl - mean) / sigma))
    if usl is not None:
        p += float(1.0 - normal_cdf((usl - mean) / sigma))
    return p


# ------------------------------------------------------------- 蒙特卡洛基础

def _base_samples(model: WearModel, n_samples: int,
                  seed: int | None = None):
    """研究级固定种子样本：制造散布封闭环 C(0) 与磨损速率斜率 g。

    返回 (closure0, g, Z)：
    * closure0 (n,)：基线制造散布下的封闭环样本（复用基线链抽样器）；
    * g (n,)：每样本的封闭环磨损速率 Σ c_i·σ_ri·Z_i，节点 N 的磨损随机
      部分为 N·g（随机速率模型，同一零件全寿命同一速率偏差）；
    * Z (n,k)：相关标准正态（更换零件时第 0 寿命段复用同一 Z）。
    seed 缺省用研究种子；维护编排可显式覆盖（结果随请求冻结）。
    """
    nc = model.nc
    k = len(model.dims)
    base = model.seed if seed is None else seed
    ss = np.random.SeedSequence([base, _WEAR_TAG])
    seed_mfg, seed_wear = ss.spawn(2)
    x = sample_dimensions(nc, n_samples, int(seed_mfg.generate_state(1)[0]))
    closure0 = x @ _s_arr(nc.dimensions)
    rng_w = np.random.default_rng(seed_wear)
    z = rng_w.standard_normal((n_samples, k))
    if not np.allclose(model.r_wear, np.eye(k), atol=1e-12):
        z = z @ _cholesky_psd(model.r_wear).T
    g = z @ (_coef(model) * _rate_arr(model))
    return closure0, g, z


def _replacement_rate_samples(model: WearModel, n_samples: int,
                              segment: int,
                              seed: int | None = None) -> np.ndarray:
    """第 segment 寿命段（更换后的新零件）的相关速率抽样 Z，形状 (n, k)。

    段号只取决于「第几次更换后」，与具体方案无关——不同方案同一寿命段
    使用同一组速率样本，方案间可比。
    """
    k = len(model.dims)
    base = model.seed if seed is None else seed
    rng = np.random.default_rng(
        np.random.SeedSequence([base, _WEAR_TAG, _REPL_TAG, segment]))
    z = rng.standard_normal((n_samples, k))
    if not np.allclose(model.r_wear, np.eye(k), atol=1e-12):
        z = z @ _cholesky_psd(model.r_wear).T
    return z


def _mc_node_summaries(nodes: np.ndarray, closure_iter,
                       lsl: float | None, usl: float | None) -> tuple[list, dict]:
    """逐节点封闭环样本统计 + 首次越界循环分布。

    closure_iter 按节点顺序产出 (n,) 封闭环样本；同一组样本贯穿所有
    节点（同一批零件），首次越界按样本逐节点扫描。
    """
    stats = []
    remaining = None
    first_idx = None
    for j, closure in enumerate(closure_iter):
        closure = np.asarray(closure, dtype=float)
        n = len(closure)
        if remaining is None:
            remaining = np.ones(n, dtype=bool)
            first_idx = np.full(n, -1, dtype=int)
        out = np.zeros(n, dtype=bool)
        if lsl is not None:
            out |= closure < lsl
        if usl is not None:
            out |= closure > usl
        newly = out & remaining
        first_idx[newly] = j
        remaining &= ~newly
        q005, q995 = np.quantile(closure, [0.005, 0.995])
        stats.append({
            "cycles": float(nodes[j]),
            "mean_gap_mm": float(closure.mean()),
            "sigma_mm": float(closure.std(ddof=1)) if n > 1 else 0.0,
            "lower_quantile_mm": float(q005),
            "upper_quantile_mm": float(q995),
            "min_sample_mm": float(closure.min()),
            "max_sample_mm": float(closure.max()),
            "reject_probability": float(out.mean()),
        })

    n = len(remaining)
    per_node = []
    for j, cycles in enumerate(nodes):
        cnt = int((first_idx == j).sum())
        per_node.append({"cycles": float(cycles), "count": cnt,
                         "fraction": cnt / n})
    never_cnt = int(remaining.sum())
    exceeded_cycles = nodes[first_idx[first_idx >= 0]]
    quantiles = None
    if len(exceeded_cycles):
        q = np.quantile(exceeded_cycles, [0.05, 0.25, 0.5, 0.75, 0.95])
        quantiles = {
            "min": float(exceeded_cycles.min()),
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]),
            "max": float(exceeded_cycles.max()),
            "mean": float(exceeded_cycles.mean()),
        }
    distribution = {
        "per_node": per_node,
        "never_exceeded": {"count": never_cnt, "fraction": never_cnt / n},
        "exceeded_count": n - never_cnt,
        "first_exceedance_cycle_quantiles": quantiles,
        "convention": "同一组样本贯穿全部计算节点（同一批零件）；"
                      "逐样本按节点顺序扫描，首次越出封闭环限值的节点即"
                      "首次越界循环；节点之间按分段线性轨迹可能更早越界，"
                      "未计入",
    }
    return stats, distribution


# ------------------------------------------------------------------ 研究评估

def analyze(model: WearModel) -> dict:
    """逐计算节点评估：极值 / RSS / 固定种子蒙特卡洛 + 首次越界分布。"""
    nc = model.nc
    nodes = model.nodes
    lsl, usl = model.lsl, model.usl
    coef = _coef(model)
    rate = _rate_arr(model)
    mu0, c0, var_mfg, contrib_mfg, half_mfg = _mfg_stats(nc)
    wear_mean = _mean_wear_matrix(model)              # (J, k)

    node_results = []
    for j, cycles in enumerate(nodes):
        shift = float(coef @ wear_mean[j])
        mean = mu0 + shift
        sw = rate * cycles
        var_wear, contrib_wear = _wear_variance(model.r_wear, coef, sw)
        sigma_wear = math.sqrt(max(var_wear, 0.0))
        var_total = var_mfg + var_wear
        sigma_total = math.sqrt(max(var_total, 0.0))
        wc_half = half_mfg + Z_STAT * sigma_wear
        lower, upper = mean - wc_half, mean + wc_half
        wc_reject = None
        if lsl is not None or usl is not None:
            wc_reject = 1.0 if ((lsl is not None and lower < lsl)
                                or (usl is not None and upper > usl)) else 0.0
        margin = None
        if lsl is not None and usl is not None:
            margin = min(lower - lsl, usl - upper)
        elif lsl is not None:
            margin = lower - lsl
        elif usl is not None:
            margin = usl - upper

        per_dimension = []
        for i, d in enumerate(model.dims):
            per_dimension.append({
                "dimension_id": d.id,
                "closure_sign": d.sign,
                "wear_direction": "increase" if d.wear_dir > 0 else "decrease",
                "cumulative_wear_mm": float(wear_mean[j, i]),
                "closure_mean_shift_mm": float(coef[i] * wear_mean[j, i]),
                "wear_rate_std_mm_per_cycle": float(rate[i]),
                "wear_std_at_node_mm": float(rate[i] * cycles),
                "manufacturing_variance_contribution_mm2":
                    float(contrib_mfg[i]),
                "wear_variance_contribution_mm2": float(contrib_wear[i]),
                "variance_share": (
                    float(contrib_mfg[i] + contrib_wear[i]) / var_total
                    if var_total > 0 else 0.0),
            })

        node_results.append({
            "cycles": float(cycles),
            "mean_gap_mm": mean,
            "wear_mean_shift_mm": shift,
            "worst_case": {
                "lower_bound_mm": lower,
                "upper_bound_mm": upper,
                "total_extreme_half_width_mm": wc_half,
                "manufacturing_half_width_mm": half_mfg,
                "wear_extended_half_width_mm": Z_STAT * sigma_wear,
                "reject_probability": wc_reject,
                "convention": "制造散布取公差带线性累加 ΣT_i；磨损不确定度"
                              f"按 ±{Z_STAT:g}σ 扩展（σ_ri×N 相关传播）",
            },
            "rss": {
                "sigma_mm": sigma_total,
                "sigma_manufacturing_mm": math.sqrt(max(var_mfg, 0.0)),
                "sigma_wear_mm": sigma_wear,
                "lower_bound_mm": mean - Z_STAT * sigma_total,
                "upper_bound_mm": mean + Z_STAT * sigma_total,
                "z": Z_STAT,
                "reject_probability": _reject_normal(
                    mean, sigma_total, lsl, usl),
            },
            "spec_margin_mm": margin,
            "per_dimension": per_dimension,
        })

    # ---------- 固定种子蒙特卡洛（所有节点共用同一组样本）----------
    closure0, g, _ = _base_samples(model, model.mc_samples)
    shifts = wear_mean @ coef                          # (J,)

    def closure_iter():
        for j, cycles in enumerate(nodes):
            yield closure0 + shifts[j] + cycles * g

    mc_stats, exceedance = _mc_node_summaries(
        nodes, closure_iter(), lsl, usl)
    for j, st in enumerate(mc_stats):
        node_results[j]["monte_carlo"] = st

    # ---------- 解析最早越界节点（先 WC 后 RSS，与热分析同口径）----------
    first_breach = None
    for j, r in enumerate(node_results):
        wc = r["worst_case"]
        if (lsl is not None and wc["lower_bound_mm"] < lsl) or (
                usl is not None and wc["upper_bound_mm"] > usl):
            first_breach = _breach_info(j, r, "worst_case", lsl, usl)
            break
    if first_breach is None:
        for j, r in enumerate(node_results):
            rr = r["rss"]
            if (lsl is not None and rr["lower_bound_mm"] < lsl) or (
                    usl is not None and rr["upper_bound_mm"] > usl):
                first_breach = _breach_info(j, r, "rss", lsl, usl)
                break

    return {
        "nodes": node_results,
        "summary": {
            "node_count": len(nodes),
            "final_cycles": float(nodes[-1]),
            "worst_reject_probability": {
                "worst_case": max(
                    (r["worst_case"]["reject_probability"] or 0.0)
                    for r in node_results),
                "rss": max(r["rss"]["reject_probability"]
                           for r in node_results),
                "monte_carlo": max(r["monte_carlo"]["reject_probability"]
                                   for r in node_results),
            },
            "first_spec_breach": first_breach,
            "first_exceedance_monte_carlo": exceedance,
        },
        "baseline_manufacturing": {
            "nominal_gap_mm": c0,
            "mean_gap_mm": mu0,
            "sigma_mm": math.sqrt(max(var_mfg, 0.0)),
            "total_tolerance_half_width_mm": half_mfg,
        },
    }


def _breach_info(j: int, node_result: dict, criterion: str,
                 lsl, usl) -> dict:
    """越界方向：边界实际穿过 LSL 记 lower，穿过 USL 记 upper。"""
    side = []
    b = node_result[criterion]
    lo, hi = b["lower_bound_mm"], b["upper_bound_mm"]
    if lsl is not None and lo < lsl:
        side.append("lower")
    if usl is not None and hi > usl:
        side.append("upper")
    return {
        "node_index": j,
        "cycles": node_result["cycles"],
        "criterion": criterion,
        "breach_side": side,
        "bounds_mm": [lo, hi],
        "spec_margin_mm": node_result["spec_margin_mm"],
    }


def model_snapshot(model: WearModel) -> dict:
    """归一化磨损模型快照（随研究冻结；重复计算据此可精确复现）。"""
    return {
        "dimension_order": model.ids,
        "dimensions": [
            {
                "dimension_id": d.id,
                "closure_sign": d.sign,
                "wear_direction": "increase" if d.wear_dir > 0 else "decrease",
                "curve_normalized": {
                    "cycles": [float(x) for x in d.curve_cycles],
                    "cumulative_wear_mm": [float(x) for x in d.curve_wear_mm],
                },
                "rate_std_dev_mm_per_cycle": float(d.rate_sigma),
                "submitted": d.raw,
            }
            for d in model.dims
        ],
        "wear_rate_correlation_matrix": model.r_wear.tolist(),
        "evaluation_cycles": [float(x) for x in model.nodes],
        "closure_spec_mm": {"lower_limit": model.lsl,
                            "upper_limit": model.usl},
        "monte_carlo": {
            "samples": model.mc_samples,
            "seed": model.seed,
            "substreams": "SeedSequence([seed, 0x77656172]).spawn(2)："
                          "制造散布（复用基线链抽样器）/ 磨损速率相关正态",
        },
        "formula": "C(N) = C(0) + Σ s_i·d_i·[μ_i(N) + σ_ri·N·Z_i]",
    }


# ------------------------------------------------------------ 维护编排

@dataclass
class _Shim:
    shim_id: str
    name: str
    thickness_mm: float
    sigma_mm: float
    sign: int
    cost: float


@dataclass
class _Action:
    node: float          # 生效节点（循环数）
    kind: str            # "shim" | "replace"
    target: str          # shim_id 或尺寸 id
    cost: float          # 动作成本（不含停机）


@dataclass
class _MaintContext:
    """维护编排的归一化输入。"""

    shims: list[_Shim]
    replace_cost: dict[str, float]     # 可更换件 -> 单次更换成本
    maintainable: set[float]
    downtime_cost: float
    criterion: str                     # "worst_case" | "rss"


def _node_state(model: WearModel, cycles: float, repl: dict[str, list[float]],
                shims_at: list[tuple[float, _Shim]],
                mu0: float, var_mfg: float, half_mfg: float):
    """单节点在给定维护状态下的 (均值, σ_total, WC 半宽)。

    repl: 尺寸 id -> 已发生的更换节点列表；shims_at: (节点, 垫片) 列表。
    """
    coef = _coef(model)
    rate = _rate_arr(model)
    shift = 0.0
    shim_var = 0.0
    for node, shim in shims_at:
        if node <= cycles:
            shift += shim.sign * shim.thickness_mm
            shim_var += shim.sigma_mm ** 2
    sw = np.zeros(len(model.dims))
    for i, d in enumerate(model.dims):
        r = 0.0
        for x in repl.get(d.id, ()):  # 取不超过当前节点的最近一次更换
            if x <= cycles:
                r = max(r, x)
        wear_now = float(np.interp(cycles, d.curve_cycles, d.curve_wear_mm))
        wear_at_r = float(np.interp(r, d.curve_cycles, d.curve_wear_mm))
        shift += coef[i] * (wear_now - wear_at_r)
        sw[i] = rate[i] * (cycles - r)
    var_wear, _ = _wear_variance(model.r_wear, coef, sw)
    mean = mu0 + shift
    sigma = math.sqrt(max(var_mfg + var_wear + shim_var, 0.0))
    half = half_mfg + Z_STAT * math.sqrt(max(var_wear + shim_var, 0.0))
    return mean, sigma, half


def _violated(model: WearModel, mean: float, sigma: float, half: float,
              criterion: str) -> bool:
    lsl, usl = model.lsl, model.usl
    if criterion == "worst_case":
        lo, hi = mean - half, mean + half
    else:
        lo, hi = mean - Z_STAT * sigma, mean + Z_STAT * sigma
    return ((lsl is not None and lo < lsl)
            or (usl is not None and hi > usl))


def _orchestrate(model: WearModel, ctx: _MaintContext, strategy: str,
                 mu0: float, var_mfg: float, half_mfg: float) -> list[_Action]:
    """前向贪心编排：逐节点检查，违规且可维护时按策略选单个动作。

    策略：
    * cheapest：动作成本（含停机）最低且能恢复合规者优先；
    * shim_first：先试垫片（成本升序），再试更换；
    * replace_first：先试更换（成本升序），再试垫片。
    每个节点至多一个动作；动作在当前节点即生效并复检。
    """
    actions: list[_Action] = []
    repl: dict[str, list[float]] = {}
    shims_at: list[tuple[float, _Shim]] = []

    def ordered_candidates():
        cands = [("shim", s, s.cost) for s in ctx.shims]
        cands += [("replace", d, c)
                  for d, c in ctx.replace_cost.items()]
        if strategy == "cheapest":
            cands.sort(key=lambda c: (c[2], c[0],
                                      c[1] if isinstance(c[1], str) else c[1].shim_id))
        elif strategy == "shim_first":
            cands.sort(key=lambda c: (0 if c[0] == "shim" else 1, c[2]))
        else:  # replace_first
            cands.sort(key=lambda c: (0 if c[0] == "replace" else 1, c[2]))
        return cands

    for cycles in model.nodes:
        mean, sigma, half = _node_state(
            model, cycles, repl, shims_at, mu0, var_mfg, half_mfg)
        if not _violated(model, mean, sigma, half, ctx.criterion):
            continue
        if float(cycles) not in ctx.maintainable:
            continue  # 不可维护节点：违规保留，由评估阶段统计
        chosen = None
        for kind, target, cost in ordered_candidates():
            if kind == "shim":
                trial = shims_at + [(float(cycles), target)]
                m2, s2, h2 = _node_state(
                    model, cycles, repl, trial, mu0, var_mfg, half_mfg)
            else:
                trial_repl = {k: list(v) for k, v in repl.items()}
                trial_repl.setdefault(target, []).append(float(cycles))
                m2, s2, h2 = _node_state(
                    model, cycles, trial_repl, shims_at,
                    mu0, var_mfg, half_mfg)
            if not _violated(model, m2, s2, h2, ctx.criterion):
                chosen = (kind, target, cost)
                break
        if chosen is None:
            continue  # 单动作无法恢复合规：保留违规，不在本节点动作
        kind, target, cost = chosen
        if kind == "shim":
            shims_at.append((float(cycles), target))
            actions.append(_Action(float(cycles), "shim",
                                   target.shim_id, target.cost))
        else:
            repl.setdefault(target, []).append(float(cycles))
            actions.append(_Action(float(cycles), "replace", target, cost))
    return actions


def _plan_arrays(model: WearModel, actions: list[_Action],
                 shim_by_id: dict[str, _Shim]):
    """方案在各节点的：均值平移、逐尺寸有效循环数、垫片方差、更换节点表。"""
    nodes = model.nodes
    coef = _coef(model)
    k = len(model.dims)
    repl: dict[str, list[float]] = {}
    shim_events: list[tuple[float, _Shim]] = []
    for a in actions:
        if a.kind == "shim":
            shim_events.append((a.node, shim_by_id[a.target]))
        else:
            repl.setdefault(a.target, []).append(a.node)

    eff_cycles = np.zeros((len(nodes), k))
    wear_shift = np.zeros(len(nodes))
    for i, d in enumerate(model.dims):
        rs = [0.0] + sorted(repl.get(d.id, []))
        # 每个节点取不超过它的最近一次更换
        seg = np.searchsorted(rs, nodes, side="right") - 1
        rvals = np.array(rs, dtype=float)[seg]
        wear_now = np.interp(nodes, d.curve_cycles, d.curve_wear_mm)
        wear_at_r = np.interp(rvals, d.curve_cycles, d.curve_wear_mm)
        wear_shift += coef[i] * (wear_now - wear_at_r)
        eff_cycles[:, i] = nodes - rvals
    shim_shift = np.zeros(len(nodes))
    shim_var = np.zeros(len(nodes))
    for node, shim in shim_events:
        mask = nodes >= node
        shim_shift[mask] += shim.sign * shim.thickness_mm
        shim_var[mask] += shim.sigma_mm ** 2
    return wear_shift + shim_shift, eff_cycles, shim_var, repl


def _evaluate_plan(model: WearModel, ctx: _MaintContext, actions: list[_Action],
                   strategy: str, shim_by_id: dict[str, _Shim],
                   mu0: float, var_mfg: float, half_mfg: float,
                   mc_samples: int, seed: int) -> dict:
    """方案完整评估：逐节点解析界 + 固定种子 MC 复核 + 违规与成本。"""
    nodes = model.nodes
    lsl, usl = model.lsl, model.usl
    coef = _coef(model)
    rate = _rate_arr(model)
    mean_shift, eff_cycles, shim_var, repl = _plan_arrays(
        model, actions, shim_by_id)

    # ---------- 解析（逐节点）----------
    sw = rate[None, :] * eff_cycles                    # (J, k)
    w = coef[None, :] * sw
    cov = model.r_wear[None, :, :] * w[:, :, None] * w[:, None, :]
    var_wear = cov.sum(axis=(1, 2))
    sigma = np.sqrt(np.maximum(var_mfg + var_wear + shim_var, 0.0))
    half = half_mfg + Z_STAT * np.sqrt(np.maximum(var_wear + shim_var, 0.0))
    means = mu0 + mean_shift
    if ctx.criterion == "worst_case":
        lo_b, hi_b = means - half, means + half
    else:
        lo_b, hi_b = means - Z_STAT * sigma, means + Z_STAT * sigma
    violated = np.zeros(len(nodes), dtype=bool)
    if lsl is not None:
        violated |= lo_b < lsl
    if usl is not None:
        violated |= hi_b > usl
    violation_cycles = [float(nodes[j]) for j in np.nonzero(violated)[0]]
    first_breach = violation_cycles[0] if violation_cycles else None

    # ---------- 蒙特卡洛复核（研究种子派生；制造样本与研究同源）----------
    closure0, _, z0 = _base_samples(model, mc_samples, seed=seed)
    max_seg = 0
    for r in repl.values():
        max_seg = max(max_seg, len(r))
    z_segs = [z0] + [
        _replacement_rate_samples(model, mc_samples, seg, seed=seed)
        for seg in range(1, max_seg + 1)
    ]
    # 各尺寸在各节点的寿命段号（第几次更换后）
    seg_of = np.zeros((len(nodes), len(model.dims)), dtype=int)
    for i, d in enumerate(model.dims):
        rs = sorted(repl.get(d.id, []))
        if rs:
            seg_of[:, i] = np.searchsorted(rs, nodes, side="right")

    def closure_iter():
        cr = coef * rate
        for j in range(len(nodes)):
            z_eff = np.empty_like(z0)
            for i in range(len(model.dims)):
                z_eff[:, i] = z_segs[seg_of[j, i]][:, i]
            yield closure0 + mean_shift[j] + (
                z_eff * (cr * eff_cycles[j])[None, :]).sum(axis=1)

    mc_stats, exceedance = _mc_node_summaries(nodes, closure_iter(), lsl, usl)

    # ---------- 成本 ----------
    shim_cost = sum(a.cost for a in actions if a.kind == "shim")
    repl_cost = sum(a.cost for a in actions if a.kind == "replace")
    event_nodes = {a.node for a in actions}
    downtime = ctx.downtime_cost * len(event_nodes)
    total = shim_cost + repl_cost + downtime

    node_rows = []
    for j in range(len(nodes)):
        node_rows.append({
            "cycles": float(nodes[j]),
            "mean_gap_mm": float(means[j]),
            "sigma_mm": float(sigma[j]),
            "criterion_bound_mm": [float(lo_b[j]), float(hi_b[j])],
            "violated": bool(violated[j]),
            "maintainable": float(nodes[j]) in ctx.maintainable,
            "monte_carlo_reject_probability":
                mc_stats[j]["reject_probability"],
        })

    return {
        "strategy": strategy,
        "actions": [
            {
                "node_cycles": a.node,
                "kind": a.kind,
                "target": a.target,
                "description": (
                    f"加垫片 {a.target}" if a.kind == "shim"
                    else f"更换零件 {a.target}（累计磨损清零重新计循环）"),
                "cost": a.cost,
            }
            for a in actions
        ],
        "maintenance_events": len(event_nodes),
        "cost": {
            "shims": shim_cost,
            "replacements": repl_cost,
            "downtime": downtime,
            "total": total,
        },
        "metrics": {
            "criterion": ctx.criterion,
            "violations": len(violation_cycles),
            "violation_cycles": violation_cycles,
            "first_breach_cycles": first_breach,
            "worst_reject_probability_mc": max(
                s["reject_probability"] for s in mc_stats),
            "first_exceedance_monte_carlo": exceedance,
        },
        "nodes": node_rows,
    }


def search_maintenance(model: WearModel, payload) -> dict:
    """生成维护方案并按「全周期违规数 → 最早越界点 → 总成本」排列。"""
    chain_ids = set(model.ids)
    unknown_locked = sorted(set(payload.locked_dimensions) - chain_ids)
    if unknown_locked:
        raise WearError(f"锁定尺寸不在基线链上: {unknown_locked}")
    unknown_repl = sorted(set(payload.replacement_costs) - chain_ids)
    if unknown_repl:
        raise WearError(f"更换成本引用了基线链外尺寸: {unknown_repl}")
    node_set = {float(x) for x in model.nodes}
    bad_nodes = sorted({float(n) for n in payload.maintainable_cycles}
                       - node_set)
    if bad_nodes:
        raise WearError(
            f"维护节点必须是研究计算节点的子集；不在计算节点中: {bad_nodes}"
            f"（研究节点: {[float(x) for x in model.nodes]}）")

    shims = [
        _Shim(s.shim_id, s.name, to_mm(s.thickness, s.unit.value),
              to_mm(s.std_dev, s.unit.value), s.closure_sign, s.cost)
        for s in payload.shim_candidates
    ]
    ctx = _MaintContext(
        shims=shims,
        replace_cost={k: float(v) for k, v in payload.replacement_costs.items()},
        maintainable={float(n) for n in payload.maintainable_cycles},
        downtime_cost=float(payload.downtime_cost),
        criterion=payload.criterion,
    )
    seed = model.seed if payload.random_seed is None else payload.random_seed

    mu0, _, var_mfg, _, half_mfg = _mfg_stats(model.nc)

    # 方案生成：零动作现状 + 三种贪心策略（动作序列相同的去重）
    raw_plans: list[tuple[str, list[_Action]]] = [("continue_only", [])]
    for strategy in ("cheapest", "shim_first", "replace_first"):
        raw_plans.append((strategy, _orchestrate(
            model, ctx, strategy, mu0, var_mfg, half_mfg)))
    plans: list[tuple[str, list[_Action]]] = []
    seen: set[tuple] = set()
    for strategy, actions in raw_plans:
        sig = tuple((a.node, a.kind, a.target) for a in actions)
        if sig in seen:
            continue
        seen.add(sig)
        plans.append((strategy, actions))

    shim_by_id = {s.shim_id: s for s in shims}
    evaluated = [
        _evaluate_plan(model, ctx, actions, strategy, shim_by_id,
                       mu0, var_mfg, half_mfg, payload.mc_samples, seed)
        for strategy, actions in plans
    ]

    def rank_key(row):
        fb = row["metrics"]["first_breach_cycles"]
        # 最早越界点：越晚越优；从不越界最优（映射为 -inf 排最前）
        breach_key = -(fb if fb is not None else float("inf"))
        return (row["metrics"]["violations"], breach_key,
                row["cost"]["total"])

    evaluated.sort(key=rank_key)
    candidates = []
    for rank, row in enumerate(evaluated[:payload.max_candidates], 1):
        candidates.append({"rank": rank, **row})

    return {
        "ranking_rule": [
            "全周期违规数（计算节点中判据边界越出封闭环限值的节点数，升序）",
            "最早越界点（首次违规节点循环数，越晚越优；从不越界最优）",
            "总成本（垫片 + 更换 + 停机，升序）",
        ],
        "criterion": ctx.criterion,
        "strategies_evaluated": [s for s, _ in plans],
        "evaluated_plans": len(evaluated),
        "returned": len(candidates),
        "monte_carlo": {
            "samples_per_plan": payload.mc_samples,
            "seed": seed,
            "substreams": "制造散布与研究同种子同源；第 s 次更换后的新零件"
                          "速率由 SeedSequence([seed, 0x77656172, 0x72706C, s])"
                          " 派生（段号与方案无关，方案间可比）",
        },
        "maintenance_convention":
            "维护动作在节点循环数处生效（该节点即按维护后状态评估）；"
            "每个维护节点至多一个动作；垫片可重复使用同一规格；"
            "更换件累计磨损清零、自更换节点重新计循环并抽新速率",
        "candidates": candidates,
    }
