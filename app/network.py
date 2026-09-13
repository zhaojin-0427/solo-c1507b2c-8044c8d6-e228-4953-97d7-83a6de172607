"""多闭环公差网络引擎（内部统一 mm）。

模型
====
同一套装配的一组**共享尺寸**（尺寸池）构成有向边集合；每个闭环是一条
**有序路径**：按经过顺序列出尺寸 id 与沿边方向（sign=+1 沿 start→end，
-1 沿 end→start），路径必须逐段衔接并回到起点。闭环 L 的功能要求

    C_L = Σ_i s_i·X_i ，  s_i = 路径方向 sign × 尺寸测量方向 direction

与单链引擎同一口径：均值 μ_L = Σ s_i·(N_i+m_i)，极值界 μ_L ± Σ|s_i|·T_i，
RSS σ_L² = s_Lᵀ(DρD)s_L（ρ 为尺寸值 Pearson 相关），±3σ 统计界，
超差率按正态近似。单位换算、理论标准差、显式 σ 语义与 Gaussian copula
抽样全部复用单链引擎（app.engine）。

共享抽样
========
蒙特卡洛对**尺寸池每轮只抽样一次**（n×k 样本矩阵由 engine.sample_dimensions
统一生成），所有闭环的封闭环样本由同一样本矩阵线性组合 C = X·Sᵀ 得到——
共享尺寸在同一轮的所有闭环中取值一致，闭环间协方差、联合合格率与同时
失效组合由此自然体现（同一固定种子下结果可精确复现）。

联合指标
========
* 闭环间协方差：解析 Σ_L = S(DρD)Sᵀ 与 MC 样本协方差并列；
* 联合合格率：同一轮样本中全部闭环落在各自规格内的经验比例；
* 同时失效组合：同一轮中失效闭环集合的经验分布（按频次排序）与
  两两联合超差矩阵；
* 尺寸×闭环敏感度矩阵：方向系数、极值带宽占比、RSS 方差占比与
  MC 回归斜率/方差占比。非路径尺寸恒为 0——其影响已通过路径尺寸间的
  相关项计入闭环方差（ρ 在协方差传播中生效）。

方案分支搜索
============
锁定尺寸保持版本当前公差带与 σ；未锁定尺寸在 scale_levels 网格上逐尺寸
选择公差带比例（σ 策略与单链批量调整同口径），beam search 以
「联合超差率代理（各闭环 RSS 超差率之和）→ 收紧成本」剪枝，候选再用
共享抽样蒙特卡洛复核，最终按
**全部闭环达标（MC 联合超差率 ≤ 目标）→ 最高优先级闭环的最小极值法
余量（大者优先）→ 总收紧成本（升序）** 排列。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import engine
from .engine import NormalizedChain, NormDimension
from .units import to_mm

Z_STAT = engine.Z_STAT  # RSS 统计界限倍数（与单链一致，±3σ）

MAX_FAILURE_COMBINATIONS = 20   # 同时失效组合返回上限
TOP_CANDIDATES_MC = 30          # 解析排序后进入 MC 复核的候选数


class NetworkError(ValueError):
    """多闭环网络输入 / 求解错误（API 层映射为 422）。"""


# ------------------------------------------------------------------ 数据结构

@dataclass
class ResolvedLoop:
    """路径已解析的闭环（数值均为 mm；系数 = 路径方向 × 测量方向）。"""

    id: str
    note: str
    priority: int
    lsl_mm: float
    usl_mm: float
    coefficients: dict[str, int]   # 尺寸 id -> ±1
    path: list[dict]               # 逐步快照（含声明方向与最终系数）
    spec_original: dict


@dataclass
class NormalizedNetwork:
    """规范化后的多闭环网络（尺寸池 sign 字段无意义，恒为 +1）。"""

    dimensions: list[NormDimension]
    loops: list[ResolvedLoop]
    corr: np.ndarray               # k×k 尺寸值 Pearson 相关矩阵
    mc_samples: int
    seed: int
    name: str = ""

    @property
    def ids(self) -> list[str]:
        return [d.id for d in self.dimensions]

    def coefficient_matrix(self) -> np.ndarray:
        """L×k 系数矩阵 S：S[l, i] 为尺寸 i 在闭环 l 中的方向系数。"""
        idx = {d.id: i for i, d in enumerate(self.dimensions)}
        s = np.zeros((len(self.loops), len(self.dimensions)))
        for li, loop in enumerate(self.loops):
            for dim_id, coef in loop.coefficients.items():
                s[li, idx[dim_id]] = coef
        return s


def normalize_network(defn, mc_samples: int, seed: int,
                      name: str = "") -> NormalizedNetwork:
    """把校验后的 NetworkDefinition 换算为 mm 并解析各闭环系数。"""
    norm_dims: list[NormDimension] = []
    for d in defn.dimensions:
        u = d.unit.value if hasattr(d.unit, "value") else str(d.unit)
        n_mm = to_mm(d.nominal, u)
        es_mm = to_mm(d.upper_deviation, u)
        ei_mm = to_mm(d.lower_deviation, u)
        t_mm = (es_mm - ei_mm) / 2.0
        sigma_explicit = d.std_dev is not None
        sigma_mm = (
            to_mm(d.std_dev, u) if sigma_explicit
            else engine._theoretical_sigma(d.distribution.value, t_mm)
        )
        norm_dims.append(
            NormDimension(
                id=d.id, sign=1,
                nominal=n_mm, upper_dev=es_mm, lower_dev=ei_mm,
                mid=(es_mm + ei_mm) / 2.0, half_width=t_mm,
                sigma=sigma_mm, sigma_explicit=sigma_explicit,
                distribution=d.distribution.value,
                start=d.start, end=d.end, direction=d.direction, unit=u,
                original={
                    "nominal": d.nominal,
                    "upper_deviation": d.upper_deviation,
                    "lower_deviation": d.lower_deviation,
                    "std_dev": d.std_dev,
                    "unit": u,
                    "unilateral": d.is_unilateral,
                },
            )
        )

    by_id = {d.id: d for d in defn.dimensions}
    loops: list[ResolvedLoop] = []
    for loop in defn.loops:
        lu = loop.unit.value if hasattr(loop.unit, "value") else str(loop.unit)
        coefficients: dict[str, int] = {}
        path: list[dict] = []
        for e in loop.path:
            d = by_id[e.dimension_id]
            coef = int(e.sign) * int(d.direction)
            coefficients[e.dimension_id] = coef
            path.append({
                "dimension_id": e.dimension_id,
                "sign": int(e.sign),
                "measurement_direction": int(d.direction),
                "coefficient": coef,
                "edge": [d.start, d.end],
            })
        loops.append(ResolvedLoop(
            id=loop.id, note=loop.note, priority=int(loop.priority),
            lsl_mm=to_mm(loop.lower_limit, lu),
            usl_mm=to_mm(loop.upper_limit, lu),
            coefficients=coefficients, path=path,
            spec_original={
                "lower_limit": loop.lower_limit,
                "upper_limit": loop.upper_limit,
                "unit": lu,
            },
        ))

    ids = [d.id for d in defn.dimensions]
    idx = {name_: i for i, name_ in enumerate(ids)}
    corr = np.eye(len(ids))
    for c in defn.correlations:
        i, j = idx[c.dim_a], idx[c.dim_b]
        corr[i, j] = corr[j, i] = c.rho

    return NormalizedNetwork(
        dimensions=norm_dims, loops=loops, corr=corr,
        mc_samples=mc_samples, seed=seed, name=name,
    )


# ------------------------------------------------------------------ 网络计算

def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    """由协方差矩阵求相关矩阵；零方差行/列置 0（对角保持 1）。"""
    d = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.zeros_like(cov)
    nz = d > 0
    if nz.any():
        sub = cov[np.ix_(nz, nz)]
        corr[np.ix_(nz, nz)] = sub / d[nz][:, None] / d[nz][None, :]
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_network(net: NormalizedNetwork,
                    sigmas: np.ndarray | None = None,
                    mids: np.ndarray | None = None,
                    halfs: np.ndarray | None = None,
                    explicit_flags=None,
                    n_samples: int | None = None,
                    seed: int | None = None) -> dict:
    """计算全部闭环的 WC/RSS/MC 与联合指标；覆盖参数供方案分支评估。

    蒙特卡洛对尺寸池每轮只抽样一次：X (n×k) 由 engine.sample_dimensions
    生成（相关结构 / 显式 σ 口径与单链完全一致），C = X·Sᵀ 为各闭环样本。
    """
    dims = net.dimensions
    k = len(dims)
    n_loops = len(net.loops)
    s_mat = net.coefficient_matrix()
    if sigmas is None:
        sigmas = np.array([d.sigma for d in dims], dtype=float)
    if mids is None:
        mids = np.array([d.mid for d in dims], dtype=float)
    if halfs is None:
        halfs = np.array([d.half_width for d in dims], dtype=float)
    nominal = np.array([d.nominal for d in dims], dtype=float)
    lsl = np.array([loop.lsl_mm for loop in net.loops], dtype=float)
    usl = np.array([loop.usl_mm for loop in net.loops], dtype=float)

    # ---- 极值法 / RSS（向量化，公式与单链引擎一致） ----
    mu = s_mat @ (nominal + mids)
    c0 = s_mat @ nominal
    band = np.abs(s_mat) @ halfs
    wc_lower, wc_upper = mu - band, mu + band
    cov_dim = net.corr * (sigmas[:, None] * sigmas[None, :])
    loop_cov = s_mat @ cov_dim @ s_mat.T
    var_l = np.maximum(np.diag(loop_cov), 0.0)
    sigma_l = np.sqrt(var_l)

    # ---- 蒙特卡洛：尺寸池共享抽样（每轮每尺寸只抽一次） ----
    n = n_samples or net.mc_samples
    used_seed = net.seed if seed is None else seed
    pool_nc = NormalizedChain(
        dimensions=dims, closure_lsl_mm=None, closure_usl_mm=None,
        mc_samples=n, seed=used_seed, corr=net.corr, name=net.name,
    )
    x = engine.sample_dimensions(
        pool_nc, n, used_seed,
        sigmas=sigmas, mids=mids, halfs=halfs, explicit_flags=explicit_flags,
    )
    closures = x @ s_mat.T                       # n × L

    # ---- 逐闭环结果 ----
    loop_results = []
    for li, loop in enumerate(net.loops):
        col = closures[:, li]
        q005, q995 = np.quantile(col, [0.005, 0.995])
        wc_in_spec = bool(wc_lower[li] >= lsl[li] and wc_upper[li] <= usl[li])
        loop_results.append({
            "loop_id": loop.id,
            "note": loop.note,
            "priority": loop.priority,
            "spec_mm": {
                "lower_limit": loop.lsl_mm,
                "upper_limit": loop.usl_mm,
                "original": loop.spec_original,
            },
            "path": loop.path,
            "worst_case": {
                "nominal_gap_mm": float(c0[li]),
                "mean_gap_mm": float(mu[li]),
                "lower_bound_mm": float(wc_lower[li]),
                "upper_bound_mm": float(wc_upper[li]),
                "total_tolerance_band_mm": float(2.0 * band[li]),
                "in_spec": wc_in_spec,
                "reject_probability": 0.0 if wc_in_spec else 1.0,
            },
            "rss": {
                "mean_gap_mm": float(mu[li]),
                "sigma_mm": float(sigma_l[li]),
                "z": Z_STAT,
                "lower_bound_mm": float(mu[li] - Z_STAT * sigma_l[li]),
                "upper_bound_mm": float(mu[li] + Z_STAT * sigma_l[li]),
                "reject_probability": engine._reject_normal(
                    float(mu[li]), float(sigma_l[li]),
                    loop.lsl_mm, loop.usl_mm),
            },
            "monte_carlo": {
                "mean_gap_mm": float(col.mean()),
                "sigma_mm": float(col.std(ddof=1)),
                "lower_bound_mm": float(q005),
                "upper_bound_mm": float(q995),
                "bounds_quantile": [0.005, 0.995],
                "min_sample_mm": float(col.min()),
                "max_sample_mm": float(col.max()),
                "reject_probability": engine._empirical_reject(
                    col, loop.lsl_mm, loop.usl_mm),
                "samples": int(n),
                "random_seed": int(used_seed),
            },
        })

    # ---- 联合指标 ----
    fail = (closures < lsl[None, :]) | (closures > usl[None, :])
    any_fail = fail.any(axis=1)
    joint_pass = float((~any_fail).mean())
    codes = fail.astype(np.int64) @ (1 << np.arange(n_loops, dtype=np.int64))
    uniq, counts = np.unique(codes[codes > 0], return_counts=True)
    order = sorted(zip(uniq.tolist(), counts.tolist()),
                   key=lambda t: (-t[1], t[0]))[:MAX_FAILURE_COMBINATIONS]
    failure_combinations = [
        {
            "loops": [net.loops[i].id for i in range(n_loops)
                      if (code >> i) & 1],
            "count": int(cnt),
            "rate": float(cnt / n),
        }
        for code, cnt in order
    ]
    pairwise = (fail.T.astype(float) @ fail.astype(float)) / n

    closures_c = closures - closures.mean(axis=0)
    mc_cov = (closures_c.T @ closures_c) / (n - 1)
    x_c = x - x.mean(axis=0)
    cov_xc = (x_c.T @ closures_c) / (n - 1)      # k × L
    var_c = np.maximum(np.diag(mc_cov), 0.0)

    joint = {
        "loop_order": [loop.id for loop in net.loops],
        "covariance_mm2": {
            "analytic": loop_cov.tolist(),
            "monte_carlo": mc_cov.tolist(),
        },
        "correlation": {
            "analytic": _corr_from_cov(loop_cov).tolist(),
            "monte_carlo": _corr_from_cov(mc_cov).tolist(),
        },
        "joint_pass_rate_monte_carlo": joint_pass,
        "joint_reject_rate_monte_carlo": float(any_fail.mean()),
        "all_loops_in_spec_worst_case": bool(
            np.all(wc_lower >= lsl) and np.all(wc_upper <= usl)),
        "pairwise_joint_reject_monte_carlo": pairwise.tolist(),
        "failure_combinations": failure_combinations,
        "formulas": [
            "Σ_闭环 = S·(DρD)·Sᵀ（解析，与单链 RSS 协方差公式同口径）",
            "MC 协方差为共享样本 C = X·Sᵀ 的样本协方差（同一批抽样）",
            "联合合格率 = #{全部闭环均在各自规格内} / N（同一轮样本）",
            "同时失效组合 = 同一轮中失效闭环集合的经验分布（按频次降序）",
            "两两联合超差矩阵 P(L_i 超差 ∧ L_j 超差)（MC 经验频率）",
        ],
    }

    # ---- 尺寸 × 闭环敏感度矩阵 ----
    entries = []
    for i, d in enumerate(dims):
        for li, loop in enumerate(net.loops):
            coef = float(s_mat[li, i])
            in_path = d.id in loop.coefficients
            wc_share = (
                abs(coef) * halfs[i] / band[li] if band[li] > 0 else 0.0)
            rss_contrib = coef * float((cov_dim @ s_mat[li])[i])
            rss_share = rss_contrib / var_l[li] if var_l[li] > 0 else 0.0
            slope = (
                float(cov_xc[i, li]) / var_c[li] if var_c[li] > 0 else 0.0)
            mc_share = (
                coef * float(cov_xc[i, li]) / var_c[li]
                if var_c[li] > 0 else 0.0)
            entries.append({
                "dimension_id": d.id,
                "loop_id": loop.id,
                "in_path": in_path,
                "coefficient": coef,
                "wc_tolerance_share": float(wc_share),
                "rss_variance_contribution_mm2": float(rss_contrib),
                "rss_variance_share": float(rss_share),
                "mc_regression_slope": float(slope),
                "mc_variance_share": float(mc_share),
            })
    sensitivity = {
        "dimension_order": [d.id for d in dims],
        "loop_order": [loop.id for loop in net.loops],
        "entries": entries,
        "note": "非路径尺寸敏感度恒为 0：其对闭环的影响已通过路径尺寸间的"
                "相关项计入闭环方差；各闭环 RSS 方差占比之和为 1",
    }

    return {
        "loops": loop_results,
        "joint": joint,
        "sensitivity_matrix": sensitivity,
        "normalized_pool": _pool_snapshot(net),
        "monte_carlo": {
            "samples": int(n),
            "seed": int(used_seed),
            "shared_sampling":
                "尺寸池每轮只抽样一次：X (n×k) 由单链引擎统一生成"
                "（Gaussian copula / 显式 σ 口径一致），各闭环样本 "
                "C = X·Sᵀ 共享同轮尺寸取值",
        },
        "traceability": {
            "unit_policy": "所有提交值按单位换算为 mm 参与计算；"
                          "原始名义值/偏差/标准差与单位原样保留",
            "mean_assumption": "制程中心假设位于公差带中点 N+m；"
                              "名义值 C0 仅由名义尺寸求和",
            "loop_coefficient": "s_i = 路径方向 sign × 尺寸测量方向 direction",
            "formulas": [
                "μ_L = Σ s_i·(N_i+m_i)；C0_L = Σ s_i·N_i",
                "WC 界: [μ_L − Σ|s_i|·T_i, μ_L + Σ|s_i|·T_i]",
                "RSS: σ_L² = s_Lᵀ(DρD)s_L，统计界 μ_L ± 3σ_L，"
                "超差率按正态近似（与单链一致）",
                "MC: 共享抽样 X 一次，C = X·Sᵀ；超差率为经验频率，"
                "报告界为样本 0.5%/99.5% 分位数",
            ],
            "correlation": {
                "matrix": net.corr.tolist(),
                "dimension_order": [d.id for d in dims],
                "definition": "输入 ρ 为尺寸值之间的 Pearson 相关系数；"
                              "RSS 协方差公式与蒙特卡洛抽样边缘采用同一定义",
            },
            "monte_carlo": {"samples": int(n), "seed": int(used_seed)},
        },
    }


def _pool_snapshot(net: NormalizedNetwork) -> list[dict]:
    """尺寸池规范化快照（mm），随版本冻结。"""
    return [
        {
            "dimension_id": d.id,
            "edge": [d.start, d.end],
            "measurement_direction": d.direction,
            "distribution": d.distribution,
            "sigma_explicit": d.sigma_explicit,
            "normalized_mm": {
                "nominal": d.nominal,
                "upper_deviation": d.upper_dev,
                "lower_deviation": d.lower_dev,
                "mid_shift": d.mid,
                "half_width": d.half_width,
                "sigma": d.sigma,
            },
            "original_representation": d.original,
        }
        for d in net.dimensions
    ]


def network_snapshot(net: NormalizedNetwork, source: dict) -> dict:
    """随版本冻结的完整快照：尺寸池 + 闭环 + 相关矩阵 + 种子 + 来源。"""
    return {
        "dimension_order": net.ids,
        "dimensions": _pool_snapshot(net),
        "loops": [
            {
                "loop_id": loop.id,
                "note": loop.note,
                "priority": loop.priority,
                "spec_mm": {
                    "lower_limit": loop.lsl_mm,
                    "upper_limit": loop.usl_mm,
                },
                "spec_original": loop.spec_original,
                "path": loop.path,
            }
            for loop in net.loops
        ],
        "correlation_matrix": net.corr.tolist(),
        "mc_samples": net.mc_samples,
        "random_seed": net.seed,
        "source": source,
    }


def network_from_snapshot(snap: dict) -> NormalizedNetwork:
    """从冻结快照重建规范化网络（方案搜索据此精确重放，不依赖基线链）。"""
    dims = [
        NormDimension(
            id=d["dimension_id"], sign=1,
            nominal=d["normalized_mm"]["nominal"],
            upper_dev=d["normalized_mm"]["upper_deviation"],
            lower_dev=d["normalized_mm"]["lower_deviation"],
            mid=d["normalized_mm"]["mid_shift"],
            half_width=d["normalized_mm"]["half_width"],
            sigma=d["normalized_mm"]["sigma"],
            sigma_explicit=bool(d["sigma_explicit"]),
            distribution=d["distribution"],
            start=d["edge"][0], end=d["edge"][1],
            direction=int(d["measurement_direction"]),
            unit=d["original_representation"]["unit"],
            original=d["original_representation"],
        )
        for d in snap["dimensions"]
    ]
    loops = [
        ResolvedLoop(
            id=loop["loop_id"], note=loop["note"],
            priority=int(loop["priority"]),
            lsl_mm=loop["spec_mm"]["lower_limit"],
            usl_mm=loop["spec_mm"]["upper_limit"],
            coefficients={p["dimension_id"]: int(p["coefficient"])
                          for p in loop["path"]},
            path=loop["path"],
            spec_original=loop["spec_original"],
        )
        for loop in snap["loops"]
    ]
    return NormalizedNetwork(
        dimensions=dims, loops=loops,
        corr=np.array(snap["correlation_matrix"], dtype=float),
        mc_samples=int(snap["mc_samples"]),
        seed=int(snap["random_seed"]),
    )


# ------------------------------------------------------------- 方案分支搜索

def _scenario_sigma(d: NormDimension, level: float, new_half: float,
                    request) -> tuple[float, bool]:
    """方案候选的 σ 与显式标志（与单链批量调整同一口径）。"""
    if d.distribution == "normal":
        sigma = d.sigma * (level if request.scale_normal_sigma else 1.0)
        flag = True
    elif d.sigma_explicit:
        sigma, flag = d.sigma, True
    else:
        sigma = engine._theoretical_sigma(d.distribution, new_half)
        flag = False
    if request.std_dev_scale is not None:
        sigma *= request.std_dev_scale
        flag = True
    return sigma, flag


def search_network_scenarios(net: NormalizedNetwork, request) -> dict:
    """锁定尺寸 + 批量调整公差/标准差的方案搜索（结果按达标/余量/成本排序）。"""
    dims = net.dimensions
    k = len(dims)
    n_loops = len(net.loops)
    ids = [d.id for d in dims]
    id_set = set(ids)
    locked = set(request.locked_dimensions)
    unknown = sorted(locked - id_set)
    if unknown:
        raise NetworkError(f"锁定尺寸不在网络尺寸池中: {unknown}")
    unknown_cost = sorted(set(request.tightening_cost) - id_set)
    if unknown_cost:
        raise NetworkError(
            f"单位收紧成本引用了尺寸池外的尺寸: {unknown_cost}")

    levels = sorted(set(request.scale_levels))
    costs = np.array([
        request.tightening_cost.get(d.id, request.default_cost)
        for d in dims
    ])
    halfs0 = np.array([d.half_width for d in dims], dtype=float)
    sigmas0 = np.array([d.sigma for d in dims], dtype=float)
    mids = np.array([d.mid for d in dims], dtype=float)
    explicit0 = [d.sigma_explicit for d in dims]
    nominal = np.array([d.nominal for d in dims], dtype=float)
    s_mat = net.coefficient_matrix()
    lsl = np.array([loop.lsl_mm for loop in net.loops], dtype=float)
    usl = np.array([loop.usl_mm for loop in net.loops], dtype=float)
    max_priority = max(loop.priority for loop in net.loops)
    top_loops = [li for li, loop in enumerate(net.loops)
                 if loop.priority == max_priority]

    # 每个尺寸的候选: (level, half, sigma, explicit, cost)
    options: list[list[tuple]] = []
    for i, d in enumerate(dims):
        if d.id in locked:
            options.append([(1.0, float(halfs0[i]), float(sigmas0[i]),
                             bool(explicit0[i]), 0.0)])
            continue
        col = []
        for lv in levels:
            h = halfs0[i] * lv
            s, fl = _scenario_sigma(d, lv, h, request)
            col.append((lv, h, s, fl, costs[i] * (halfs0[i] - h)))
        options.append(col)

    center = nominal + mids

    def analytic(halfs: np.ndarray, sigmas: np.ndarray):
        """返回 (各闭环 RSS 超差率, 各闭环极值法余量, 联合代理超差率)。"""
        mu = s_mat @ center
        band = np.abs(s_mat) @ halfs
        w = s_mat * sigmas[None, :]
        var_l = np.maximum(
            np.einsum("li,ij,lj->l", w, net.corr, w), 0.0)
        sig_l = np.sqrt(var_l)
        rej = np.array([
            engine._reject_normal(float(mu[li]), float(sig_l[li]),
                                  float(lsl[li]), float(usl[li]))
            for li in range(n_loops)
        ])
        margins = np.minimum(usl - (mu + band), (mu - band) - lsl)
        return rej, margins, float(rej.sum())

    # ---------------- beam search（联合超差率代理 → 成本） ----------------
    beam_width = max(20, min(400, request.max_evaluations // max(k, 1) + 10))
    beam = [{"levels": [], "halfs": [], "sigmas": [], "flags": [],
             "cost": 0.0}]
    for i in range(k):
        expanded = []
        for st in beam:
            for lv, h, s, fl, c in options[i]:
                expanded.append({
                    "levels": st["levels"] + [lv],
                    "halfs": st["halfs"] + [h],
                    "sigmas": st["sigmas"] + [s],
                    "flags": st["flags"] + [fl],
                    "cost": st["cost"] + c,
                })
        # 部分状态用基准值补齐未展开尺寸后评估代理超差率
        m = len(expanded)
        ph = np.tile(halfs0, (m, 1))
        ps = np.tile(sigmas0, (m, 1))
        for row, st in enumerate(expanded):
            ph[row, :len(st["halfs"])] = st["halfs"]
            ps[row, :len(st["sigmas"])] = st["sigmas"]
        mu = s_mat @ center
        band = (np.abs(s_mat) @ ph.T)                    # L × m
        w = s_mat[None, :, :] * ps[:, None, :]           # m × L × k
        var = np.einsum("mli,ij,mlj->ml", w, net.corr, w)
        sig_l = np.sqrt(np.maximum(var, 0.0))
        proxy = np.zeros(m)
        for li in range(n_loops):
            sig_li = sig_l[:, li]
            p = np.zeros(m)
            nz = sig_li > 0
            p[nz] = (
                engine.normal_cdf((lsl[li] - mu[li]) / sig_li[nz])
                + 1.0 - engine.normal_cdf((usl[li] - mu[li]) / sig_li[nz])
            )
            zmask = ~nz
            if zmask.any():
                p[zmask] = ((mu[li] < lsl[li]) or (mu[li] > usl[li]))
            proxy += p
        order = sorted(range(m), key=lambda r: (proxy[r],
                                                expanded[r]["cost"]))
        beam = [expanded[r] for r in order[:beam_width]]

    # ---------------- 解析排序，取前列进入 MC 复核 ----------------
    baseline_state = {
        "levels": [1.0] * k,
        "halfs": [float(h) for h in halfs0],
        "sigmas": [float(s) for s in sigmas0],
        "flags": [bool(f) for f in explicit0],
        "cost": 0.0,
        "is_baseline": True,
    }
    # 去重键含 σ：std_dev_scale 生效时全 1.0 比例的束状态与真基线不同
    def _state_key(st: dict) -> tuple:
        return (tuple(round(v, 12) for v in st["levels"])
                + tuple(round(v, 12) for v in st["sigmas"]))

    base_key = _state_key(baseline_state)
    states = [baseline_state]
    seen_keys = {base_key}
    for st in beam:
        key = _state_key(st)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        st["is_baseline"] = False
        states.append(st)

    scored = []
    for st in states:
        halfs = np.array(st["halfs"])
        sigmas = np.array(st["sigmas"])
        rej, margins, proxy = analytic(halfs, sigmas)
        priority_margin = float(min(margins[li] for li in top_loops))
        scored.append({
            "state": st, "halfs": halfs, "sigmas": sigmas,
            "rss_reject": rej, "margins": margins,
            "proxy_joint": proxy, "priority_margin": priority_margin,
            "pass_proxy": proxy <= request.target_reject_rate,
        })
    scored.sort(key=lambda item: (
        0 if item["pass_proxy"] else 1,
        -item["priority_margin"],
        item["state"]["cost"],
    ))
    mc_states = scored[:TOP_CANDIDATES_MC]
    if not any(item["state"].get("is_baseline") for item in mc_states):
        base_scored = next(item for item in scored
                           if item["state"].get("is_baseline"))
        mc_states.append(base_scored)

    # ---------------- 共享抽样 MC 复核 ----------------
    seed = net.seed if request.random_seed is None else request.random_seed
    verified = []
    for item in mc_states:
        st = item["state"]
        is_base = bool(st.get("is_baseline"))
        if is_base:
            # 零成本现状候选必须与冻结网络版本结果完全一致：用网络自身的
            # 种子/样本数复核（而非本次搜索请求的种子/样本数），否则稀有
            # 尾部超差率会因抽样配置不同而与基线结果不一致。
            res = compute_network(
                net, sigmas=item["sigmas"], mids=mids, halfs=item["halfs"],
                explicit_flags=st["flags"],
                n_samples=net.mc_samples, seed=net.seed)
        else:
            res = compute_network(
                net, sigmas=item["sigmas"], mids=mids, halfs=item["halfs"],
                explicit_flags=st["flags"],
                n_samples=request.mc_samples, seed=seed)
        joint_reject = res["joint"]["joint_reject_rate_monte_carlo"]
        verified.append({
            "item": item,
            "joint_reject": joint_reject,
            "joint_pass": res["joint"]["joint_pass_rate_monte_carlo"],
            "loops": res["loops"],
            "all_pass": joint_reject <= request.target_reject_rate,
            "monte_carlo": {"samples": (net.mc_samples if is_base
                                        else request.mc_samples),
                            "seed": (net.seed if is_base else seed)},
        })

    verified.sort(key=lambda v: (
        0 if v["all_pass"] else 1,
        -v["item"]["priority_margin"],
        v["item"]["state"]["cost"],
    ))

    def _candidate(rank: int, v: dict) -> dict:
        st = v["item"]["state"]
        is_base = bool(st.get("is_baseline"))
        return {
            "rank": rank,
            "is_current_baseline": is_base,
            "levels": {d.id: float(lv)
                       for d, lv in zip(dims, st["levels"])},
            "locked_dimensions": sorted(locked),
            "total_cost": round(st["cost"], 9),
            "all_loops_pass": v["all_pass"],
            "joint_reject_rate_monte_carlo": v["joint_reject"],
            "joint_pass_rate_monte_carlo": v["joint_pass"],
            "priority_margin_mm": v["item"]["priority_margin"],
            "min_margin_all_loops_mm": float(min(v["item"]["margins"])),
            "cost_breakdown": [
                {
                    "dimension_id": d.id,
                    "tolerance_scale": float(st["levels"][i]),
                    "old_half_width_mm": float(halfs0[i]),
                    "new_half_width_mm": float(st["halfs"][i]),
                    "tightening_mm": float(halfs0[i] - st["halfs"][i]),
                    "new_sigma_mm": float(st["sigmas"][i]),
                    "sigma_explicit": bool(st["flags"][i]),
                    "locked": d.id in locked,
                    "unit_cost_per_mm": float(costs[i]),
                    "cost": round(costs[i] * (halfs0[i] - st["halfs"][i]), 9),
                }
                for i, d in enumerate(dims)
            ],
            "loops": [
                {
                    "loop_id": loop_res["loop_id"],
                    "priority": loop_res["priority"],
                    "worst_case": {
                        "lower_bound_mm":
                            loop_res["worst_case"]["lower_bound_mm"],
                        "upper_bound_mm":
                            loop_res["worst_case"]["upper_bound_mm"],
                        "in_spec": loop_res["worst_case"]["in_spec"],
                    },
                    "rss": {
                        "sigma_mm": loop_res["rss"]["sigma_mm"],
                        "reject_probability":
                            loop_res["rss"]["reject_probability"],
                    },
                    "monte_carlo": {
                        "reject_probability":
                            loop_res["monte_carlo"]["reject_probability"],
                    },
                }
                for loop_res in v["loops"]
            ],
            "verification": {
                "method": "monte_carlo_shared_sampling",
                "samples": request.mc_samples,
                "random_seed": seed,
                "criterion": "联合超差率 ≤ "
                             f"{request.target_reject_rate} 记为全部闭环达标",
            },
        }

    ordered = list(enumerate(verified, start=1))
    ranked = [
        _candidate(rank, v) for rank, v in ordered
    ]
    # 零成本现状候选必须始终出现在候选列表中（即使排序后落在截断窗口外）：
    # 否则调用方无法对比「不收紧」方案。先取前 max_candidates，缺失则补入
    # 现状候选（标注其真实排序名次）。
    cap = request.max_candidates
    selected = ranked[:cap]
    base_v = next(v for v in verified
                  if v["item"]["state"].get("is_baseline"))
    base_rank = next(rank for rank, v in ordered
                     if v["item"]["state"].get("is_baseline"))
    if not any(c["is_current_baseline"] for c in selected):
        base_candidate = _candidate(base_rank, base_v)
        base_candidate["included_beyond_limit"] = True
        selected = selected + [base_candidate]
    candidates = selected
    baseline_v = base_v
    baseline = {
        "joint_reject_rate_monte_carlo": baseline_v["joint_reject"],
        "joint_pass_rate_monte_carlo": baseline_v["joint_pass"],
        "all_loops_pass": baseline_v["all_pass"],
        "priority_margin_mm": baseline_v["item"]["priority_margin"],
        "total_cost": 0.0,
    }

    return {
        "ranking_note":
            "排序键：全部闭环达标（共享抽样 MC 联合超差率 ≤ 目标）→ "
            "最高优先级闭环的最小极值法余量（大者优先）→ 总收紧成本（升序）",
        "target_reject_rate": request.target_reject_rate,
        "highest_priority": max_priority,
        "highest_priority_loops": [net.loops[li].id for li in top_loops],
        "locked_dimensions": sorted(locked),
        "cost_model": {
            "formula": "cost_i = c_i · (T_i − T_i·level_i)，T_i 为版本当前"
                       "公差带半宽(mm)；level>1 为放宽（成本为负，表示节省）",
            "unit_cost_per_mm": {d.id: float(costs[i])
                                 for i, d in enumerate(dims)},
            "std_dev_scale": request.std_dev_scale,
            "scale_normal_sigma": request.scale_normal_sigma,
        },
        "sigma_policy_note":
            "均匀/三角未显式给 σ 时随新公差带取理论值；正态默认保留用户 σ"
            "（scale_normal_sigma=true 时随公差带等比缩放）；std_dev_scale "
            "给定则未锁定尺寸 σ 统一再乘该系数并视为显式 σ；锁定尺寸不调整",
        "baseline": baseline,
        "candidates": candidates,
        "search": {
            "beam_width": beam_width,
            "states_after_dedup": len(states),
            "mc_verified": len(verified),
            "mc_samples": request.mc_samples,
            "random_seed": seed,
        },
    }
