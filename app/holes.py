"""孔系装配分析：一对零件上的孔 / 销 / 螺栓在基准框架下的容差叠加。

物理模型（2D 刚性件）
=====================
每个匹配位 i 给出两侧名义中心 pA_i、pB_i（各自基准框架坐标）与要素尺寸。
装配位姿 (tx, ty, θ) 把 B 相对 A 做平移与小转角，匹配位的相对错位：

    δ_i = t + R(θ)·pB_i − pA_i

两类连接件（GD&T 紧固件公式）：

* 固定连接件（孔 + 销/固定螺栓）：径向允许错位
      a_i = (D_孔,min − d_销,max)/2 − (t_A + t_B)/2
* 浮动连接件（两个孔 + 穿过两孔的自由螺栓）：
      a_i = (D_A,min + D_B,min)/2 − d_螺栓,max − (t_A + t_B)/2

最坏边界（virtual condition）
-----------------------------
实体状态补偿公差（bonus）与基准偏移在 MMC/LMC 边界尺寸上均为 0，故最坏边界
退化为按 VC 尺寸的圆盘可行性：对所有匹配位同时满足

      |t + θ·J·pB_i − Δp_i| ≤ a_i − b_i

（Δp_i = pB_i − pA_i；b_i 为两侧 rz 基准角度公差在该位的径向折算）。
即在 3 个位姿自由度上求一族圆盘的公共交集。θ 在声明网格上扫描，每个 θ 解
2D 圆盘交集（凸可行 + 切比雪夫中心，子梯度迭代），给出可行位姿范围。

蒙特卡洛（固定种子，复用全局分布模型）
--------------------------------------
逐要素按 normal / uniform / triangular 抽样直径偏差（单位统一 mm，口径与
基线尺寸链一致）；实际尺寸偏离实体边界时产生补偿：

* 孔径 bonus（MMC）= max(0, D_实际 − D_min)；销 bonus = max(0, d_max − d_实际)
* LMC 对称取对侧偏离；RFS 无补偿
* 基准偏移（尺寸基准 MMC/LMC）= 基准要素相对边界尺寸的 diametral 间隙，
  按位置度基准引用计入该要素有效位置公差（GD&T 紧固件表惯例：每孔可独立
  占用基准模拟器间隙）
* rz 尺寸基准：间隙/(2·力臂) 给出模式转角；边线 rz 基准按角度公差抽样；
  公共模式转角与装配 θ 优化后仍可能残留（两侧力臂不同）

要素位置误差按两轴独立抽样，有效半宽 = (t + bonus + 基准偏移)/2；
normal 默认 σ = t/6（公差带直径 t、±t/2 为 ±3σ 包络）。
每个样本求解能容纳全部连接件的最优位姿，统计装配成功率、位姿分布、
最先干涉匹配位与各匹配位余量。

快照冻结：规范化 mm 输入、基准框架、抽样配置（SeedSequence 子流）与种子
随版本写入 SQLite，重复读取/计算精确复现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .engine import _standard_quantile, _theoretical_sigma
from .hole_schemas import (
    DatumKind,
    FeatureKind,
    HolePatternCreate,
    MaterialCondition,
)
from .units import to_mm

# SeedSequence 域标签（"hole"），与 thermal 等模块的子流空间互不相关
_SEED_DOMAIN = 0x686F6C65

# 求解容差与迭代预算
_FEAS_TOL = 1e-7       # mm
_POLISH_ITERS = 60     # 单样本 3D 子梯度迭代
_WC_POLISH_ITERS = 80  # 最坏边界每个 θ 的 2D 迭代


class HoleError(ValueError):
    """创建期 / 求解期可定位错误（端点转 422）。"""


# ------------------------------------------------------------- 规范化模型

@dataclass
class _Datum:
    label: str
    order: int
    kind: str
    constrains: tuple[str, ...]
    material_condition: str
    # size
    nom: float = 0.0
    es: float = 0.0
    ei: float = 0.0
    distribution: str = "uniform"
    sigma: float | None = None
    lever: float | None = None
    # edge
    nx: float = 0.0
    ny: float = 1.0
    offset: float = 0.0
    ang_half_rad: float = 0.0


@dataclass
class _Feature:
    mate_id: str
    side: str
    kind: str                       # hole / pin
    x: float
    y: float
    nom: float
    es: float
    ei: float
    tol: float                      # 位置度公差带直径
    mc: str                         # 实体条件
    distribution: str
    d_sigma: float | None
    p_sigma: float | None
    refs: tuple[str, ...]


@dataclass
class HoleModel:
    name: str
    unit: str
    factor: float                   # 提交单位 -> mm
    mate_ids: list[str]
    features_a: list[_Feature]
    features_b: list[_Feature]
    # 每个匹配位：kind='fixed'（外要素在 ext_side）或 'float'（两孔+螺栓）
    mate_kinds: list[str]
    ext_sides: list[str | None]
    bolt_nom: list[float | None]
    bolt_es: list[float | None]
    bolt_ei: list[float | None]
    frames: dict[str, list[_Datum]]
    mc_samples: int
    seed: int
    theta_max_rad: float
    theta_grid: int
    mc_theta_points: int
    request: Any = None

    # ---- 便捷数组
    @property
    def n(self) -> int:
        return len(self.mate_ids)


def _build_datum(spec, factor: float) -> _Datum:
    d = _Datum(
        label=spec.label, order=spec.order, kind=spec.kind.value,
        constrains=tuple(spec.constrains),
        material_condition=spec.material_condition.value,
        distribution=spec.distribution.value,
        sigma=spec.diameter_std_dev,
    )
    if spec.kind == DatumKind.SIZE:
        d.nom = spec.nominal_diameter * factor
        d.es = spec.diameter_upper_deviation * factor
        d.ei = spec.diameter_lower_deviation * factor
        d.lever = (spec.lever_radius * factor
                   if spec.lever_radius is not None else None)
    else:
        norm = math.hypot(spec.normal_x, spec.normal_y)
        d.nx, d.ny = spec.normal_x / norm, spec.normal_y / norm
        d.offset = spec.line_offset * factor
        if spec.angular_half_range_deg is not None:
            d.ang_half_rad = math.radians(spec.angular_half_range_deg)
    return d


def build_model(payload: HolePatternCreate) -> HoleModel:
    """Pydantic 校验之后做单位归一化与跨要素结构校验，返回 mm 模型。"""
    factor = to_mm(1.0, payload.unit.value)
    fa, fb, kinds, ext_sides = [], [], [], []
    bolt_nom: list[float | None] = []
    bolt_es: list[float | None] = []
    bolt_ei: list[float | None] = []

    def feat(f, mate_id: str, side: str) -> _Feature:
        return _Feature(
            mate_id=mate_id, side=side, kind=f.kind.value,
            x=f.x * factor, y=f.y * factor,
            nom=f.nominal_diameter * factor,
            es=f.diameter_upper_deviation * factor,
            ei=f.diameter_lower_deviation * factor,
            tol=f.position_tolerance * factor,
            mc=f.material_condition.value,
            distribution=f.distribution.value,
            d_sigma=None if f.diameter_std_dev is None
            else f.diameter_std_dev * factor,
            p_sigma=None if f.position_std_dev is None
            else f.position_std_dev * factor,
            refs=tuple(f.position_datum_refs),
        )

    for m in payload.mates:
        fa.append(feat(m.feature_a, m.id, "A"))
        a_hole = m.feature_a.kind == FeatureKind.HOLE
        bolt = m.bolt_diameter_upper is not None
        if m.feature_b is None:
            # Pydantic 已保证：无 feature_b 时必有螺栓极限；固定螺栓位于 B，
            # 几何上等价于 B 侧外要素（销），无独立位置（与 A 孔名义同位）。
            fb.append(_Feature(
                mate_id=m.id, side="B", kind=FeatureKind.PIN.value,
                x=m.feature_a.x * factor, y=m.feature_a.y * factor,
                nom=0.5 * (m.bolt_diameter_upper + m.bolt_diameter_lower)
                * factor,
                es=(m.bolt_diameter_upper
                    - 0.5 * (m.bolt_diameter_upper + m.bolt_diameter_lower))
                * factor,
                ei=(0.5 * (m.bolt_diameter_upper + m.bolt_diameter_lower)
                    - m.bolt_diameter_lower) * factor,
                tol=0.0, mc=MaterialCondition.RFS.value,
                distribution="uniform", d_sigma=None, p_sigma=None,
                refs=(),
            ))
            if not a_hole:
                raise HoleError(
                    f"匹配位 {m.id}: A 侧为外要素且 B 侧仅给固定螺栓直径，"
                    "不存在可容纳连接件的内要素")
            kinds.append("fixed")
            ext_sides.append("B")
            bolt_nom.append(None)
            bolt_es.append(None)
            bolt_ei.append(None)
            continue

        fb.append(feat(m.feature_b, m.id, "B"))
        b_hole = m.feature_b.kind == FeatureKind.HOLE
        if not a_hole and not b_hole:
            raise HoleError(
                f"匹配位 {m.id}: 两侧均为外要素（销/销），无内要素容纳连接件")
        if a_hole and b_hole:
            if not bolt:
                raise HoleError(
                    f"匹配位 {m.id}: 两侧均为孔但未给螺栓直径极限"
                    "（浮动连接件必须声明 bolt_diameter_upper/lower）")
            kinds.append("float")
            ext_sides.append(None)
            mid = 0.5 * (m.bolt_diameter_upper + m.bolt_diameter_lower)
            bolt_nom.append(mid * factor)
            bolt_es.append((m.bolt_diameter_upper - mid) * factor)
            bolt_ei.append((mid - m.bolt_diameter_lower) * factor)
        else:
            kinds.append("fixed")
            ext_sides.append("B" if b_hole is False else "A")
            bolt_nom.append(None)
            bolt_es.append(None)
            bolt_ei.append(None)

    frames = {
        "A": [_build_datum(d, factor) for d in payload.frame_a.datums],
        "B": [_build_datum(d, factor) for d in payload.frame_b.datums],
    }
    return HoleModel(
        name=payload.name, unit=payload.unit.value, factor=factor,
        mate_ids=[m.id for m in payload.mates],
        features_a=fa, features_b=fb, mate_kinds=kinds,
        ext_sides=ext_sides, bolt_nom=bolt_nom, bolt_es=bolt_es,
        bolt_ei=bolt_ei, frames=frames,
        mc_samples=payload.mc_samples, seed=payload.random_seed,
        theta_max_rad=math.radians(payload.theta_search_deg),
        theta_grid=payload.theta_grid_points,
        mc_theta_points=payload.mc_theta_points,
        request=payload,
    )


# ------------------------------------------------------------- 抽样

def _sample_diameter(rng, nom: float, es: float, ei: float,
                     distribution: str, sigma: float | None,
                     n: int) -> np.ndarray:
    """名义+偏差的直径样本（mm）；口径与 engine.sample_dimensions 一致。"""
    mid, half = 0.5 * (es + ei), 0.5 * (es - ei)
    if distribution == "normal":
        s = sigma if sigma is not None else _theoretical_sigma(distribution, half)
        return nom + mid + s * rng.standard_normal(n)
    u = rng.random(n)
    return nom + mid + _standard_quantile(u, distribution) * half


def _position_components(rng, half, distribution: str,
                         sigma: float | None, n: int) -> np.ndarray:
    """2D 位置误差分量样本 (n,2)。

    half 可为标量（固定单轴半宽 =有效位置度/2）或逐样本数组
    （uniform/triangular 的 bonus、基准偏移按实际样本缩放）；
    normal 的 σ 固定（缺省 = 名义公差带 t/6），不随单样本 bonus 变化。
    """
    half_arr = np.asarray(half, dtype=float)
    if distribution == "normal":
        nominal = half_arr.mean() if half_arr.ndim else float(half_arr)
        s = sigma if sigma is not None else nominal / 3.0
        if s <= 0.0:
            return np.zeros((n, 2))
        return s * rng.standard_normal((n, 2))
    if half_arr.ndim == 0:
        if half_arr <= 0.0:
            return np.zeros((n, 2))
        scale = half_arr
    else:
        scale = half_arr[:, None]
    u = rng.random((n, 2))
    return _standard_quantile(u, distribution) * scale


@dataclass
class _DatumDraws:
    shifts: dict[str, np.ndarray]   # label -> diametral 间隙（n,）
    rho: np.ndarray                 # 模式转角（n,）弧度


def _sample_datums(rng_d, rng_r, frame: list[_Datum], n: int) -> _DatumDraws:
    shifts: dict[str, np.ndarray] = {}
    rho = np.zeros(n)
    for d in frame:
        if d.kind == DatumKind.SIZE.value:
            actual = _sample_diameter(
                rng_d, d.nom, d.es, d.ei, d.distribution, d.sigma, n)
            # 内要素（基准孔）：MMC 尺寸 = 最小孔，LMC = 最大孔
            if d.material_condition == MaterialCondition.MMC.value:
                gap = np.maximum(0.0, actual - (d.nom + d.ei))
            elif d.material_condition == MaterialCondition.LMC.value:
                gap = np.maximum(0.0, (d.nom + d.es) - actual)
            else:
                gap = np.zeros(n)
            shifts[d.label] = gap
            if "rz" in d.constrains:
                # 切向单边间隙 = diametral/2，转角 ≤ 间隙/(2·力臂)
                rho = rho + (gap / (2.0 * d.lever)) * (
                    2.0 * rng_r.random(n) - 1.0)
        else:  # edge：RFS 无尺寸漂移
            shifts[d.label] = np.zeros(n)
            if "rz" in d.constrains and d.ang_half_rad > 0:
                rho = rho + d.ang_half_rad * (2.0 * rng_r.random(n) - 1.0)
    return _DatumDraws(shifts=shifts, rho=rho)


def _bonus(actual: np.ndarray, f: _Feature) -> np.ndarray:
    """实体状态补偿公差（diametral）。"""
    if f.mc == MaterialCondition.MMC.value:
        boundary_lo = f.nom + f.ei   # 孔 MMC=最小；销 MMC=最大
        boundary_hi = f.nom + f.es
        if f.kind == FeatureKind.HOLE.value:
            return np.maximum(0.0, actual - boundary_lo)
        return np.maximum(0.0, boundary_hi - actual)
    if f.mc == MaterialCondition.LMC.value:
        if f.kind == FeatureKind.HOLE.value:
            return np.maximum(0.0, (f.nom + f.es) - actual)
        return np.maximum(0.0, actual - (f.nom + f.ei))
    return np.zeros_like(actual)


def _rng_family(seed: int, family: int):
    from numpy.random import SeedSequence, default_rng

    ss = SeedSequence(seed, spawn_key=(_SEED_DOMAIN, family))
    return default_rng(ss)


def sample_realization(model: HoleModel, n: int, seed: int | None = None,
                       ) -> dict[str, Any]:
    """生成 n 个装配样本的全部偏差量（mm / rad），固定种子可精确复现。

    子流族（family）编号固定：0/1 两侧直径，2 螺栓，3/4 位置，
    5/6 基准尺寸，7/8 基准转角。
    """
    seed = model.seed if seed is None else seed
    sides = {"A": model.features_a, "B": model.features_b}
    diameters: dict[str, np.ndarray] = {}
    positions: dict[str, list[np.ndarray]] = {"A": [], "B": []}

    for si, side in enumerate(("A", "B")):
        rng_d = _rng_family(seed, si)
        feats = sides[side]
        dcols = np.empty((n, len(feats)))
        for j, f in enumerate(feats):
            dcols[:, j] = _sample_diameter(
                rng_d, f.nom, f.es, f.ei, f.distribution, f.d_sigma, n)
            diameters[(side, j)] = dcols[:, j]
        datum_draws = _sample_datums(
            _rng_family(seed, 5 + si), _rng_family(seed, 7 + si),
            model.frames[side], n)

        rng_p = _rng_family(seed, 3 + si)
        for j, f in enumerate(feats):
            bonus = _bonus(dcols[:, j], f)
            shift = np.zeros(n)
            for label in f.refs:
                shift = shift + datum_draws.shifts.get(label, np.zeros(n))
            half = 0.5 * (f.tol + bonus + shift)
            # normal 位置 σ 固定（缺省 t/6，不含逐样本 bonus）；bonus 单独加性
            if f.distribution == "normal":
                psigma = f.p_sigma if f.p_sigma is not None else f.tol / 6.0
                comp = _position_components(
                    rng_p, f.tol / 2.0, "normal", psigma, n)
                # 实体补偿与基准偏移在 normal 口径下也按实际样本附加（均匀有界）
                extra_half = 0.5 * (bonus + shift)
                u = rng_p.random((n, 2))
                comp = comp + (2.0 * u - 1.0) * extra_half[:, None]
            else:
                comp = _position_components(
                    rng_p, half, f.distribution, f.p_sigma, n)
            # 公共模式转角（rz 基准）绕名义中心旋转
            if model.frames[side]:
                c, s = np.cos(datum_draws.rho), np.sin(datum_draws.rho)
                rx = c * f.x - s * f.y - f.x
                ry = s * f.x + c * f.y - f.y
                comp = comp + np.column_stack([rx, ry])
            positions[side].append(comp)

    bolt = None
    if any(k == "float" for k in model.mate_kinds):
        rng_b = _rng_family(seed, 2)
        bolt = np.empty((n, model.n))
        for i in range(model.n):
            if model.mate_kinds[i] != "float":
                continue
            bolt[:, i] = _sample_diameter(
                rng_b, model.bolt_nom[i], model.bolt_es[i], model.bolt_ei[i],
                "uniform", None, n)
    return {"diameters": diameters, "positions": positions, "bolt": bolt}


# ------------------------------------------------------------- 位姿求解
#
# 位姿可行性是凸问题：最小化 g(t,θ)=max_i(|R(θ)pB_i+t−C_i|−r_i)，g*≤0
# 即可容纳全部连接件，-g* 为最差匹配位余量。采用活动约束的确定性子梯度法
# （步长 γ0/√t，保留历史最优点）：MC 跨样本向量化，WC 逐 θ 长迭代抛光。

def _rotate(th, PB):
    ct, st = math.cos(th), math.sin(th)
    return ct * PB[:, 0] - st * PB[:, 1], st * PB[:, 0] + ct * PB[:, 1]


def _g_eval(th, tx, ty, PA, PB, T, rad):
    """目标圆盘中心 T 为 A 侧实际中心：|R(θ)pB+t−T| ≤ r。"""
    rx, ry = _rotate(float(th), PB)
    dx = rx + float(tx) - T[:, 0]
    dy = ry + float(ty) - T[:, 1]
    gv = np.hypot(dx, dy) - rad
    k = int(np.argmax(gv))
    return float(gv[k]), k, gv


def _ls_start_scalar(PA, PB, T, theta_max: float) -> np.ndarray:
    """名义刚体最小二乘热启动 (tx,ty,θ)（小角线性化法方程）。"""
    n = len(PB)
    sx, sy = PB[:, 0].sum(), PB[:, 1].sum()
    sxx = float((PB[:, 0] ** 2).sum())
    syy = float((PB[:, 1] ** 2).sum())
    A = np.array([[n, 0.0, -sy],
                  [0.0, n, sx],
                  [-sy, sx, sxx + syy]])
    b = np.array([float((T[:, 0] - PB[:, 0]).sum()),
                  float((T[:, 1] - PB[:, 1]).sum()),
                  float((-PB[:, 1] * (T[:, 0] - PB[:, 0])
                         + PB[:, 0] * (T[:, 1] - PB[:, 1])).sum())])
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.array([0.0, 0.0, 0.0])
        x[:2] = (T - PB).mean(axis=0)
    x[2] = min(theta_max, max(-theta_max, x[2]))
    return x


def _subgrad_scalar(x, PA, PB, T, rad, theta_max: float, iters: int,
                    fixed_theta: float | None = None,
                    gamma0: float | None = None):
    """单系统子梯度（2D 固定 θ 或 3D），返回 (x, g*, 逐位 g 值)。"""
    x = x.copy()
    if fixed_theta is not None:
        x[2] = fixed_theta
    best_x, best_g, best_gv = x.copy(), np.inf, None
    g0, _, _ = _g_eval(x[2], x[0], x[1], PA, PB, T, rad)
    scale = max(1.0, float(np.linalg.norm(PB, axis=1).mean()))
    gamma0 = gamma0 if gamma0 is not None else max(abs(g0), 0.05)
    for t in range(iters):
        th = x[2] if fixed_theta is None else fixed_theta
        g, k, gv = _g_eval(th, x[0], x[1], PA, PB, T, rad)
        if g < best_g:
            best_g, best_x, best_gv = g, x.copy(), gv.copy()
        rx, ry = _rotate(th, PB)
        dx = rx[k] + x[0] - T[k, 0]
        dy = ry[k] + x[1] - T[k, 1]
        den = math.hypot(dx, dy)
        if den < 1e-12:
            break
        ux, uy = dx / den, dy / den
        gamma = gamma0 / math.sqrt(t + 1)
        if fixed_theta is not None:
            x[0] -= gamma * ux
            x[1] -= gamma * uy
        else:
            ct, st = math.cos(th), math.sin(th)
            gth = ux * (-st * PB[k, 0] - ct * PB[k, 1]) \
                + uy * (ct * PB[k, 0] - st * PB[k, 1])
            x[0] -= gamma * ux
            x[1] -= gamma * uy
            x[2] -= gamma * gth / scale
            x[2] = min(theta_max, max(-theta_max, x[2]))
    g, _, gv = _g_eval(best_x[2] if fixed_theta is None else fixed_theta,
                       best_x[0], best_x[1], PA, PB, T, rad)
    if g < best_g:
        best_g, best_gv = g, gv
    return best_x, best_g, best_gv


def solve_pose(PA, PB, T, rad, theta_max: float,
               theta_scan: int | None = None,
               extra_radius: np.ndarray | None = None
               ) -> dict[str, Any]:
    """求容纳全部连接件的最优位姿（θ 网格扫描 + 3D 抛光）。

    PA/PB：(n,2) 名义中心；T：(n,2) 目标中心（A 侧实际中心）；rad：(n,) 径向余量；
    extra_radius：(n,) 孔位修正上限（搜索整改时按自由定向修正等效扩径）。
    """
    r = rad if extra_radius is None else rad + extra_radius
    x_ls = _ls_start_scalar(PA, PB, T, theta_max)
    if theta_scan:
        thetas = np.linspace(-theta_max, theta_max, theta_scan)
        best = None
        for th in thetas:
            x, g, gv = _subgrad_scalar(
                x_ls, PA, PB, T, r, theta_max, 120, fixed_theta=float(th))
            if best is None or g < best[1]:
                best = (x, g, gv)
        x3, g3, gv3 = _subgrad_scalar(
            x_ls, PA, PB, T, r, theta_max, 300)
        if g3 < best[1]:
            best = (x3, g3, gv3)
        x, g, gv = best
    else:
        x, g, gv = _subgrad_scalar(
            x_ls, PA, PB, T, r, theta_max, 120)
        thetas = None
    return {"feasible": g <= _FEAS_TOL, "tx": x[0], "ty": x[1],
            "theta": x[2], "margin": -g, "margins": -gv,
            "theta_grid": thetas}


def solve_pose_batch(P, T, rad, theta_max: float, iters: int = 60,
                     theta_starts: tuple[float, ...] = (0.0,)
                     ) -> dict[str, np.ndarray]:
    """向量化求解 K 个样本：返回 tx,ty,theta, g*（K,）与逐位 g（K,n）。

    P 为 (n,2) 名义 B 中心或 (K,n,2) 实际 B 中心；T 为 (K,n,2) A 目标中心。
    每个样本同一迭代结构（活动匹配位可不同），全程 numpy 批运算；
    固定种子 + 确定性算法，同输入精确复现。
    """
    P3 = P if P.ndim == 3 else np.broadcast_to(P[None, :, :], T.shape)
    PB0 = P3[0]
    K, n, _ = T.shape
    Nn = float(n)
    sx = PB0[:, 0].sum()
    sy = PB0[:, 1].sum()
    sxx = float((PB0[:, 0] ** 2).sum())
    syy = float((PB0[:, 1] ** 2).sum())
    det = Nn * Nn * (sxx + syy) - Nn * (sx * sx + sy * sy)
    e = T - P3                                    # (K,n,2)
    r1 = e[:, :, 0].sum(axis=1)
    r2 = e[:, :, 1].sum(axis=1)
    r3 = (-P3[:, :, 1] * e[:, :, 0]
          + P3[:, :, 0] * e[:, :, 1]).sum(axis=1)
    if abs(det) > 1e-12:
        theta0 = (Nn * r3 + sy * r1 - sx * r2) / det
        tx0 = (r1 + sy * theta0) / Nn
        ty0 = (r2 - sx * theta0) / Nn
    else:
        mean_e = e.mean(axis=1)
        tx0, ty0, theta0 = mean_e[:, 0], mean_e[:, 1], np.zeros(K)
    theta0 = np.clip(theta0, -theta_max, theta_max)

    scale = max(1.0, float(np.linalg.norm(PB0, axis=1).mean()))
    best_g = np.full(K, np.inf)
    best_state = None
    for th_start in theta_starts:
        tx, ty = tx0.copy(), ty0.copy()
        th = np.full(K, float(np.clip(th_start, -theta_max, theta_max)))
        bg = np.full(K, np.inf)
        bs = None
        g0v = None
        kidx0 = np.arange(K)
        for t in range(iters):
            cth, sth = np.cos(th), np.sin(th)
            rx = cth[:, None] * P3[:, :, 0] - sth[:, None] * P3[:, :, 1]
            ry = sth[:, None] * P3[:, :, 0] + cth[:, None] * P3[:, :, 1]
            dx = rx + tx[:, None] - T[:, :, 0]
            dy = ry + ty[:, None] - T[:, :, 1]
            dist = np.hypot(dx, dy)
            gv = dist - rad
            k = np.argmax(gv, axis=1)
            kidx = (kidx0, k)
            g = gv[kidx]
            improved = g < bg
            if improved.any():
                bg = np.where(improved, g, bg)
                bs = (tx.copy(), ty.copy(), th.copy(), gv.copy())
            ux = dx[kidx]
            uy = dy[kidx]
            ud = np.hypot(ux, uy)
            ux = np.where(ud > 1e-12, ux / np.maximum(ud, 1e-12), 0.0)
            uy = np.where(ud > 1e-12, uy / np.maximum(ud, 1e-12), 0.0)
            if t == 0:
                g0v = np.maximum(np.abs(g), 0.02)
            gamma = g0v / math.sqrt(t + 1)
            pbx_k = P3[kidx0, k, 0]
            pby_k = P3[kidx0, k, 1]
            gth = ux * (-sth * pbx_k - cth * pby_k) \
                + uy * (cth * pbx_k - sth * pby_k)
            tx = tx - gamma * ux
            ty = ty - gamma * uy
            th = np.clip(th - gamma * gth / scale, -theta_max, theta_max)
        better = bg < best_g
        if best_state is None:
            best_g, best_state = bg, bs
        elif better.any():
            best_g = np.where(better, bg, best_g)
            mask = better[:, None]
            best_state = (
                np.where(better, bs[0], best_state[0]),
                np.where(better, bs[1], best_state[1]),
                np.where(better, bs[2], best_state[2]),
                np.where(mask, bs[3], best_state[3]))
    tx, ty, th, gv = best_state
    return {"tx": tx, "ty": ty, "theta": th, "g": best_g, "gv": gv}


# ------------------------------------------------------------- 最坏边界

def _vc_allowance(model: HoleModel) -> tuple[np.ndarray, list[dict]]:
    """VC 边界尺寸下的径向允许错位与逐匹配位余量贡献分解（mm）。"""
    n = model.n
    rad = np.zeros(n)
    contribs: list[dict] = []
    for i in range(n):
        fa, fb = model.features_a[i], model.features_b[i]
        if model.mate_kinds[i] == "float":
            da_min, db_min = fa.nom + fa.ei, fb.nom + fb.ei
            d_bolt_max = model.bolt_nom[i] + model.bolt_es[i]
            nom_clear = 0.5 * (fa.nom + fb.nom) - model.bolt_nom[i]
            size_a, size_b = 0.5 * fa.ei, 0.5 * fb.ei
            size_bolt = -model.bolt_es[i]
            allow = (da_min + db_min) * 0.5 - d_bolt_max \
                - 0.5 * (fa.tol + fb.tol)
            pieces = {"nominal_clearance": nom_clear,
                      "hole_a_size_deviation": size_a,
                      "hole_b_size_deviation": size_b,
                      "bolt_size_deviation": size_bolt,
                      "position_a": -0.5 * fa.tol,
                      "position_b": -0.5 * fb.tol}
        else:
            hole = fa if model.ext_sides[i] == "B" else fb
            pin = fb if model.ext_sides[i] == "B" else fa
            d_hole_min = hole.nom + hole.ei
            d_pin_max = pin.nom + pin.es
            nom_clear = 0.5 * (hole.nom - pin.nom)
            size_hole = 0.5 * hole.ei
            size_pin = -0.5 * pin.es
            allow = 0.5 * (d_hole_min - d_pin_max) \
                - 0.5 * (hole.tol + pin.tol)
            pieces = {"nominal_clearance": nom_clear,
                      "hole_size_deviation": size_hole,
                      "pin_size_deviation": size_pin,
                      "position_hole": -0.5 * hole.tol,
                      "position_pin": -0.5 * pin.tol}
        # rz 基准角度公差在该位的径向折算（两侧相加，保守最坏符号）
        ang = 0.0
        for side, f in (("A", fa), ("B", fb)):
            rz = [d for d in model.frames[side] if "rz" in d.constrains]
            r = math.hypot(f.x, f.y)
            for d in rz:
                if d.kind == DatumKind.EDGE.value:
                    ang += d.ang_half_rad * r
        pieces["datum_angular"] = -ang
        allow -= ang
        rad[i] = allow
        pieces["total_radial_allowance"] = allow
        contribs.append(pieces)
    return rad, contribs


def _translation_box(PA, PB, C, rad, x: np.ndarray, span: float) -> dict:
    """在最优 θ 处沿 ±x/±y 二分平移余量（mm，圆盘公共交集方向半径）。"""
    th = x[2]

    def line_margin(axis: int, direction: int) -> float:
        lo, hi = 0.0, span
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            t = x[:2].copy()
            t[axis] += direction * mid
            g, _, _ = _g_eval(th, t[0], t[1], PA, PB, C, rad)
            if g <= _FEAS_TOL:
                lo = mid
            else:
                hi = mid
        return lo

    return {
        "x_negative_mm": line_margin(0, -1),
        "x_positive_mm": line_margin(0, +1),
        "y_negative_mm": line_margin(1, -1),
        "y_positive_mm": line_margin(1, +1),
    }


def worst_case(model: HoleModel,
               extra_radius: np.ndarray | None = None) -> dict:
    PA = np.array([[f.x, f.y] for f in model.features_a])
    PB = np.array([[f.x, f.y] for f in model.features_b])
    Delta = PB - PA
    rad, contribs = _vc_allowance(model)
    if extra_radius is not None:
        rad = rad + np.asarray(extra_radius, dtype=float)

    # 名义位姿（t=0,θ=0）：目标中心为名义 A 中心 PA，
    # 名义错位 = |pB−pA|，最先干涉 = 名义余量最小者
    g0, _, gv0 = _g_eval(0.0, 0.0, 0.0, PA, PB, PA, rad)
    nom_margins = -gv0
    first_idx = int(np.argmin(nom_margins))

    grid = model.theta_grid if model.theta_grid % 2 == 1 else model.theta_grid + 1
    thetas = np.linspace(-model.theta_max_rad, model.theta_max_rad, grid)
    x_ls = _ls_start_scalar(PA, PB, PA, model.theta_max_rad)
    feas_flags, g_grid = [], []
    best = None
    for th in thetas:
        x, g, gv = _subgrad_scalar(
            x_ls, PA, PB, PA, rad, model.theta_max_rad,
            _WC_POLISH_ITERS, fixed_theta=float(th))
        feas_flags.append(g <= _FEAS_TOL)
        g_grid.append(g)
        if best is None or g < best[1]:
            best = (x, g, gv)
    x3, g3, gv3 = _subgrad_scalar(
        x_ls, PA, PB, PA, rad, model.theta_max_rad, 400)
    if g3 < best[1]:
        best = (x3, g3, gv3)
    x, g, gv = best
    feas_arr = np.array(feas_flags)
    if feas_arr.any():
        th_lo = float(thetas[feas_arr].min())
        th_hi = float(thetas[feas_arr].max())
    else:
        th_lo = th_hi = None

    box = _translation_box(
        PA, PB, PA, rad, x,
        span=max(10.0, 5.0 * float(rad.max()) + float(np.abs(PB).max())))
    margins = -gv
    limiting = int(np.argmin(margins))
    per_mate = [
        {"mate_id": model.mate_ids[i],
         "nominal_misalignment_mm": float(np.linalg.norm(Delta[i])),
         "vc_radial_allowance_mm": float(rad[i]),
         "margin_at_nominal_pose_mm": float(nom_margins[i]),
         "margin_at_best_pose_mm": float(margins[i]),
         "contributions_mm": contribs[i]}
        for i in range(model.n)
    ]
    return {
        "feasible": bool(g <= _FEAS_TOL),
        "best_pose": {
            "tx_mm": float(x[0]), "ty_mm": float(x[1]),
            "theta_deg": math.degrees(x[2]),
            "worst_margin_mm": float(-g),
        },
        "feasible_theta_range_deg": (
            None if th_lo is None
            else [math.degrees(th_lo), math.degrees(th_hi)]),
        "theta_feasible_fraction_of_grid": float(feas_arr.mean()),
        "theta_grid_margin_mm": [float(-gg) for gg in g_grid],
        "translation_range_mm": {
            "center": {"tx_mm": float(x[0]), "ty_mm": float(x[1])}, **box},
        "first_interference_at_nominal": {
            "mate_id": model.mate_ids[first_idx],
            "margin_mm": float(nom_margins[first_idx]),
            "interferes": bool(nom_margins[first_idx] < 0),
        },
        "limiting_mate_at_best_pose": {
            "mate_id": model.mate_ids[limiting],
            "margin_mm": float(margins[limiting]),
        },
        "per_mate": per_mate,
    }


# ------------------------------------------------------------- 蒙特卡洛

def _realization_batches(model: HoleModel, draw: dict[str, Any],
                         radius_inflation: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """全部样本：实际相对错位 C (K,n,2) 与径向余量 rad (K,n)。"""
    K = draw["diameters"][("A", 0)].shape[0]
    n = model.n
    PA = np.array([[f.x, f.y] for f in model.features_a])
    PB = np.array([[f.x, f.y] for f in model.features_b])
    CA = np.tile(PA[None, :, :], (K, 1, 1)).astype(float)
    CB = np.tile(PB[None, :, :], (K, 1, 1)).astype(float)
    for side, centers, feats in (("A", CA, model.features_a),
                                 ("B", CB, model.features_b)):
        for j in range(len(feats)):
            centers[:, j, :] += draw["positions"][side][j]
    rad = np.zeros((K, n))
    da = np.column_stack([draw["diameters"][("A", i)] for i in range(n)])
    db = np.column_stack([draw["diameters"][("B", i)] for i in range(n)])
    for i in range(n):
        if model.mate_kinds[i] == "float":
            rad[:, i] = 0.5 * (da[:, i] + db[:, i]) - draw["bolt"][:, i]
        elif model.ext_sides[i] == "B":
            rad[:, i] = 0.5 * (da[:, i] - db[:, i])
        else:
            rad[:, i] = 0.5 * (db[:, i] - da[:, i])
    if radius_inflation is not None:
        rad = rad + radius_inflation[None, :]
    # B：实际 B 中心（被旋转平移）；T：实际 A 中心（目标）
    return CB, CA, rad


def monte_carlo(model: HoleModel, n: int | None = None,
                seed: int | None = None,
                radius_inflation: np.ndarray | None = None,
                draw: dict[str, Any] | None = None) -> dict[str, Any]:
    n = n or model.mc_samples
    seed = model.seed if seed is None else seed
    if draw is None:
        draw = sample_realization(model, n, seed=seed)
    P, T, rad = _realization_batches(model, draw, radius_inflation)

    mc_points = model.mc_theta_points
    if mc_points % 2 == 0:
        mc_points += 1
    theta_starts = tuple(np.linspace(
        -model.theta_max_rad, model.theta_max_rad, mc_points))
    # 固定 θ 起点 + 多 θ 起点联合优化两批，逐样本取更优
    sol0 = solve_pose_batch(P, T, rad, model.theta_max_rad, iters=70)
    sol1 = solve_pose_batch(P, T, rad, model.theta_max_rad, iters=70,
                            theta_starts=theta_starts)
    pick1 = sol1["g"] < sol0["g"]
    g = np.where(pick1, sol1["g"], sol0["g"])
    tx = np.where(pick1, sol1["tx"], sol0["tx"])
    ty = np.where(pick1, sol1["ty"], sol0["ty"])
    th = np.where(pick1, sol1["theta"], sol0["theta"])
    gv = np.where(pick1[:, None], sol1["gv"], sol0["gv"])
    margins = -gv

    ok = g <= _FEAS_TOL
    poses = np.column_stack([tx, ty, th])
    worst_idx = np.argmin(margins, axis=1)
    first_counts = {mid: 0 for mid in model.mate_ids}
    for wi in worst_idx[~ok]:
        first_counts[model.mate_ids[int(wi)]] += 1

    per_mate = []
    for i, mid in enumerate(model.mate_ids):
        feas_margins = margins[ok, i] if ok.any() else margins[:, i]
        per_mate.append({
            "mate_id": mid,
            "mean_margin_mm": float(feas_margins.mean()),
            "p05_margin_mm": float(np.quantile(feas_margins, 0.05)),
            "interference_frequency": float((margins[:, i] < 0).mean()),
            "first_interference_count": int(first_counts[mid]),
            "first_interference_share_of_failures":
                (first_counts[mid] / int((~ok).sum()) if (~ok).any() else 0.0),
        })

    def pose_stats(col, deg=False):
        v = poses[ok, col] if ok.any() else np.array([0.0])
        if deg:
            v = np.degrees(v)
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()),
                "p05": float(np.quantile(v, 0.05)),
                "p95": float(np.quantile(v, 0.95))}

    return {
        "samples": n,
        "random_seed": seed,
        "assembly_success_rate": float(ok.mean()),
        "failure_probability": float(1.0 - ok.mean()),
        "feasible_pose_range": {
            "tx_mm": pose_stats(0), "ty_mm": pose_stats(1),
            "theta_deg": pose_stats(2, deg=True),
            "note": "统计量仅在可行样本上计算；min/max 即样本可行位姿范围",
        },
        "first_interference": sorted(
            ({"mate_id": mid, "count": cnt}
             for mid, cnt in first_counts.items()),
            key=lambda z: (-z["count"], z["mate_id"])),
        "per_mate": per_mate,
    }


# ------------------------------------------------------------- 快照与总装
# ------------------------------------------------------------- 快照与总装

def _feature_snapshot(f: _Feature) -> dict:
    return {
        "mate_id": f.mate_id, "side": f.side, "kind": f.kind,
        "x_mm": f.x, "y_mm": f.y,
        "nominal_diameter_mm": f.nom,
        "diameter_upper_deviation_mm": f.es,
        "diameter_lower_deviation_mm": f.ei,
        "position_tolerance_mm": f.tol,
        "material_condition": f.mc,
        "distribution": f.distribution,
        "diameter_std_dev_mm": f.d_sigma,
        "position_std_dev_mm": f.p_sigma,
        "position_datum_refs": list(f.refs),
    }


def model_snapshot(model: HoleModel) -> dict:
    return {
        "name": model.name,
        "unit_submitted": model.unit,
        "normalized_unit": "mm",
        "mates": [
            {"mate_id": model.mate_ids[i], "kind": model.mate_kinds[i],
             "external_feature_side": model.ext_sides[i],
             "bolt_nominal_mm": model.bolt_nom[i],
             "bolt_upper_deviation_mm": model.bolt_es[i],
             "bolt_lower_deviation_mm": model.bolt_ei[i]}
            for i in range(model.n)],
        "features_a": [_feature_snapshot(f) for f in model.features_a],
        "features_b": [_feature_snapshot(f) for f in model.features_b],
        "datum_frames": {
            side: [
                {"label": d.label, "order": d.order, "kind": d.kind,
                 "constrains": list(d.constrains),
                 "material_condition": d.material_condition,
                 "nominal_diameter_mm": d.nom or None,
                 "diameter_upper_deviation_mm": d.es or None,
                 "diameter_lower_deviation_mm": d.ei or None,
                 "lever_radius_mm": d.lever,
                 "normal": [d.nx, d.ny], "line_offset_mm": d.offset,
                 "angular_half_range_deg":
                     math.degrees(d.ang_half_rad) if d.ang_half_rad else 0.0,
                 "distribution": d.distribution}
                for d in model.frames[side]]
            for side in ("A", "B")},
        "monte_carlo": {"samples": model.mc_samples, "seed": model.seed,
                        "substreams":
                            "SeedSequence([seed, 0x686F6C65, family])："
                            "0/1 两侧直径，2 螺栓，3/4 位置，5/6 基准尺寸，"
                            "7/8 基准转角"},
        "theta_search": {"half_width_deg": math.degrees(model.theta_max_rad),
                         "worst_case_grid": model.theta_grid,
                         "mc_scan_points": model.mc_theta_points},
        "formulas": [
            "固定连接件径向余量：(D_孔,min − d_销,max)/2 − (t_A+t_B)/2",
            "浮动连接件：(D_A,min+D_B,min)/2 − d_螺栓,max − (t_A+t_B)/2",
            "最坏边界取 VC 边界尺寸：bonus=0、基准偏移=0；圆盘族 "
            "|t+θ·J·pB_i−Δp_i| ≤ a_i 求公共交集",
            "bonus（MMC，diametral）：孔 max(0,D_实际−D_min)；"
            "销 max(0,d_max−d_实际)；LMC 取对侧；RFS=0",
            "基准偏移：尺寸基准 MMC/LMC 的实际间隙按位置度引用逐孔计入"
            "有效位置度；边线 RFS 基准无漂移",
            "位置误差两轴：uniform 半宽=(t+bonus+偏移)/2；triangular "
            "同半宽；normal σ=position_std_dev（缺省 t/6）",
            "rz 基准：尺寸基准转角 ≤ 间隙/(2·力臂)；边线基准按角度公差；"
            "公共模式转角经装配 θ 优化后取残差",
            "固定种子 np.random SeedSequence 子流，同种子/同样本数精确复现",
        ],
    }


def analyze(payload: HolePatternCreate) -> dict:
    model = build_model(payload)
    wc = worst_case(model)
    mc = monte_carlo(model)
    return {
        "status": "feasible" if wc["feasible"] else "worst_case_infeasible",
        "worst_case": wc,
        "monte_carlo": mc,
        "snapshot": model_snapshot(model),
        "traceability": {
            "unit_policy": "所有提交长度经 LengthUnit 换算为 mm，原始单位保留"
                          "在 submitted_input",
            "distribution_policy":
                "直径/位置偏差复用基线尺寸链的 normal/uniform/triangular 口径",
            "frozen_inputs": "孔系、匹配关系、基准框架与随机种子随版本冻结",
        },
    }
