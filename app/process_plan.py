"""工序尺寸方案引擎（内部统一 mm）。

建模
====
一次零件加工路线独立成版。表面是基本节点：

* **毛坯面**：加工前已存在。``blank_dimensions`` 给出毛坯面之间（或相对
  ``origin_surface``）的有向毛坯尺寸，按提交顺序把新表面挂到已存在的表面
  集合上——任意一条毛坯尺寸的两个端面必须有一个已在集合内，否则判为
  基准路径断开。
* **工序面**：``operations`` 严格按加工顺序排列。每道工序给出定位基准
  ``datum_surface``（必须在本工序之前形成：原点 / 毛坯面 / 更早工序面）
  与被加工面 ``machined_surface``（必须是本工序首次形成；重复加工同一
  表面判为重复约束），工序尺寸边为 datum→machined。
* **不可变设计尺寸链**：``design_dimensions`` 是两端（或多段路径）落在
  已形成表面上的**设计闭环**，带名义值与上下偏差；可由冻结基线链导入
  （导入后整条链快照随版本冻结，来源链后续更新不改写历史方案）。
* **余量闭环**：可选的最小加工余量要求（毛坯面→同/邻近工序面），作为
  单边规格参与校核与排序，不参与名义反算的消元。

边变量与表面坐标
----------------
内部先固定原点表面坐标 x_origin = 0（图纸基准），每条有向边 e: a→b 取值
y_e（工序尺寸/毛坯尺寸）。表面坐标方程 H·x = E·y（x_b−x_a = y_e）构成
按提交顺序逐边挂接的树，H 去掉原点列后可逆；设计闭环 D·x 落在 [LSL,USL]，
经 ``P = D·H⁻¹·E`` 映射到边空间，再取工序列得到**工序尺寸→设计闭环的
传递矩阵 B**。

结构校验（创建方案时拒绝，422）
==============================
* 工序引用了尚未形成的表面（datum 晚于本工序形成）；
* 基准路径断开（毛坯尺寸无法挂接到已存在表面集合）；
* 重复约束（重复加工同一表面 / 同一对表面上的重复有向边）；
* 设计闭环系数全为 0（闭环端点不是方案表面）或闭环行线性相关
  （冗余设计约束，指出相关闭环）；
* 存在完全不出现在任何设计/余量闭环里的工序尺寸——其工序自由度不受设计
  尺寸链约束，传递矩阵对应列为零（秩不足），必须补充余量闭环或调整基准；
  422 诊断给出相关工序与自由度数（零空间方向）。

名义反算与误差传播
==================
调用方可锁定已定工序尺寸（``locked_dimensions`` 给名义值或留用提交值），
其余工序尺寸标为待反算。系统对自由变量做两段单纯形线性规划，求出满足
全部设计闭环名义区间 [N+EI, N+ES] 的**名义值范围**（无上界/下界的维度
判为欠约束，指出工序与剩余自由度），再逐项传播：

* **极值法 WC**：μ_L ± Σ|P_Le|·T_e；
* **RSS**：σ_L² = P_Lᵀ(DσDρ)P_L（工序/毛坯尺寸值之间的 Pearson 相关）；
* **固定种子蒙特卡洛**：全部边只抽样一次（复用单链引擎的 copula 抽样），
  C = Y·Pᵀ 得到各闭环同源样本，报告 σ / 分位 / 经验超差率。

多解排序
========
待反算工序尺寸在「机床能力档位（公差系数）× 标准尺寸步进名义网格」上
组合枚举：先 WC 粗筛，再对前列候选用与方案同源的固定种子 MC 复核，按
**设计闭环达标数（多者优先）→ 最差余量（大者优先）→ 尺寸改动量（小者
优先）→ 制造成本（低者优先）** 排列。选定后冻结设计链快照、工序路线、
传递矩阵与随机种子。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import engine
from .units import to_mm

EPS = 1e-9
RANKS = (1, 3)

# 名义组合 / MC 复核的保护性上限（避免自由维度过大时组合爆炸）
MAX_GRADE_COMBINATIONS = 20_000
MAX_NOMINAL_GRID_TOTAL = 20_000
TOP_CANDIDATES_MC = 30


class ProcessPlanError(ValueError):
    """工序尺寸方案输入 / 结构错误（API 层映射为 422）。"""


# ------------------------------------------------------------------ 数据结构

@dataclass
class Edge:
    """一条有向表面关系边（毛坯尺寸或工序尺寸，数值均为 mm）。"""

    id: str
    start: str
    end: str
    kind: str                  # blank / operation
    op_index: int | None       # 工序序号（从 1 起），毛坯为 None
    op_id: str | None
    datum: str | None
    machined: str | None
    nominal: float
    upper_dev: float
    lower_dev: float
    mid: float
    half_width: float
    sigma: float
    sigma_explicit: bool
    distribution: str
    unit: str
    manufacturing_cost: float
    note: str = ""
    original: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.op_id if self.kind == "operation" else self.id


@dataclass
class Closure:
    """一个设计 / 余量闭环（表面系数已解析为边空间系数 P）。"""

    id: str
    kind: str                  # design / stock
    note: str
    lsl_mm: float
    usl_mm: float | None       # 余量闭环只有下限
    nominal: float
    upper_dev: float
    lower_dev: float
    surface_coeff: dict[str, float]
    path: list[dict]
    p_edge: dict[str, float] = field(default_factory=dict)
    op_refs: list[str] = field(default_factory=list)


@dataclass
class ProcessPlan:
    """规范化后的工序尺寸方案（一次加工路线一版）。"""

    name: str
    origin: str
    surfaces: list[str]
    edges: list[Edge]
    closures: list[Closure]
    corr: np.ndarray
    mc_samples: int
    seed: int
    design_snapshot: dict
    blank_surfaces: list[str]

    @property
    def ids(self) -> list[str]:
        return [e.id for e in self.edges]

    def edge(self, eid: str) -> Edge:
        return self._by_id[eid]

    def __post_init__(self):
        self._by_id = {e.id: e for e in self.edges}

    # ---- 矩阵 ----
    def _h_e_matrices(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """返回 (H_full(n×s), E(n×n 边方程), 表面顺序)。

        边 e: a→b 给 x_b−x_a = y_e；H 行对应边，列按表面顺序（含原点）。
        """
        sidx = {s: i for i, s in enumerate(self.surfaces)}
        n, k = len(self.surfaces), len(self.edges)
        h = np.zeros((k, n))
        e_mat = np.zeros((k, k))
        for j, e in enumerate(self.edges):
            h[j, sidx[e.start]] = -1.0
            h[j, sidx[e.end]] = 1.0
            e_mat[j, j] = 1.0
        return h, e_mat, self.surfaces

    def coordinate_map(self) -> tuple[np.ndarray, np.ndarray]:
        """边值 → 表面坐标 的映射（x = M·y，原点坐标 0）及边→边方程。

        固定原点列为 0 后对 H 取伪逆；按挂接顺序成树时恰为逐边累加。
        返回 (x_from_y (s×n 含原点行), E)。
        """
        h, e_mat, surfaces = self._h_e_matrices()
        o = surfaces.index(self.origin)
        keep = [i for i in range(len(surfaces)) if i != o]
        h_na = h[:, keep]
        # 按挂接顺序每边引入一个新表面：边数 = 非原点表面数，H_na 为方阵。
        # 坐标方程 H_na·x_na = y，直接求逆即 x = H_na⁻¹·y（逐边累加）。
        if h_na.shape != (len(keep), len(keep)):
            raise ProcessPlanError(
                "表面关系图不是树：边数 "
                f"{h_na.shape[0]} 与非原点表面数 {len(keep)} 不一致"
                "（存在断开或闭环表面关系——重复约束/基准路径断开）")
        try:
            x_from_y_na = np.linalg.solve(h_na, e_mat)
        except np.linalg.LinAlgError as exc:
            raise ProcessPlanError(
                "表面坐标矩阵奇异：表面关系挂接断开或存在闭环"
                "（基准路径断开 / 重复约束）") from exc
        x_from_y = np.zeros((len(surfaces), len(self.edges)))
        for col, si in enumerate(keep):
            x_from_y[si, :] = x_from_y_na[col, :]
        return x_from_y, e_mat

    def transfer_matrix(self) -> np.ndarray:
        """L×n 传递矩阵 P：闭环 L = Σ_e P[L,e]·y_e（边顺序 = self.ids）。"""
        x_from_y, _ = self.coordinate_map()
        sidx = {s: i for i, s in enumerate(self.surfaces)}
        p = np.zeros((len(self.closures), len(self.edges)))
        for li, cl in enumerate(self.closures):
            for surf, coef in cl.surface_coeff.items():
                p[li, :] += coef * x_from_y[sidx[surf], :]
        return p


# ------------------------------------------------------------- 方案构建

def _mm(value, unit) -> float:
    u = unit.value if hasattr(unit, "value") else str(unit)
    return to_mm(float(value), u), u


def build_plan(req, design_dims: list[dict]) -> ProcessPlan:
    """由校验后的请求与设计维度（mm 字典）构建规范化方案并做结构校验。

    design_dims 每项：id/start/end/nominal/upper_deviation/lower_deviation/
    unit/source(chain|inline)，值已换算 mm。
    """
    name = req.name
    origin = req.origin_surface

    surfaces = [origin]
    formed = {origin}
    edges: list[Edge] = []
    edge_pairs: set[tuple[str, str]] = set()

    def add_edge(edge_id, a, b, kind):
        if a == b:
            raise ProcessPlanError(f"尺寸 {edge_id}: 起点与终点不能相同（{a}）")
        pair = (a, b)
        if pair in edge_pairs:
            label = "有向表面关系"
            raise ProcessPlanError(
                f"重复约束: {a}→{b} 上已存在尺寸（{label}），{edge_id} 与其重复")
        edge_pairs.add(pair)

    # ---- 毛坯尺寸：按提交顺序把新表面挂到已形成集合 ----
    blank_surfaces: list[str] = []
    for bd in req.blank_dimensions:
        a, b = bd.start_surface, bd.end_surface
        if a not in formed and b not in formed:
            raise ProcessPlanError(
                f"毛坯尺寸 {bd.id}: 基准路径断开——端面 {a} 与 {b} 均未形成"
                f"（已形成表面: {sorted(formed)}）；毛坯尺寸必须把新表面"
                "挂接到原点或已声明的毛坯面上")
        if a in formed and b in formed:
            raise ProcessPlanError(
                f"毛坯尺寸 {bd.id}: 两端面 {a}、{b} 均已形成，"
                "毛坯尺寸必须恰好引入一个新毛坯面（重复约束/闭环毛坯链）")
        add_edge(bd.id, a, b, "blank")
        new_surf = b if b not in formed else a
        # 若新面在 start，则有向边方向与挂接方向相反：交换端点保持
        # datum(已形成)→machined(新面) 的边语义。
        if new_surf == a:
            a, b = b, a
        n_mm, u = _mm(bd.nominal, bd.unit)
        es_mm = to_mm(bd.upper_deviation, u)
        ei_mm = to_mm(bd.lower_deviation, u)
        sigma_explicit = bd.std_dev is not None
        sigma_mm = (to_mm(bd.std_dev, u) if sigma_explicit
                    else engine._theoretical_sigma(bd.distribution.value,
                                                   (es_mm - ei_mm) / 2.0))
        edges.append(Edge(
            id=bd.id, start=a, end=b, kind="blank", op_index=None,
            op_id=None, datum=a, machined=b, nominal=n_mm,
            upper_dev=es_mm, lower_dev=ei_mm,
            mid=(es_mm + ei_mm) / 2.0, half_width=(es_mm - ei_mm) / 2.0,
            sigma=sigma_mm, sigma_explicit=sigma_explicit,
            distribution=bd.distribution.value, unit=u,
            manufacturing_cost=float(bd.manufacturing_cost),
            note=getattr(bd, "note", "") or "",
            original={"nominal": bd.nominal,
                      "upper_deviation": bd.upper_deviation,
                      "lower_deviation": bd.lower_deviation,
                      "std_dev": bd.std_dev, "unit": u}))
        surfaces.append(new_surf)
        formed.add(new_surf)
        blank_surfaces.append(new_surf)

    # ---- 工序：严格顺序，基准已形成、被加工面首次形成 ----
    op_ids: set[str] = set()
    for i, op in enumerate(req.operations, start=1):
        if op.id in op_ids:
            raise ProcessPlanError(f"工序 id 重复: {op.id}")
        op_ids.add(op.id)
        datum, machined = op.datum_surface, op.machined_surface
        if datum not in formed:
            later = [o.id for o in req.operations[i:]
                     if o.machined_surface == datum]
            hint = (f"，该表面由后续工序 {later} 形成" if later else
                    "，该表面既非原点/毛坯面，也不是更早工序的被加工面")
            raise ProcessPlanError(
                f"工序 {op.id}（第 {i} 道）引用了尚未形成的定位基准 "
                f"{datum!r}{hint}；定位基准必须在本工序之前形成")
        if machined in formed:
            if machined == origin:
                where = "原点表面"
            elif machined in blank_surfaces:
                where = "毛坯面"
            else:
                prev = next((e.op_id for e in edges
                             if e.kind == "operation" and e.end == machined),
                            "更早工序")
                where = f"工序 {prev} 已加工的表面"
            raise ProcessPlanError(
                f"工序 {op.id}（第 {i} 道）的被加工面 {machined!r} 已形成"
                f"（{where}）：同一表面不得重复加工（重复约束）")
        add_edge(op.id, datum, machined, "operation")
        n_mm, u = _mm(op.nominal, op.unit)
        es_mm = to_mm(op.upper_deviation, u)
        ei_mm = to_mm(op.lower_deviation, u)
        sigma_explicit = op.std_dev is not None
        sigma_mm = (to_mm(op.std_dev, u) if sigma_explicit
                    else engine._theoretical_sigma(op.distribution.value,
                                                   (es_mm - ei_mm) / 2.0))
        edges.append(Edge(
            id=op.id, start=datum, end=machined, kind="operation",
            op_index=i, op_id=op.id, datum=datum, machined=machined,
            nominal=n_mm, upper_dev=es_mm, lower_dev=ei_mm,
            mid=(es_mm + ei_mm) / 2.0, half_width=(es_mm - ei_mm) / 2.0,
            sigma=sigma_mm, sigma_explicit=sigma_explicit,
            distribution=op.distribution.value, unit=u,
            manufacturing_cost=float(op.manufacturing_cost),
            note=op.note or "",
            original={"nominal": op.nominal,
                      "upper_deviation": op.upper_deviation,
                      "lower_deviation": op.lower_deviation,
                      "std_dev": op.std_dev, "unit": u}))
        surfaces.append(machined)
        formed.add(machined)

    if not edges:
        raise ProcessPlanError("方案至少需要一条毛坯尺寸或工序尺寸")

    # ---- 设计闭环（不可变设计尺寸链） ----
    closures: list[Closure] = []
    closure_ids: set[str] = set()
    for dd in design_dims:
        cid = dd["id"]
        if cid in closure_ids:
            raise ProcessPlanError(f"设计闭环 id 重复: {cid}")
        closure_ids.add(cid)
        a, b = dd["start"], dd["end"]
        sign = int(dd.get("sign", 1))
        missing = [s for s in (a, b) if s not in formed]
        if missing:
            raise ProcessPlanError(
                f"设计尺寸 {cid} 引用了方案中不存在的表面: {missing}；"
                "设计闭环端面必须是毛坯面或某道工序的被加工面")
        # 偏差字段兼容两种键：API 统一构造 *_deviation；快照重建可给 *_dev
        es = dd.get("upper_deviation", dd.get("upper_dev"))
        ei = dd.get("lower_deviation", dd.get("lower_dev"))
        coeff = {a: -float(sign), b: float(sign)}
        path = [{"surface_a": a, "surface_b": b, "sign": sign,
                 "edge": [a, b]}]
        closures.append(Closure(
            id=cid, kind="design", note=dd.get("note", ""),
            lsl_mm=dd["nominal"] + ei,
            usl_mm=dd["nominal"] + es,
            nominal=dd["nominal"], upper_dev=es, lower_dev=ei,
            surface_coeff=coeff, path=path))

    # ---- 余量闭环 ----
    for sc in req.stock_closures:
        cid = sc.id
        if cid in closure_ids:
            raise ProcessPlanError(f"闭环 id 重复: {cid}")
        closure_ids.add(cid)
        a, b = sc.blank_surface, sc.machined_surface
        missing = [s for s in (a, b) if s not in formed]
        if missing:
            raise ProcessPlanError(
                f"余量闭环 {cid} 引用了方案中不存在的表面: {missing}")
        if a not in blank_surfaces and a != origin:
            raise ProcessPlanError(
                f"余量闭环 {cid}: blank_surface {a!r} 不是毛坯面/原点")
        sign = int(sc.sign)
        min_mm, u = _mm(sc.min_allowance, sc.unit)
        closures.append(Closure(
            id=cid, kind="stock", note=sc.note or "",
            lsl_mm=min_mm, usl_mm=None, nominal=min_mm,
            upper_dev=0.0, lower_dev=0.0,
            surface_coeff={a: -float(sign), b: float(sign)},
            path=[{"surface_a": a, "surface_b": b, "sign": sign,
                   "edge": [a, b]}]))

    # ---- 相关矩阵（尺寸值 Pearson 相关） ----
    ids = [e.id for e in edges]
    idx = {eid: i for i, eid in enumerate(ids)}
    corr = np.eye(len(ids))
    seen_pairs: set[frozenset[str]] = set()
    for c in req.correlations:
        if c.dim_a not in idx or c.dim_b not in idx:
            raise ProcessPlanError(
                f"相关系数引用了方案外尺寸: {c.dim_a!r}, {c.dim_b!r}；"
                f"方案尺寸: {ids}")
        key = frozenset((c.dim_a, c.dim_b))
        if key in seen_pairs:
            raise ProcessPlanError(
                f"尺寸 {c.dim_a} 与 {c.dim_b} 的相关系数重复声明")
        seen_pairs.add(key)
        i, j = idx[c.dim_a], idx[c.dim_b]
        corr[i, j] = corr[j, i] = c.rho
    eig = np.linalg.eigvalsh((corr + corr.T) / 2.0).min()
    if eig < -1e-8:
        raise ProcessPlanError(
            f"相关矩阵非半正定: 最小特征值 {eig:.3e}")

    plan = ProcessPlan(
        name=name, origin=origin, surfaces=surfaces, edges=edges,
        closures=closures, corr=corr, mc_samples=req.mc_samples,
        seed=req.random_seed, design_snapshot={"dimensions": design_dims},
        blank_surfaces=blank_surfaces)

    _resolve_transfer_and_validate(plan)
    return plan


def _resolve_transfer_and_validate(plan: ProcessPlan) -> None:
    """把闭环表面系数映射到边空间，并做传递矩阵结构校验。"""
    p_full = plan.transfer_matrix()
    for li, cl in enumerate(plan.closures):
        cl.p_edge = {plan.edges[j].id: float(p_full[li, j])
                     for j in range(len(plan.edges))
                     if abs(p_full[li, j]) > EPS}
        cl.op_refs = [plan.edges[j].op_id
                      for j in range(len(plan.edges))
                      if abs(p_full[li, j]) > EPS
                      and plan.edges[j].kind == "operation"]

    design_rows = [i for i, c in enumerate(plan.closures)
                   if c.kind == "design"]
    if not design_rows:
        raise ProcessPlanError(
            "方案必须引用至少一个设计尺寸（设计闭环）；"
            "无设计闭环时无法构造工序尺寸到设计闭环的传递矩阵")

    # 1) 零行闭环：端面不是方案表面（理论上前面已拦截，防御性检查）
    zero_rows = [plan.closures[i].id for i in design_rows
                 if np.all(np.abs(p_full[i]) < EPS)]
    if zero_rows:
        raise ProcessPlanError(
            f"设计闭环 {zero_rows} 的传递系数全为 0：闭环端点未连接到"
            "工序/毛坯表面关系图（基准路径断开）")

    # 2) 设计闭环行线性相关：冗余设计约束
    pd_ = p_full[design_rows, :]
    r = int(np.linalg.matrix_rank(pd_, tol=EPS * max(pd_.shape[0], 1)))
    if r < len(design_rows):
        # 找出可由其余行线性表示的闭环
        dep = _dependent_rows(pd_)
        names = [plan.closures[design_rows[i]].id for i in dep]
        raise ProcessPlanError(
            f"设计闭环存在冗余（重复约束）: 闭环 {names} 与其它设计闭环"
            "线性相关；请去除重复或相互推导的设计尺寸")

    # 3) 零列工序尺寸：不出现在任何闭环里，工序自由度不受设计链约束
    op_cols = [j for j, e in enumerate(plan.edges) if e.kind == "operation"]
    any_row = np.abs(p_full).max(axis=0)
    zero_cols = [j for j in op_cols if any_row[j] <= EPS]
    if zero_cols:
        ops = [f"工序 {plan.edges[j].op_id}（被加工面 "
               f"{plan.edges[j].machined}）" for j in zero_cols]
        raise ProcessPlanError(
            "传递矩阵秩不足：以下工序尺寸不出现在任何设计/余量闭环中，"
            f"其工序自由度不受设计尺寸链约束（{len(zero_cols)} 个自由度）: "
            + "；".join(ops)
            + "。请补充连接该表面的余量闭环，或改用已被设计尺寸覆盖的"
            "表面作为定位基准")

    # 4) 工序子矩阵列空间：标出设计约束未覆盖的工序自由度（不拒绝，
    #    留给余量闭环 + 锁定；若求解时仍欠约束，solve 阶段按自由度报错）。


def _dependent_rows(mat: np.ndarray) -> list[int]:
    """返回一组线性相关行的下标（可由其余行张成的行）。"""
    out, chosen = [], []
    for i in range(mat.shape[0]):
        basis = mat[chosen, :] if chosen else np.zeros((0, mat.shape[1]))
        if np.linalg.matrix_rank(
                np.vstack([basis, mat[i]]), tol=EPS * max(mat.shape[0], 1)) \
                == np.linalg.matrix_rank(basis, tol=EPS * max(mat.shape[0], 1)):
            out.append(i)
        else:
            chosen.append(i)
    return out


# ------------------------------------------------------------- 快照与重建

def plan_snapshot(plan: ProcessPlan, source: dict) -> dict:
    """随版本冻结的完整快照（设计链/路线/矩阵/种子；来源更新不改写）。"""
    return {
        "name": plan.name,
        "origin_surface": plan.origin,
        "surface_order": plan.surfaces,
        "blank_surfaces": plan.blank_surfaces,
        "edges": [
            {
                "edge_id": e.id,
                "kind": e.kind,
                "operation_index": e.op_index,
                "operation_id": e.op_id,
                "edge": [e.start, e.end],
                "datum_surface": e.datum,
                "machined_surface": e.machined,
                "distribution": e.distribution,
                "sigma_explicit": e.sigma_explicit,
                "manufacturing_cost": e.manufacturing_cost,
                "note": e.note,
                "normalized_mm": {
                    "nominal": e.nominal,
                    "upper_deviation": e.upper_dev,
                    "lower_deviation": e.lower_dev,
                    "mid_shift": e.mid,
                    "half_width": e.half_width,
                    "sigma": e.sigma,
                },
                "original_representation": e.original,
            }
            for e in plan.edges
        ],
        "closures": [
            {
                "closure_id": cl.id,
                "kind": cl.kind,
                "note": cl.note,
                "path": cl.path,
                "surface_coefficients": cl.surface_coeff,
                "edge_coefficients": cl.p_edge,
                "operation_refs": cl.op_refs,
                "spec_mm": {
                    "lower_limit": cl.lsl_mm,
                    "upper_limit": cl.usl_mm,
                },
                "nominal_mm": cl.nominal,
                "deviations_mm": {
                    "upper": cl.upper_dev, "lower": cl.lower_dev},
            }
            for cl in plan.closures
        ],
        "transfer_matrix": plan.transfer_matrix().tolist(),
        "correlation_matrix": plan.corr.tolist(),
        "monte_carlo": {"samples": plan.mc_samples, "seed": plan.seed},
        "design_chain_snapshot": plan.design_snapshot,
        "source": source,
    }


def plan_from_snapshot(snap: dict) -> ProcessPlan:
    """从冻结快照重建规范化方案（求解/复核据此精确重放）。"""
    edges = [
        Edge(
            id=e["edge_id"], start=e["edge"][0], end=e["edge"][1],
            kind=e["kind"], op_index=e["operation_index"],
            op_id=e["operation_id"], datum=e["datum_surface"],
            machined=e["machined_surface"],
            nominal=e["normalized_mm"]["nominal"],
            upper_dev=e["normalized_mm"]["upper_deviation"],
            lower_dev=e["normalized_mm"]["lower_deviation"],
            mid=e["normalized_mm"]["mid_shift"],
            half_width=e["normalized_mm"]["half_width"],
            sigma=e["normalized_mm"]["sigma"],
            sigma_explicit=bool(e["sigma_explicit"]),
            distribution=e["distribution"],
            unit=e["original_representation"]["unit"],
            manufacturing_cost=float(e["manufacturing_cost"]),
            note=e.get("note", ""),
            original=e["original_representation"])
        for e in snap["edges"]
    ]
    closures = [
        Closure(
            id=c["closure_id"], kind=c["kind"], note=c.get("note", ""),
            lsl_mm=c["spec_mm"]["lower_limit"],
            usl_mm=c["spec_mm"]["upper_limit"],
            nominal=c["nominal_mm"],
            upper_dev=c["deviations_mm"]["upper"],
            lower_dev=c["deviations_mm"]["lower"],
            surface_coeff={k: float(v)
                           for k, v in c["surface_coefficients"].items()},
            path=c["path"],
            p_edge={k: float(v) for k, v in c["edge_coefficients"].items()},
            op_refs=c.get("operation_refs", []))
        for c in snap["closures"]
    ]
    plan = ProcessPlan(
        name=snap["name"], origin=snap["origin_surface"],
        surfaces=snap["surface_order"], edges=edges, closures=closures,
        corr=np.array(snap["correlation_matrix"], dtype=float),
        mc_samples=snap["monte_carlo"]["samples"],
        seed=snap["monte_carlo"]["seed"],
        design_snapshot=snap["design_chain_snapshot"],
        blank_surfaces=snap["blank_surfaces"])
    return plan


# ------------------------------------------------------------- 线性规划

class LPError(ValueError):
    pass


def linprog_bounds(n_vars, objective, a_ub, b_ub,
                   lower_bounds=None, upper_bounds=None):
    """两段单纯形：求 min objectiveᵀx s.t. A_ub·x ≤ b_ub，lb ≤ x ≤ ub。

    ``objective`` 长度 = n_vars（目标系数向量）。
    返回 (x, optimal_value, status)，status ∈ {optimal, infeasible,
    unbounded}。变量默认非负（工序名义尺寸 > 0）。
    """
    n = int(n_vars)
    a_ub = np.array(a_ub, dtype=float)
    a_ub = a_ub.reshape(0, n) if a_ub.size == 0 else a_ub
    b_ub = np.array(b_ub, dtype=float).reshape(a_ub.shape[0])
    lb = np.zeros(n) if lower_bounds is None else np.array(lower_bounds, float)
    ub = (np.full(n, np.inf) if upper_bounds is None
          else np.array(upper_bounds, float))

    # 平移 x = z + lb（z ≥ 0 由单纯形自然保证，不另加非负行，否则翻转后
    # 产生系数为负的伪 ≥ 约束会破坏阶段 I）；一般约束 A z ≤ b − A·lb，
    # 有限上界另加 z ≤ ub−lb。
    shift = lb.copy()
    rows = [a_ub.reshape(-1, n)]
    rhs = [(b_ub - a_ub.reshape(-1, n) @ shift).reshape(-1)]
    finite = np.isfinite(ub - lb)
    if finite.any():
        rows.append(np.eye(n)[finite])
        rhs.append((ub - lb)[finite])
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    return _simplex(np.array(objective, dtype=float), A, b, shift)


def _simplex(c_vec, A_ub, b_ub, shift, tol=1e-9):
    """标准两段单纯形表（Bland 规则）。约束 A_ub·z ≤ b_ub，z ≥ 0。

    对 b_i≥0 的 ≤ 约束引入非负松弛变量 s_i（初始基可行）；对 b_i<0 的
    约束先乘 −1 化为 ≥ 约束，再引入剩余变量 s_i（系数 −1）与人工变量
    a_i（系数 +1，初始基）。阶段 I 极小化 w=Σ a_i：目标行以
    [0…0, 松弛/剩余, −1(人工), RHS=0] 初始化并对人工基行求和消去基列，
    检验数 <0 即入基；w*>0 判不可行。阶段 II 换入真实目标行后继续。
    表布局 [ 变量 n | 松弛/剩余 m | 人工 m | RHS ]。
    """
    m, n = A_ub.shape
    A_ub = A_ub.copy()
    b_ub = b_ub.copy()
    flipped = b_ub < -tol                       # 翻转后为 ≥ 约束
    A_ub[flipped] *= -1.0
    b_ub[flipped] *= -1.0
    artificial_rows = [i for i in range(m) if flipped[i]]

    width = n + 2 * m
    tab = np.zeros((m + 2, width + 1))
    tab[:m, :n] = A_ub
    basis = [-1] * m
    for i in range(m):
        tab[i, n + i] = -1.0 if flipped[i] else 1.0   # 剩余 / 松弛列
        tab[i, n + m + i] = 1.0                        # 人工列
        basis[i] = n + m + i if flipped[i] else n + i
        tab[i, -1] = b_ub[i]
    phase1_row, phase2_row = m, m + 1

    def pivot(row, col, obj_rows):
        tab[row, :] /= tab[row, col]
        for i in obj_rows:
            if i != row and abs(tab[i, col]) > tol:
                tab[i, :] -= tab[i, col] * tab[row, :]
        basis[row] = col

    def enter_any(obj_row, columns):
        """Bland：下标最小的负检验数列（不要求存在正比值）。"""
        for j in columns:
            if tab[obj_row, j] < -tol:
                return j
        return None

    def enter_feasible(obj_row, columns):
        """阶段 I 用：跳过在所有约束行均无正系数（无可入主元）的列。"""
        for j in columns:
            if tab[obj_row, j] < -tol and any(
                    tab[i, j] > tol for i in range(m)):
                return j
        return None

    def ratio_test(col):
        ratios = [(tab[i, -1] / tab[i, col], i) for i in range(m)
                  if tab[i, col] > tol]
        if not ratios:
            return None
        ratios.sort(key=lambda t: (t[0], basis[t[1]]))
        return ratios[0][1]

    # 阶段 I 允许在所有非人工列（原变量 + 松弛 + 剩余）中选入基列：剩余变量
    # （≥ 约束，列系数 −1）可在其余行为正，是把人工变量逐出基的必要桥梁；
    # enter 已跳过在所有约束行无正系数（无可行比值）的列。
    enter_columns = list(range(n + m))

    # ---- 阶段 I：min w = Σ 人工变量。目标行初值 w−Σa=0（人工列 −1），
    # 再**减去**各人工基行使基列归零：z 列检验数为负即入基、把人工变量
    # 逐出行基；w*>0 判不可行。 ----
    if artificial_rows:
        tab[phase1_row, n + m:n + 2 * m] = -1.0
        for i in artificial_rows:
            tab[phase1_row, :] -= tab[i, :]
        while True:
            col = enter_feasible(phase1_row, enter_columns)
            if col is None:
                break
            row = ratio_test(col)
            if row is None:
                return None, None, "unbounded"
            pivot(row, col, list(range(m + 2)))
        # 表中目标行为 −w：终止 RHS = −w*，w*>0（RHS<−tol）即不可行
        if tab[phase1_row, -1] < -1e-7:
            return None, None, "infeasible"
    # 人工变量仍在基（退化 0 值）：换出到任意可行非人工列
    for i, bv in enumerate(basis):
        if bv >= n + m:
            ncol = next((j for j in enter_columns
                         if abs(tab[i, j]) > tol), None)
            if ncol is not None:
                pivot(i, ncol, list(range(m)) + [phase2_row])
    assert all(bv < n + m for bv in basis), "阶段 I 后基仍含人工变量"

    # ---- 阶段 II：放入真实目标行并消去基列 ----
    cfull = np.zeros(n + 2 * m)
    cfull[:n] = c_vec
    tab[phase2_row, :width] = cfull
    tab[phase2_row, -1] = 0.0
    for i, bv in enumerate(basis):
        if bv < n + m and abs(tab[phase2_row, bv]) > tol:
            tab[phase2_row, :] -= tab[phase2_row, bv] * tab[i, :]
    while True:
        col = enter_any(phase2_row, enter_columns)
        if col is None:
            break
        row = ratio_test(col)
        if row is None:
            # 负检验数列在所有约束行无正系数：目标可无限改进（无界）
            return None, None, "unbounded"
        pivot(row, col, list(range(m)) + [phase2_row])

    x = np.zeros(n)
    for i, bv in enumerate(basis):
        if bv < n:
            x[bv] = tab[i, -1]
    x += shift
    val = -tab[phase2_row, -1] + float(np.asarray(c_vec[:n]) @ shift)
    return x, float(val), "optimal"


def variable_ranges(p_lower, p_upper, lo_lim, hi_lim, free_idx,
                    fixed_values=None, fixed_const_lo=None,
                    fixed_const_hi=None):
    """对每个自由工序变量求名义可行范围。

    闭环下界约束 ``P⁻·y_free + c_lo ≥ LSL``、上界约束
    ``P⁺·y_free + c_hi ≤ USL``（``p_lower``/``p_upper`` 只取自由列；
    固定边贡献由 ``fixed_const_lo/hi`` 给）。名义情形 P⁻=P⁺=P、
    c_lo=c_hi=P·fixed。WC 护栏情形 P⁻/P⁺ 取逐边偏差方向（固定边可单边）。

    返回 {free_col: (lo, hi, bounded)}；整体不可行抛 ProcessPlanError。
    """
    p_lower = np.asarray(p_lower, dtype=float)
    p_upper = np.asarray(p_upper, dtype=float)
    free_idx = list(free_idx)
    nf = len(free_idx)
    c_lo = (np.zeros(p_lower.shape[0]) if fixed_const_lo is None
            else np.asarray(fixed_const_lo, dtype=float))
    c_hi = (np.zeros(p_upper.shape[0]) if fixed_const_hi is None
            else np.asarray(fixed_const_hi, dtype=float))
    # -P⁻ z ≤ c_lo − LSL ；P⁺ z ≤ USL − c_hi
    A = np.vstack([-p_lower, p_upper])
    b = np.concatenate([c_lo - np.asarray(lo_lim, dtype=float),
                        np.asarray(hi_lim, dtype=float) - c_hi])

    def feasible():
        _, _, st = linprog_bounds(
            nf, np.zeros(nf), A, b, lower_bounds=np.full(nf, 1e-9))
        return st != "infeasible"

    if not feasible():
        raise ProcessPlanError(
            "锁定的工序尺寸与设计闭环要求矛盾：不存在满足全部设计闭环的"
            "反算解（请放宽锁定 / 能力档位或调整设计尺寸）")
    out: dict[int, tuple] = {}
    for k, col in enumerate(free_idx):
        c_lo_obj = np.zeros(nf); c_lo_obj[k] = 1.0
        c_hi_obj = np.zeros(nf); c_hi_obj[k] = -1.0
        lo_x, _, st_lo = linprog_bounds(
            nf, c_lo_obj, A, b, lower_bounds=np.full(nf, 1e-9))
        hi_x, _, st_hi = linprog_bounds(
            nf, c_hi_obj, A, b, lower_bounds=np.full(nf, 1e-9))
        unbounded = (st_lo == "unbounded" or st_hi == "unbounded")
        lo_v = -np.inf if st_lo == "unbounded" else float(lo_x[k])
        hi_v = np.inf if st_hi == "unbounded" else float(hi_x[k])
        out[col] = (lo_v, hi_v, not unbounded)
    return out


# ------------------------------------------------------------- 代数消元

def elimination_trace(p_design: np.ndarray,
                      targets_mid: np.ndarray,
                      edge_ids: list[str],
                      locked_mask: np.ndarray,
                      closure_ids: list[str],
                      op_mask: np.ndarray | None = None) -> dict:
    """对设计闭环名义方程 P·y = t 做高斯-若尔当消元（自由列优先主元）。

    返回 pivot/free 变量划分、逐步消元记录与把 pivot 变量表为自由变量
    仿射函数的代数表达式。目标取设计尺寸带中点 N+(ES+EI)/2（名义反算的
    居中解）；锁定列与毛坯尺寸列（op_mask=False）视为已知常数。
    """
    m, n = p_design.shape
    if op_mask is None:
        op_mask = np.ones(n, dtype=bool)
    # 已知量（不可反算）：锁定工序尺寸 + 全部毛坯尺寸
    fixed_known = locked_mask | (~op_mask)
    # 消元列序：自由（待反算工序）优先，其次已知列，保证主元落在待求量上
    col_order = [j for j in range(n) if not fixed_known[j]] + \
                [j for j in range(n) if fixed_known[j]]
    a = np.array(p_design, dtype=float)
    b = np.array(targets_mid, dtype=float)
    pivots: list[tuple[int, int]] = []
    steps: list[dict] = []
    row_by_pivot: dict[int, int] = {}
    r = 0
    for col in col_order:
        if r >= m:
            break
        cand = sorted(range(r, m), key=lambda i: -abs(a[i, col]))
        sel = next((i for i in cand if abs(a[i, col]) > 1e-8), None)
        if sel is None:
            continue
        if sel != r:
            a[[r, sel]] = a[[sel, r]]
            b[[r, sel]] = b[[sel, r]]
            steps.append({"action": "swap_rows",
                          "rows": [r + 1, sel + 1]})
        pivot_val = a[r, col]
        a[r, :] /= pivot_val
        b[r] /= pivot_val
        eliminated = []
        for i in range(m):
            if i != r and abs(a[i, col]) > 1e-12:
                factor = a[i, col]
                a[i, :] -= factor * a[r, :]
                b[i] -= factor * b[r]
                eliminated.append({"row": i + 1, "factor": round(float(factor), 9)})
        steps.append({
            "action": "eliminate",
            "closure_row": r + 1,
            "closure_id": closure_ids[r],
            "pivot_edge": edge_ids[col],
            "pivot_kind": ("待反算" if not fixed_known[col]
                           else ("锁定" if locked_mask[col] else "毛坯已知")),
            "normalized_row": [round(float(v), 9) for v in a[r, :]],
            "rhs": round(float(b[r]), 9),
            "eliminated_rows": eliminated,
        })
        pivots.append((r, col))
        row_by_pivot[col] = r
        r += 1

    pivot_cols = [c for _, c in pivots]
    free_cols = [j for j in range(n)
                 if j not in pivot_cols and not fixed_known[j]]
    fixed_cols = [j for j in range(n) if fixed_known[j]]

    # pivot 边的仿射表达式：y_pivot = rhs − Σ(自由/已知系数)·y_j
    expressions = []
    for rr, col in pivots:
        terms = []
        for j in free_cols:
            if abs(a[rr, j]) > 1e-9:
                terms.append({"edge": edge_ids[j], "role": "待反算自由度",
                              "coefficient": round(float(-a[rr, j]), 9)})
        for j in fixed_cols:
            if abs(a[rr, j]) > 1e-9:
                role = "锁定已知值" if locked_mask[j] else "毛坯尺寸已知值"
                terms.append({"edge": edge_ids[j], "role": role,
                              "coefficient": round(float(-a[rr, j]), 9)})
        expressions.append({
            "edge": edge_ids[col],
            "role": ("待反算" if not fixed_known[col]
                     else ("锁定主元" if locked_mask[col] else "毛坯主元")),
            "expression": f"{edge_ids[col]} = {round(float(b[rr]), 6)}"
                          + "".join(
                              (f" {'+' if t['coefficient'] >= 0 else '-'} "
                               f"{abs(t['coefficient']):g}·{t['edge']}")
                              for t in terms),
            "constant": round(float(b[rr]), 9),
            "terms": terms,
        })

    # 冗余行（消元后全零行）
    dependent = [closure_ids[i] for i in range(m)
                 if np.all(np.abs(a[i, :]) < 1e-8)]
    return {
        "method": "高斯-若尔当消元（Gauss-Jordan RREF，待反算列优先选主元）",
        "matrix_shape": [m, n],
        "rank": len(pivots),
        "degrees_of_freedom": len(free_cols),
        "pivot_edges": [edge_ids[c] for c in pivot_cols],
        "free_edges": [edge_ids[j] for j in free_cols],
        "fixed_edges": [edge_ids[j] for j in fixed_cols],
        "locked_edges": [edge_ids[j] for j in fixed_cols if locked_mask[j]],
        "blank_edges": [edge_ids[j] for j in fixed_cols if not op_mask[j]],
        "steps": steps,
        "expressions": expressions,
        "dependent_closures": dependent,
    }


# ------------------------------------------------------------- 误差传播

def _chain_for(plan: ProcessPlan, nominals, mids, halfs, sigmas,
               explicit_flags) -> engine.NormalizedChain:
    """把方案边伪装成单链维度以复用 copula 抽样（sign 恒 +1）。"""
    dims = []
    for e, nom, mid, half, sigma, flag in zip(
            plan.edges, nominals, mids, halfs, sigmas, explicit_flags):
        dims.append(engine.NormDimension(
            id=e.id, sign=1, nominal=float(nom),
            upper_dev=float(mid + half), lower_dev=float(mid - half),
            mid=float(mid), half_width=float(half), sigma=float(sigma),
            sigma_explicit=bool(flag), distribution=e.distribution,
            start=e.start, end=e.end, direction=1, unit=e.unit,
            original=e.original))
    return engine.NormalizedChain(
        dimensions=dims, closure_lsl_mm=None, closure_usl_mm=None,
        mc_samples=plan.mc_samples, seed=plan.seed, corr=plan.corr,
        name=plan.name)


def propagate(plan: ProcessPlan, nominals, halfs, sigmas, mids=None,
              explicit_flags=None, mc_samples: int | None = None,
              seed: int | None = None, p_mat: np.ndarray | None = None,
              dev_lo=None, dev_hi=None) -> dict:
    """对全部闭环传播 WC / RSS / 固定种子 MC。

    nominals/halfs/sigmas 按 plan.edges 顺序（mm）；闭环样本 C = Y·Pᵀ，
    每条边每轮只抽样一次，各闭环同源。dev_lo/dev_hi 给逐边非对称偏差
    （自由边 =∓half，锁定/毛坯边用提交 EI/ES）；缺省按 mid±half 对称。
    """
    n_e = len(plan.edges)
    nominals = np.asarray(nominals, dtype=float)
    halfs = np.asarray(halfs, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if mids is None:
        mids = np.array([e.mid for e in plan.edges])
    mids = np.asarray(mids, dtype=float)
    if dev_lo is None:
        dev_lo = mids - halfs
    if dev_hi is None:
        dev_hi = mids + halfs
    dev_lo = np.asarray(dev_lo, dtype=float)
    dev_hi = np.asarray(dev_hi, dtype=float)
    if explicit_flags is None:
        explicit_flags = [e.sigma_explicit for e in plan.edges]
    p = plan.transfer_matrix() if p_mat is None else p_mat
    p_pos, p_neg = np.maximum(p, 0.0), np.minimum(p, 0.0)
    c0 = p @ nominals
    mu = p @ (nominals + mids)
    wc_lo = p_pos @ (nominals + dev_lo) + p_neg @ (nominals + dev_hi)
    wc_hi = p_pos @ (nominals + dev_hi) + p_neg @ (nominals + dev_lo)
    band = (wc_hi - wc_lo) / 2.0
    cov_edge = plan.corr * (sigmas[:, None] * sigmas[None, :])
    closure_cov = p @ cov_edge @ p.T
    var_l = np.maximum(np.diag(closure_cov), 0.0)
    sigma_l = np.sqrt(var_l)

    n = mc_samples or plan.mc_samples
    used_seed = plan.seed if seed is None else seed
    nc = _chain_for(plan, nominals, mids, halfs, sigmas, explicit_flags)
    y = engine.sample_dimensions(
        nc, n, used_seed, sigmas=sigmas, mids=np.asarray(mids),
        halfs=halfs, explicit_flags=explicit_flags)
    closure_samples = y @ p.T

    results = []
    for li, cl in enumerate(plan.closures):
        col = closure_samples[:, li]
        q005, q995 = np.quantile(col, [0.005, 0.995])
        # WC 逐边公差贡献（取该边上下偏差半宽）
        wc_contrib = []
        for j, e in enumerate(plan.edges):
            if abs(p[li, j]) > EPS:
                edge_half = (dev_hi[j] - dev_lo[j]) / 2.0
                wc_contrib.append({
                    "edge_id": e.id,
                    "operation_id": e.op_id,
                    "kind": e.kind,
                    "coefficient": float(p[li, j]),
                    "lower_deviation_mm": float(dev_lo[j]),
                    "upper_deviation_mm": float(dev_hi[j]),
                    "half_width_mm": float(edge_half),
                    "tolerance_contribution_mm": float(
                        abs(p[li, j]) * edge_half),
                    "tolerance_share": (
                        float(abs(p[li, j]) * edge_half / band[li])
                        if band[li] > EPS else 0.0),
                })
        # RSS / MC 方差贡献（协方差法）
        rss_row = p[li] @ cov_edge
        mc_centered = col - col.mean()
        y_centered = y - y.mean(axis=0)
        cov_yc = (y_centered.T @ mc_centered) / (n - 1)
        rss_contrib, mc_contrib = [], []
        for j, e in enumerate(plan.edges):
            if abs(p[li, j]) <= EPS:
                continue
            vcont = float(p[li, j] * rss_row[j])
            rss_contrib.append({
                "edge_id": e.id, "operation_id": e.op_id, "kind": e.kind,
                "coefficient": float(p[li, j]),
                "sigma_mm": float(sigmas[j]),
                "variance_contribution_mm2": vcont,
                "variance_share": vcont / var_l[li] if var_l[li] > EPS else 0.0,
            })
            slope = float(cov_yc[j]) / var_l[li] if var_l[li] > EPS else 0.0
            mc_contrib.append({
                "edge_id": e.id, "operation_id": e.op_id, "kind": e.kind,
                "coefficient": float(p[li, j]),
                "regression_slope": slope,
                "variance_share": (
                    float(p[li, j] * cov_yc[j]) / var_l[li]
                    if var_l[li] > EPS else 0.0),
            })
        lsl, usl = cl.lsl_mm, cl.usl_mm
        wc_in = (wc_lo[li] >= lsl - EPS
                 and (usl is None or wc_hi[li] <= usl + EPS))
        results.append({
            "closure_id": cl.id,
            "kind": cl.kind,
            "note": cl.note,
            "operation_refs": cl.op_refs,
            "spec_mm": {"lower_limit": lsl, "upper_limit": usl},
            "nominal_mm": float(c0[li]),
            "mean_mm": float(mu[li]),
            "worst_case": {
                "lower_bound_mm": float(wc_lo[li]),
                "upper_bound_mm": (None if cl.kind == "stock"
                                   else float(wc_hi[li])),
                "total_tolerance_band_mm": float(2 * band[li]),
                "in_spec": bool(wc_in),
                "margin_mm": _margin(wc_lo[li], wc_hi[li], lsl, usl),
                "contributions": sorted(
                    wc_contrib, key=lambda d: -abs(d["tolerance_share"])),
            },
            "rss": {
                "sigma_mm": float(sigma_l[li]),
                "z": engine.Z_STAT,
                "lower_bound_mm": float(mu[li] - engine.Z_STAT * sigma_l[li]),
                "upper_bound_mm": float(mu[li] + engine.Z_STAT * sigma_l[li]),
                "in_spec": bool(
                    mu[li] - engine.Z_STAT * sigma_l[li] >= lsl - EPS
                    and (usl is None
                         or mu[li] + engine.Z_STAT * sigma_l[li] <= usl + EPS)),
                "reject_probability": _reject(mu[li], sigma_l[li], lsl, usl),
                "contributions": sorted(
                    rss_contrib, key=lambda d: -abs(d["variance_share"])),
            },
            "monte_carlo": {
                "samples": int(n),
                "random_seed": int(used_seed),
                "mean_mm": float(col.mean()),
                "sigma_mm": float(col.std(ddof=1)),
                "quantile_0p5_mm": float(q005),
                "quantile_99p5_mm": float(q995),
                "min_sample_mm": float(col.min()),
                "max_sample_mm": float(col.max()),
                "in_spec": bool(
                    col.min() >= lsl - EPS
                    and (usl is None or col.max() <= usl + EPS)),
                "reject_probability": _empirical(col, lsl, usl),
                "contributions": sorted(
                    mc_contrib, key=lambda d: -abs(d["variance_share"])),
            },
        })
    return {
        "closures": results,
        "monte_carlo": {"samples": int(n), "seed": int(used_seed),
                        "shared_sampling": "全部工序/毛坯边每轮只抽样一次，"
                        "闭环样本 C = Y·Pᵀ 共享同轮取值"},
    }


def _margin(wc_lo, wc_hi, lsl, usl):
    m_lo = wc_lo - lsl
    if usl is None:
        return float(m_lo)
    return float(min(m_lo, usl - wc_hi))


def _reject(mean, sigma, lsl, usl):
    if usl is None:
        if sigma <= 0:
            return 0.0 if mean >= lsl else 1.0
        return float(engine.normal_cdf((lsl - mean) / sigma))
    return engine._reject_normal(float(mean), float(sigma), lsl, usl)


def _empirical(samples, lsl, usl):
    mask = samples < lsl
    if usl is not None:
        mask |= samples > usl
    return float(mask.mean())


# ------------------------------------------------------------- 反算求解

@dataclass
class SolveRequest:
    locked: dict[str, float | None]      # edge_id -> 锁定名义（None=用提交值）
    grades: dict[str, float]             # 档位 id -> IT 公差系数 K（T=K·∛N）
    grade_costs: dict[str, float]        # 档位 id -> 每道选用该档位工序的附加成本
    default_grade: str
    edge_grades: dict[str, str]          # 待反算边 -> 唯一档位（限定）
    edge_grade_list: dict[str, list[str]]  # 待反算边 -> 可选档位子集
    step_sizes: list[float]              # 标准尺寸步进（mm）
    edge_steps: dict[str, float]
    mc_samples: int
    seed: int
    max_candidates: int


def _snap_step(value: float, step: float) -> float:
    """吸到最近的标准步进倍数（≥ step 本身）。"""
    k = max(1, round(value / step))
    return round(k * step, 9)


def solve_plan(plan: ProcessPlan, req: SolveRequest) -> dict:
    """锁定部分工序尺寸、反算其余，枚举档位×步进多解并排序。"""
    ids = plan.ids
    n = len(plan.edges)
    op_mask = np.array([e.kind == "operation" for e in plan.edges])
    locked_mask = np.zeros(n, dtype=bool)
    nominal0 = np.array([e.nominal for e in plan.edges])
    locked_nom = nominal0.copy()
    for eid, val in req.locked.items():
        if eid not in plan._by_id:
            raise ProcessPlanError(f"锁定尺寸 {eid!r} 不在方案中：{ids}")
        j = ids.index(eid)
        if plan.edges[j].kind != "operation":
            raise ProcessPlanError(
                f"毛坯尺寸 {eid} 由毛坯图固定，不能作为工序锁定项；"
                "锁定仅适用于工序尺寸")
        locked_mask[j] = True
        if val is not None:
            locked_nom[j] = to_mm(val, plan.edges[j].unit)
    free_idx = [j for j in range(n) if op_mask[j] and not locked_mask[j]]
    if not free_idx:
        # 全部锁定：直接校核，不做多解枚举
        return _single_candidate(plan, locked_nom, locked_mask, req)

    p = plan.transfer_matrix()
    design_rows = [i for i, c in enumerate(plan.closures)
                   if c.kind == "design"]
    stock_rows = [i for i, c in enumerate(plan.closures)
                  if c.kind == "stock"]
    pd_ = p[design_rows, :]
    lsl = np.array([c.lsl_mm for c in plan.closures])
    usl = np.array([c.usl_mm if c.usl_mm is not None else np.inf
                    for c in plan.closures])
    t_mid = np.array([c.nominal + (c.upper_dev + c.lower_dev) / 2.0
                      for c in plan.closures])
    d_mid = np.array([t_mid[i] for i in design_rows])
    d_lsl = np.array([lsl[i] for i in design_rows])
    d_usl = np.array([usl[i] for i in design_rows])

    # ---- 名义可行范围（先不带公差护栏，得到理论范围） ----
    # 非自由列（锁定工序 + 毛坯）名义作为已知常数，自由列位置填 0
    fixed_nominal = nominal0.copy()
    fixed_nominal[free_idx] = 0.0
    pf_nom = pd_[:, free_idx]
    fixed_const_nom = pd_ @ fixed_nominal
    ranges = variable_ranges(
        pf_nom, pf_nom, d_lsl, d_usl, free_idx,
        fixed_const_lo=fixed_const_nom, fixed_const_hi=fixed_const_nom)
    unbounded = [(plan.edges[j].op_id, plan.edges[j].machined)
                 for j in free_idx if not ranges[j][2]]
    if unbounded:
        nullity = _design_nullity(pd_, locked_mask, free_idx)
        raise ProcessPlanError(
            "传递矩阵对锁定后的待反算工序秩不足（自由度 "
            f"{nullity}）：以下工序名义值无上界或下界，设计闭环无法确定其"
            "尺寸——" + "；".join(f"工序 {op}（被加工面 {sf}）"
                             for op, sf in unbounded)
            + "。请锁定该工序尺寸，或补充连接该表面的余量闭环")

    # ---- 代数消元（名义方程，目标取带中点） ----
    trace = elimination_trace(
        pd_, d_mid, ids, locked_mask,
        [plan.closures[i].id for i in design_rows], op_mask=op_mask)

    # ---- 档位 × 步进 候选枚举 ----
    grade_options = _grade_options(plan, free_idx, req)
    grade_combos = _grade_combinations(grade_options)
    halfs0 = np.array([e.half_width for e in plan.edges])
    sigmas0 = np.array([e.sigma for e in plan.edges])
    # 自由边按对称偏差带（mid=0）；锁定工序/毛坯边保留提交公差带中点
    # （单边公差的带中点偏移按原值计入闭环均值）。
    mids_fixed = np.array([e.mid for e in plan.edges])

    scored: list[dict] = []
    for gi, combo in enumerate(grade_combos):
        halfs = halfs0.copy()
        sigmas = sigmas0.copy()
        flags = [e.sigma_explicit for e in plan.edges]
        grade_record = {}
        for j in free_idx:
            e = plan.edges[j]
            gid, k_factor = combo[j]
            half_est = _grade_half(nominal0[j], k_factor)
            halfs[j] = half_est
            sigmas[j] = engine._theoretical_sigma(e.distribution, half_est)
            flags[j] = False
            grade_record[e.id] = {
                "grade": gid, "factor": k_factor,
                "half_width_mm_estimate": half_est}
        # WC 护栏：固定边保留原始（可单边）偏差带，自由边以对称半宽预留。
        # 下界约束用 P⁻（正系数取下偏差方向）、上界用 P⁺（正系数取上偏差）。
        pf = pd_[:, free_idx]
        fixed_lo = fixed_nominal.copy()
        fixed_hi = fixed_nominal.copy()
        for j in range(n):
            if j not in free_idx:
                fixed_lo[j] = nominal0[j] + plan.edges[j].lower_dev
                fixed_hi[j] = nominal0[j] + plan.edges[j].upper_dev
        c_lo = pd_ @ fixed_lo - np.abs(pf) @ halfs[free_idx]
        c_hi = pd_ @ fixed_hi + np.abs(pf) @ halfs[free_idx]
        try:
            rng_guard = variable_ranges(
                pf, pf, d_lsl, d_usl, free_idx,
                fixed_const_lo=c_lo, fixed_const_hi=c_hi)
        except ProcessPlanError:
            continue
        grid = _nominal_grid(plan, free_idx, rng_guard, req, nominal0)
        if grid is None:
            continue
        for nominals in grid:
            nom_full = fixed_nominal.copy()
            # 用最终名义修正档位半宽（T=K·∛N）
            h_final = halfs.copy()
            s_final = sigmas.copy()
            m_final = mids_fixed.copy()
            d_lo = np.zeros(n)
            d_hi = np.zeros(n)
            for j in free_idx:
                h_final[j] = _grade_half(nominals[j], combo[j][1])
                s_final[j] = engine._theoretical_sigma(
                    plan.edges[j].distribution, h_final[j])
                m_final[j] = 0.0            # 自由边对称偏差带
                d_lo[j], d_hi[j] = -h_final[j], h_final[j]
                nom_full[j] = nominals[j]
            # 提交值（锁定/毛坯）边保留原 mid/half/sigma
            d_lo = m_final - h_final
            d_hi = m_final + h_final
            for j in range(n):
                if not op_mask[j] or locked_mask[j]:
                    h_final[j] = halfs0[j]
                    s_final[j] = sigmas0[j]
                    m_final[j] = mids_fixed[j]
                    d_lo[j] = plan.edges[j].lower_dev
                    d_hi[j] = plan.edges[j].upper_dev
            wc = _wc_check(p, nom_full, d_lo, d_hi, lsl, usl,
                           design_rows, stock_rows)
            if wc is None:
                continue
            change = sum(abs(nom_full[j] - nominal0[j])
                         for j in free_idx)
            total_cost = _total_cost(
                plan, h_final, locked_mask, free_idx, combo, req)
            scored.append({
                "grade_combo_index": gi,
                "combo": dict(combo),
                "nominals": nom_full.copy(),
                "halfs": h_final, "sigmas": s_final,
                "mids": m_final, "flags": flags,
                "grade_record": grade_record,
                "manufacturing_cost": total_cost,
                "change": change,
                "wc": wc,
            })
            if len(scored) >= MAX_NOMINAL_GRID_TOTAL:
                break
        if len(scored) >= MAX_NOMINAL_GRID_TOTAL:
            break

    if not scored:
        return {
            "status": "infeasible",
            "locked_edges": [ids[j] for j in range(n) if locked_mask[j]],
            "free_edges": [ids[j] for j in free_idx],
            "nominal_ranges": _ranges_payload(plan, ranges),
            "elimination": trace,
            "candidates": [],
            "message": "给定能力档位与步进下没有满足全部闭环 WC 边界的组合；"
                       "名义可行范围已给出，可放宽档位公差或调整步进",
            "enumeration": {
                "grade_combinations": len(grade_combos),
                "wc_feasible_states": 0,
                "mc_verified": 0,
                "limits": {"grade": MAX_GRADE_COMBINATIONS,
                           "grid": MAX_NOMINAL_GRID_TOTAL},
            },
        }

    # ---- 排序：设计闭环达标数 → 最差余量 → 改动量 → 成本 ----
    n_design = len(design_rows)

    def sort_key(it):
        wc = it["wc"]
        design_pass = sum(1 for i in design_rows
                          if wc["in_spec"][i])
        worst = min(wc["margin"][i] for i in range(len(plan.closures)))
        return (-design_pass, -worst, it["change"], it["manufacturing_cost"])

    scored.sort(key=sort_key)
    top = scored[:TOP_CANDIDATES_MC]

    candidates = []
    for rank, it in enumerate(top, start=1):
        prop = propagate(
            plan, it["nominals"], it["halfs"], it["sigmas"],
            mids=it["mids"], explicit_flags=it["flags"],
            mc_samples=req.mc_samples, seed=req.seed, p_mat=p)
        design_pass = sum(
            1 for i, clr in zip(range(len(plan.closures)), prop["closures"])
            if i in design_rows and clr["worst_case"]["in_spec"])
        worst_margin = min(
            clr["worst_case"]["margin_mm"] for clr in prop["closures"])
        worst_stock = min(
            (clr["worst_case"]["margin_mm"]
             for i, clr in enumerate(prop["closures"]) if i in stock_rows),
            default=None)
        candidates.append(_candidate_payload(
            plan, rank, it, prop, locked_mask, free_idx, req,
            design_pass, worst_margin, worst_stock, nominal0))

    # 参考：提交名义的基线校核（不锁定时仅用于对比改动量）
    return {
        "status": "optimal",
        "locked_edges": [ids[j] for j in range(n) if locked_mask[j]],
        "free_edges": [ids[j] for j in free_idx],
        "nominal_ranges": _ranges_payload(plan, ranges),
        "elimination": trace,
        "ranking": [
            "设计闭环 WC 达标数（多者优先）",
            "最差余量（所有闭环 WC 余量最小值，大者优先）",
            "尺寸改动量 Σ|反算名义−提交名义|（小者优先）",
            "制造成本（低者优先）",
        ],
        "mc_seed": req.seed,
        "candidates": candidates[:req.max_candidates],
        "enumeration": {
            "grade_combinations": len(grade_combos),
            "wc_feasible_states": len(scored),
            "mc_verified": len(top),
            "limits": {"grade": MAX_GRADE_COMBINATIONS,
                       "grid": MAX_NOMINAL_GRID_TOTAL},
        },
    }


def _single_candidate(plan, locked_nom, locked_mask, req: SolveRequest) -> dict:
    """全部工序尺寸锁定：直接按提交公差做一次 WC/RSS/MC 校核。"""
    n = len(plan.edges)
    halfs0 = np.array([e.half_width for e in plan.edges])
    sigmas0 = np.array([e.sigma for e in plan.edges])
    p = plan.transfer_matrix()
    design_rows = [i for i, c in enumerate(plan.closures)
                   if c.kind == "design"]
    prop = propagate(plan, locked_nom, halfs0, sigmas0,
                     mc_samples=req.mc_samples, seed=req.seed, p_mat=p)
    it = {"nominals": locked_nom, "halfs": halfs0, "sigmas": sigmas0,
          "mids": np.array([e.mid for e in plan.edges]),
          "flags": [e.sigma_explicit for e in plan.edges],
          "manufacturing_cost": sum(e.manufacturing_cost for e in plan.edges),
          "change": 0.0, "grade_record": {}, "combo": {}}
    worst = min(c["worst_case"]["margin_mm"] for c in prop["closures"])
    payload = _candidate_payload(
        plan, 1, it, prop, locked_mask,
        [j for j in range(n) if plan.edges[j].kind == "operation"
         and not locked_mask[j]], req,
        sum(1 for i in design_rows
            if prop["closures"][i]["worst_case"]["in_spec"]),
        worst, None, locked_nom)
    payload["all_locked"] = True
    return {"status": "optimal", "locked_edges": plan.ids, "free_edges": [],
            "nominal_ranges": [], "elimination": None,
            "ranking": ["全部工序尺寸已锁定，无多解；直接校核"],
            "mc_seed": req.seed, "candidates": [payload],
            "enumeration": {"grade_combinations": 0, "wc_feasible_states": 1,
                            "mc_verified": 1, "limits": {}}}


def _grade_half(nominal: float, factor: float) -> float:
    """IT 口径公差半宽 T = K·∛N（N 单位 mm，K 为档位系数）。"""
    return factor * (max(nominal, 1e-6) ** (1.0 / 3.0)) / 2.0


def edge_cost(e: Edge, new_half: float | None, locked: bool) -> float:
    """单边制造成本（mm 口径无关，仅按相对公差带缩放）。

    锁定工序边 / 毛坯边取提交成本；待反算边按能力档位收紧程度加价：
    cost = base·T_submit/T_new + 档位附加成本（越紧越贵）。
    """
    if locked or new_half is None:
        return float(e.manufacturing_cost)
    ratio = (e.half_width / new_half
             if e.half_width > EPS and new_half > EPS else 1.0)
    return float(e.manufacturing_cost) * ratio


def _grade_options(plan, free_idx, req: SolveRequest) -> dict[int, list[tuple]]:
    """每条待反算边可选的 (档位 id, 系数)：边级限定 > 全部声明档位。"""
    options: dict[int, list[tuple]] = {}
    for j in free_idx:
        e = plan.edges[j]
        if e.id in req.edge_grades:
            gids = [req.edge_grades[e.id]]
        elif e.id in req.edge_grade_list:
            gids = list(req.edge_grade_list[e.id])
        else:
            gids = list(req.grades.keys())          # 默认在全部档位中选
        for gid in gids:
            if gid not in req.grades:
                raise ProcessPlanError(
                    f"工序 {e.op_id} 使用的能力档位 {gid!r} 未在 "
                    f"capability_tiers 中声明；可用档位: {sorted(req.grades)}")
        options[j] = sorted((gid, float(req.grades[gid])) for gid in gids)
    return options


def _grade_combinations(grade_options: dict[int, list[tuple]]):
    """各自由边档位选择的笛卡尔积（combo: edge_col -> (gid, factor)）。"""
    combos = [{}]
    for j in sorted(grade_options):
        combos = [{**c, j: opt} for c in combos for opt in grade_options[j]]
    if len(combos) > MAX_GRADE_COMBINATIONS:
        raise ProcessPlanError(
            f"档位组合数 {len(combos)} 超过上限 {MAX_GRADE_COMBINATIONS}："
            f"{len(grade_options)} 个待反算工序，请用 edge_grades 限定可选"
            "档位或锁定更多工序尺寸")
    return combos


def _total_cost(plan, halfs, locked_mask, free_idx, combo, req):
    """候选方案的总制造成本。

    锁定/毛坯边 = 提交成本；待反算边 = base·T_submit/T_new + 档位附加成本。
    """
    total = 0.0
    for j, e in enumerate(plan.edges):
        if j in free_idx:
            gid = combo[j][0]
            total += edge_cost(e, halfs[j], False) \
                + float(req.grade_costs.get(gid, 0.0))
        else:
            total += e.manufacturing_cost
    return float(total)


def _nominal_grid(plan, free_idx, ranges, req, nominal0):
    """在 LP 可行范围内按标准步进构造名义网格（每边取若干点，总量受控）。"""
    per_edge_points = max(2, int(MAX_NOMINAL_GRID_TOTAL **
                                 (1.0 / max(len(free_idx), 1))))
    per_edge_points = min(per_edge_points, 12)
    grids = []
    for j in free_idx:
        lo, hi, bounded = ranges[j]
        if not bounded or hi - lo < -1e-9:
            return None
        step = req.edge_steps.get(plan.edges[j].id,
                                  req.step_sizes[0] if req.step_sizes else 1.0)
        lo = max(lo, step)
        center = min(max(nominal0[j], lo), hi)
        pts = {_snap_step(center, step), _snap_step(lo, step),
               _snap_step(hi, step)}
        # 在范围内均匀补点
        for frac in np.linspace(0.0, 1.0, per_edge_points):
            v = lo + frac * (hi - lo)
            sv = _snap_step(float(v), step)
            if lo - 1e-7 <= sv <= hi + 1e-7:
                pts.add(sv)
        pts = sorted(p for p in pts if lo - 1e-7 <= p <= hi + 1e-7)
        if not pts:
            return None
        grids.append(sorted(pts))
    combos = [[]]
    for pts in grids:
        combos = [c + [v] for c in combos for v in pts]
    if len(combos) > MAX_NOMINAL_GRID_TOTAL:
        # 每边等距抽样到受控规模
        keep = max(
            2, int(MAX_NOMINAL_GRID_TOTAL ** (1.0 / len(free_idx))))
        grids = [g[::max(1, len(g) // keep)][:keep] for g in grids]
        combos = [[]]
        for pts in grids:
            combos = [c + [v] for c in combos for v in pts]
    out = []
    for combo in combos:
        arr = {j: v for j, v in zip(free_idx, combo)}
        out.append(arr)
    return out


def _wc_check(p, nominals, dev_lo, dev_hi, lsl, usl,
              design_rows, stock_rows):
    """极值法边界校核（逐边非对称偏差）；任一设计闭环越界返回 None。

    闭环最小值取 Σ P⁺·(N+EI) + P⁻·(N+ES)，最大值反之（P⁺=max(P,0)、
    P⁻=min(P,0)）。自由边 EI=−T、ES=+T，锁定/毛坯边用提交偏差。
    """
    p_pos = np.maximum(p, 0.0)
    p_neg = np.minimum(p, 0.0)
    lo = p_pos @ (nominals + dev_lo) + p_neg @ (nominals + dev_hi)
    hi = p_pos @ (nominals + dev_hi) + p_neg @ (nominals + dev_lo)
    n = p.shape[0]
    in_spec = np.zeros(n, dtype=bool)
    margin = np.zeros(n)
    for i in range(n):
        in_spec[i] = (lo[i] >= lsl[i] - EPS
                      and (usl[i] == np.inf or hi[i] <= usl[i] + EPS))
        margin[i] = (lo[i] - lsl[i] if usl[i] == np.inf
                     else min(lo[i] - lsl[i], usl[i] - hi[i]))
    if not in_spec[design_rows].all():
        return None
    return {"in_spec": in_spec, "margin": margin, "lo": lo, "hi": hi}


def _design_nullity(pd_, locked_mask, free_idx):
    """锁定后设计矩阵在自由列上的零空间维数（欠约束自由度数）。"""
    sub = pd_[:, free_idx]
    return len(free_idx) - int(np.linalg.matrix_rank(sub, tol=EPS))


def _ranges_payload(plan, ranges):
    return [
        {"edge_id": plan.edges[j].id,
         "operation_id": plan.edges[j].op_id,
         "machined_surface": plan.edges[j].machined,
         "nominal_lower_mm": (None if np.isinf(lo) else round(lo, 6)),
         "nominal_upper_mm": (None if np.isinf(hi) else round(hi, 6)),
         "bounded": bool(bounded)}
        for j, (lo, hi, bounded) in sorted(ranges.items())
    ]


def _candidate_payload(plan, rank, it, prop, locked_mask, free_idx, req,
                       design_pass, worst_margin, worst_stock, nominal0):
    p = plan.transfer_matrix()
    n = len(plan.edges)
    op_rows = []
    for j, e in enumerate(plan.edges):
        if e.kind != "operation":
            continue
        locked = bool(locked_mask[j])
        gid = None if locked else it["grade_record"].get(e.id, {}).get("grade")
        base_cost = edge_cost(e, None if locked else it["halfs"][j], locked)
        tier_add = (0.0 if locked or gid is None
                    else float(req.grade_costs.get(gid, 0.0)))
        op_rows.append({
            "operation_id": e.op_id,
            "edge": [e.start, e.end],
            "datum_surface": e.datum,
            "machined_surface": e.machined,
            "locked": locked,
            "nominal_mm": round(float(it["nominals"][j]), 6),
            "upper_deviation_mm": round(float(it["mids"][j] + it["halfs"][j]), 6),
            "lower_deviation_mm": round(float(it["mids"][j] - it["halfs"][j]), 6),
            "half_width_mm": round(float(it["halfs"][j]), 6),
            "sigma_mm": round(float(it["sigmas"][j]), 6),
            "distribution": e.distribution,
            "nominal_change_mm": (0.0 if locked else
                                 round(float(it["nominals"][j] - nominal0[j]), 6)),
            "grade": gid,
            "manufacturing_cost": round(base_cost + tier_add, 6),
        })
    cl_summary = []
    for clr in prop["closures"]:
        cl_summary.append({
            "closure_id": clr["closure_id"],
            "kind": clr["kind"],
            "nominal_mm": clr["nominal_mm"],
            "wc_in_spec": clr["worst_case"]["in_spec"],
            "wc_margin_mm": clr["worst_case"]["margin_mm"],
            "rss_sigma_mm": clr["rss"]["sigma_mm"],
            "rss_reject_probability": clr["rss"]["reject_probability"],
            "mc_reject_probability": clr["monte_carlo"]["reject_probability"],
            "mc_quantile_mm": [clr["monte_carlo"]["quantile_0p5_mm"],
                               clr["monte_carlo"]["quantile_99p5_mm"]],
        })
    return {
        "rank": rank,
        "design_closures_total": sum(
            1 for c in plan.closures if c.kind == "design"),
        "design_closures_in_spec_wc": int(design_pass),
        "worst_margin_mm": round(float(worst_margin), 6),
        "worst_stock_margin_mm": (
            None if worst_stock is None else round(float(worst_stock), 6)),
        "nominal_change_total_mm": round(float(it["change"]), 6),
        "manufacturing_cost_total": round(float(it["manufacturing_cost"]), 6),
        "operations": op_rows,
        "closures": cl_summary,
        "closure_detail": prop["closures"],
        "error_propagation": {
            "method_note": "WC: μ±Σ|P|T；RSS: P(DσDρ)Pᵀ；MC: 全边共享抽样 "
                           "C=Y·Pᵀ；逐项贡献见 closure_detail 各方法 contributions",
            "monte_carlo": prop["monte_carlo"],
        },
    }
