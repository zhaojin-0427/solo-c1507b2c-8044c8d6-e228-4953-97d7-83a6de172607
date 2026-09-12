"""选择性装配：按实测尺寸在零件池间做合格组合匹配。

业务口径
========
* 调用方从同一基线链的多个**冻结检验批次**取数，把链上尺寸分组映射为
  零件池（尺寸恰好划分链，不漏不重）；同一池内多个尺寸必须来自同一
  工件序号，实例键为 ``(batch_id, serial)``。
* 池内实例必须测齐该池映射的全部尺寸；缺测实例在 ``excluded_instances``
  中注明缺口后排除，不参与求解。
* 偏倚修正沿用实例来源批次引用的测量方案：``x_c = x + b``；无方案的
  批次不修正、不确定度为 0。
* 一个组合由每个池各出一个实例构成。封闭环间隙
  ``G = Σ_p Σ_{i∈p} s_i·x_c,i``。

量具不确定度传播
----------------
同一工件实例内（同一批次、同一测量方案）按方案 ρ 矩阵传播共用量具相关；
**不同工件实例之间测量误差相互独立**（共用量具 ρ 只刻画同一测量行内
不同尺寸的误差相关，不跨工件）：

    u_C² = Σ_实例 [ Σ_i s_i²u_res,i²
                    + Σ_{i,j∈实例} s_i s_j ρ_ij u_rest,i u_rest,j ]

扩展不确定度 U = k_out·u_C；保护带 w = h·U（multiple）或固定长度。
组合判定与检验批次同一套双边保护带规则：合格 / 不确定 / 不合格。

求解目标（字典序）
------------------
1. 最大化合格（accept）装配数（直到装配数量）；
2. 中心偏差总和更小：Σ|G − 目标中心|；
3. 最差保护余量更大：min_assembly min(G−(LSL+w), (USL−w)−G)；
4. 跨批次数更少：Σ(组合内不同来源批次数 − 1)。

组合枚举与集合装箱均为 NP 难，实现为确定性分支定界（按池 0 最小剩余
实例锚定消除排列对称、逐层规则剪枝、字典序界剪枝）；超节点预算时
回退到同序贪心并在 ``solver.exact=false`` 标明。

快照冻结：每个任务版本冻结来源批次行、测量方案组合结果、保护带设置
与随机种子，历史版本后续读取不变。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .engine import NormalizedChain, _cholesky_psd
from .measurement import _decide
from .schemas import GuardBandSpec
from .units import to_mm

# 组合枚举上限与求解节点预算（超出后截断 / 贪心回退，结果中标明）
COMBO_ENUM_CAP = 2_000_000
SOLVER_NODE_BUDGET = 200_000

ASSEMBLY_FORMULAS = [
    "实例偏倚修正：x_c = x + b（b 取实例来源批次引用测量方案的偏倚修正；"
    "批次未引用方案时 b=0、不确定度为 0）",
    "组合封闭环间隙：G = Σ_p Σ_{i∈池p} s_i·x_c,i（s_i 为闭环方向系数）",
    "实例内量具不确定度按方案 ρ 传播（共用量具），不同工件实例相互独立："
    "u_C² = Σ_实例[Σ_i s_i²u_res,i² "
    "+ Σ_{i,j∈实例} s_i s_j ρ_ij u_rest,i u_rest,j]",
    "扩展不确定度 U = k_out·u_C（k_out 为任务输出覆盖因子，默认 2.0）",
    "保护带：mode=multiple 时 w = h·U；mode=fixed 时 w 为固定长度",
    "判定：LSL+w ≤ G ≤ USL−w → 合格 accept；G < LSL−w 或 G > USL+w → "
    "不合格 reject；其余 → 不确定 indeterminate",
    "中心偏差：|G − 目标中心|，目标中心 = (目标间隙下限+上限)/2",
    "保护余量：min(G−(LSL+w), (USL−w)−G)，仅合格组合为正",
    "跨批次数：组合内不同来源批次数 − 1；总跨批次数为各装配之和",
    "求解字典序：合格装配数最大 → 中心偏差总和最小 → 最差保护余量最大 → "
    "总跨批次数最小；每个实例全局只能使用一次",
    "固定种子蒙特卡洛：按实例独立生成测量误差列（实例内方案 ρ 相关，"
    "SeedSequence 按 (批次,序号) 派生独立子流），组合误差 = 各实例列之和",
]


# ------------------------------------------------------------- 实例构建

class AssemblyError(ValueError):
    """创建期可定位的校验错误（端点转 422）。"""


def _plan_lookup(plan_combined: dict | None) -> dict[str, dict[str, Any]]:
    if not plan_combined:
        return {}
    return {p["dimension_id"]: p for p in plan_combined["dimensions"]}


def _plan_corr(plan_combined: dict | None,
               dims: list[str]) -> np.ndarray | None:
    """取方案相关矩阵在给定尺寸（实例所测池尺寸）上的主子阵。"""
    if not plan_combined:
        return None
    order = plan_combined["correlation"]["dimension_order"]
    full = np.asarray(plan_combined["correlation"]["matrix"], dtype=float)
    idx = [order.index(d) for d in dims]
    return full[np.ix_(idx, idx)]


def build_pools(nc: NormalizedChain, batch_rows: list[Any],
                payload) -> dict[str, Any]:
    """校验池映射并从冻结批次构建各池实例 / 缺测排除清单。

    batch_rows 为数据库 InspectionBatchRow（必须已确认同链）。
    """
    chain_dims = [d.id for d in nc.dimensions]
    dim_sign = {d.id: d.sign for d in nc.dimensions}

    # ---- 池映射：名称唯一；尺寸恰好划分链上全部尺寸
    pool_names = [p.name for p in payload.pools]
    if len(set(pool_names)) != len(pool_names):
        dup = sorted({n for n in pool_names if pool_names.count(n) > 1})
        raise AssemblyError(f"零件池名称重复: {dup}")
    mapped: list[str] = []
    for p in payload.pools:
        if len(set(p.dimensions)) != len(p.dimensions):
            dup = sorted({d for d in p.dimensions
                          if p.dimensions.count(d) > 1})
            raise AssemblyError(f"零件池 {p.name} 内尺寸重复映射: {dup}")
        mapped.extend(p.dimensions)
    missing = [d for d in chain_dims if d not in mapped]
    external = sorted({d for d in mapped if d not in chain_dims})
    dup = sorted({d for d in mapped if mapped.count(d) > 1})
    problems = []
    if missing:
        problems.append(f"链上尺寸漏映射: {missing}")
    if dup:
        problems.append(f"尺寸被重复映射到多个池: {dup}")
    if external:
        problems.append(f"池映射引用了基线链之外的尺寸: {external}")
    if problems:
        raise AssemblyError("；".join(problems))

    pools: list[dict[str, Any]] = []
    for spec in payload.pools:
        dims = list(spec.dimensions)
        signs = np.array([dim_sign[d] for d in dims], dtype=float)
        instances: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []

        for brow in batch_rows:
            plan_by = _plan_lookup(
                brow.measurement_json.get("plan_snapshot", {}).get("combined")
                if brow.measurement_json else None)
            corr_sub = _plan_corr(
                brow.measurement_json.get("plan_snapshot", {}).get("combined")
                if brow.measurement_json else None, dims)
            for row in brow.rows_json:
                serial = row["serial"]
                by_dim = {m["dimension_id"]: m for m in row["measurements"]}
                miss = [d for d in dims
                        if d not in by_dim or by_dim[d]["value"] is None]
                if miss:
                    excluded.append({
                        "pool": spec.name,
                        "batch_id": brow.id,
                        "batch_name": brow.name,
                        "serial": serial,
                        "reason": "该工件未测齐池映射的全部尺寸，"
                                  "按缺测实例排除",
                        "missing_dimension_ids": miss,
                    })
                    continue
                corrected: dict[str, float] = {}
                measured: dict[str, dict[str, Any]] = {}
                u_res = np.zeros(len(dims))
                u_rest = np.zeros(len(dims))
                for j, d in enumerate(dims):
                    m = by_dim[d]
                    x_mm = to_mm(m["value"], m["unit"])
                    pd = plan_by.get(d)
                    bias = float(pd["bias_correction_mm"]) if pd else 0.0
                    corrected[d] = x_mm + bias
                    measured[d] = {
                        "value": m["value"], "unit": m["unit"],
                        "value_mm": x_mm,
                        "bias_correction_mm": bias,
                        "corrected_mm": x_mm + bias,
                        "measurement_plan_id": (
                            brow.measurement_plan_id
                            if pd is not None else None),
                    }
                    if pd is not None:
                        u_res[j] = pd["components_mm"]["u_resolution"]
                        u_rest[j] = pd["u_rest_mm"]
                xc = np.array([corrected[d] for d in dims], dtype=float)
                # 实例内封闭环传播方差（分辨率独立 + ρ 相关的 rest）
                var = float((signs * u_res) @ (signs * u_res))
                if corr_sub is not None and u_rest.any():
                    wr = signs * u_rest
                    var += float((corr_sub * wr[:, None] * wr[None, :]).sum())
                instances.append({
                    "pool": spec.name,
                    "batch_id": brow.id,
                    "batch_name": brow.name,
                    "serial": serial,
                    "dimensions": dims,
                    "measured": measured,
                    "corrected_mm": corrected,
                    "gap_part": float(xc @ signs),
                    "u2": max(var, 0.0),
                    "signs": signs.tolist(),
                    "u_res_mm": u_res.tolist(),
                    "u_rest_mm": u_rest.tolist(),
                    "has_plan": corr_sub is not None and bool(u_rest.any()),
                    "corr": corr_sub.tolist() if corr_sub is not None else None,
                })

        pools.append({
            "name": spec.name,
            "dimensions": dims,
            "instances": instances,
            "excluded": excluded,
            "batch_ids": sorted({i["batch_id"] for i in instances}),
        })

    return {"pools": pools}


# ------------------------------------------------------------- 规则校验

def _expand_forbidden(payload, pools: list[dict[str, Any]]) -> set[frozenset]:
    """把禁配关系展开为 ((池位,实例位),(池位,实例位)) 集合。"""
    pool_idx = {p["name"]: q for q, p in enumerate(pools)}
    pairs: set[frozenset] = set()
    raw_seen: set[frozenset] = set()
    for f in payload.forbidden_matches:
        for side_name in ("pool_a", "pool_b"):
            if getattr(f, side_name) not in pool_idx:
                raise AssemblyError(
                    f"禁配关系引用了不存在的零件池: "
                    f"{getattr(f, side_name)!r}")
        if f.pool_a == f.pool_b:
            raise AssemblyError(
                f"禁配关系必须声明在两个不同零件池之间（池 {f.pool_a}）")
        raw_key = frozenset((
            (f.pool_a, f.batch_a, f.serial_a),
            (f.pool_b, f.batch_b, f.serial_b)))
        if raw_key in raw_seen:
            raise AssemblyError(
                f"禁配关系重复声明: {f.serial_a}({f.pool_a}) ~ "
                f"{f.serial_b}({f.pool_b})")
        raw_seen.add(raw_key)

        def resolve(pool_name, batch_id, serial):
            q = pool_idx[pool_name]
            hits = [(q, k) for k, ins in enumerate(pools[q]["instances"])
                    if ins["serial"] == serial
                    and (batch_id is None or ins["batch_id"] == batch_id)]
            if not hits:
                scope = f"批次 {batch_id} " if batch_id is not None else ""
                raise AssemblyError(
                    f"禁配关系引用了池 {pool_name} {scope}中不存在或缺测"
                    f"被排除的工件序号: {serial!r}")
            return hits

        left = resolve(f.pool_a, f.batch_a, f.serial_a)
        right = resolve(f.pool_b, f.batch_b, f.serial_b)
        for a in left:
            for b in right:
                pairs.add(frozenset((a, b)))
    return pairs


def validate_rules(payload, pools: list[dict[str, Any]]
                   ) -> tuple[dict[int, int], set[frozenset]]:
    """跨批限制 / 同批组 / 禁配关系一致性检查。

    返回 (pool_position -> 同批等价类 id, 展开后的禁配边集合)。
    """
    P = len(pools)
    name_to_pos = {p["name"]: q for q, p in enumerate(pools)}

    limit = (payload.cross_batch_limit if payload.cross_batch_limit is not None
             else P)
    if limit < 1 or limit > P:
        raise AssemblyError(
            f"跨批限制必须在 1..零件池数({P}) 之间，收到 {limit}")

    # 同批组：并查集传递合并
    parent = list(range(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in payload.same_batch_groups:
        if len(g.pools) != len(set(g.pools)):
            raise AssemblyError(f"同批组内零件池重复: {g.pools}")
        unknown = [n for n in g.pools if n not in name_to_pos]
        if unknown:
            raise AssemblyError(f"同批组引用了不存在的零件池: {unknown}")
        positions = [name_to_pos[n] for n in g.pools]
        for q in positions[1:]:
            union(positions[0], q)

    class_of = {q: find(q) for q in range(P)}
    class_members: dict[int, list[int]] = {}
    for q, c in class_of.items():
        class_members.setdefault(c, []).append(q)

    forbidden = _expand_forbidden(payload, pools)

    # 矛盾 1：每个同批组（≥2 池）在成员池的可用批次交集上必须非空
    #    （独立池 / 独立同批组之间可自由使用同一批次，不强制异批）
    for c, members in class_members.items():
        if len(members) < 2:
            continue
        common = set(pools[members[0]]["batch_ids"])
        for q in members[1:]:
            common &= set(pools[q]["batch_ids"])
        if not common:
            names = [pools[q]["name"] for q in members]
            raise AssemblyError(
                f"关系矛盾：同批组 {names} 各池没有共同来源批次，"
                "无法满足同批要求")

    # 矛盾 3：跨批限制为 1 时所有池必须有共同批次
    if limit == 1:
        common = set(pools[0]["batch_ids"])
        for p in pools[1:]:
            common &= set(p["batch_ids"])
        if not common:
            raise AssemblyError(
                "关系矛盾：跨批限制为 1（全部同批），但各零件池的"
                "可用来源批次没有交集")

    # 矛盾 4：禁配双方均为所在池唯一可用实例 => 任何装配都不可能
    for edge in forbidden:
        (a, ai), (b, bj) = sorted(edge)
        if (len(pools[a]["instances"]) == 1
                and len(pools[b]["instances"]) == 1):
            raise AssemblyError(
                "关系矛盾：禁配关系 "
                f"{pools[a]['instances'][ai]['serial']}（池 {pools[a]['name']}）"
                f" 与 {pools[b]['instances'][bj]['serial']}（池 "
                f"{pools[b]['name']}）均为各自池内唯一可用实例，"
                "没有任何可行装配")

    # 矛盾 5：装配数量超过最小池容量
    min_cap = min(len(p["instances"]) for p in pools)
    if payload.assembly_count > min_cap:
        smallest = min(pools, key=lambda p: len(p["instances"]))
        raise AssemblyError(
            f"装配数量 {payload.assembly_count} 超过最小零件池 "
            f"{smallest['name']} 的可用实例数 {len(smallest['instances'])}")

    return class_of, forbidden


# ------------------------------------------------------------- 组合枚举

def enumerate_combos(pools: list[dict[str, Any]], class_of: dict[int, int],
                     cross_limit: int, forbidden: set[frozenset],
                     lsl: float, usl: float, center: float,
                     k_out: float, guard: dict[str, Any]
                     ) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    """枚举可行组合并计算间隙 / 不确定度 / 保护带判定。

    除同批 / 跨批 / 禁配规则外，同一物理工件 (batch_id, serial) 不得
    跨池出现在同一装配中（一个工件不能同时充当两类零件）。
    """
    P = len(pools)
    combos: list[dict[str, Any]] = []
    prune = {"same_batch_group": 0, "cross_batch_limit": 0,
             "forbidden_match": 0, "same_workpiece_across_pools": 0}
    truncated = False
    fixed_w = guard.get("fixed_mm")

    def rec(p, chosen, chosen_keys, class_batch, distinct, gap_sum, u2_sum):
        nonlocal truncated
        if truncated:
            return
        if p == P:
            u = math.sqrt(max(u2_sum, 0.0))
            w = float(fixed_w) if fixed_w is not None else (
                guard["multiple"] * k_out * u)
            decision = _decide(gap_sum, lsl, usl, w)
            if decision == "accept":
                margin = min(gap_sum - (lsl + w), (usl - w) - gap_sum)
            else:
                margin = None
            combos.append({
                "indices": tuple(chosen),
                "key_mask": 0,  # 由调用方在全局工件编号后回填
                "batches": tuple(sorted(distinct)),
                "gap_mm": gap_sum,
                "u_mm": u,
                "guard_band_mm": w,
                "decision": decision,
                "center_deviation_mm": abs(gap_sum - center),
                "guard_margin_mm": margin,
                "cross_transitions": len(distinct) - 1,
            })
            if len(combos) >= COMBO_ENUM_CAP:
                truncated = True
            return
        cls = class_of[p]
        required_batch = class_batch.get(cls)
        for i, ins in enumerate(pools[p]["instances"]):
            b = ins["batch_id"]
            if required_batch is not None and b != required_batch:
                prune["same_batch_group"] += 1
                continue
            new_distinct = distinct | {b}
            if len(new_distinct) > cross_limit:
                prune["cross_batch_limit"] += 1
                continue
            if ins["key_id"] in chosen_keys:
                prune["same_workpiece_across_pools"] += 1
                continue
            if any(frozenset(((p, i), (q, chosen[q]))) in forbidden
                   for q in range(p)):
                prune["forbidden_match"] += 1
                continue
            nxt_class_batch = dict(class_batch)
            nxt_class_batch[cls] = b
            rec(p + 1, chosen + [i], chosen_keys | {ins["key_id"]},
                nxt_class_batch, new_distinct,
                gap_sum + ins["gap_part"], u2_sum + ins["u2"])

    rec(0, [], set(), {}, set(), 0.0, 0.0)
    counts = {"accept": 0, "indeterminate": 0, "reject": 0}
    for c in combos:
        counts[c["decision"]] += 1
    return combos, {**prune, **{f"decision_{k}": v for k, v in counts.items()}}, truncated


# ------------------------------------------------------------- 组合优选

def _combo_key(c: dict[str, Any]) -> tuple:
    return (c["center_deviation_mm"],
            -(c["guard_margin_mm"] if c["guard_margin_mm"] is not None
              else -math.inf),
            c["cross_transitions"])


def greedy_pack(accept: list[dict[str, Any]], P: int,
                target: int, sizes: list[int]) -> list[dict[str, Any]]:
    """同序贪心初解：按字典序逐个取实例 / 工件均不冲突的组合。"""
    avail = [(1 << n) - 1 for n in sizes]
    keys_used = 0
    chosen: list[dict[str, Any]] = []
    for c in accept:  # accept 已按字典序排序
        if (all(avail[p] >> c["indices"][p] & 1 for p in range(P))
                and (c["key_mask"] & keys_used) == 0):
            chosen.append(c)
            for p in range(P):
                avail[p] &= ~(1 << c["indices"][p])
            keys_used |= c["key_mask"]
            if len(chosen) >= target:
                break
    return chosen


def solve(combos: list[dict[str, Any]], sizes: list[int],
          target: int, pool_key_ids: list[list[int]] | None = None
          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """分支定界选实例 / 物理工件均不相交的合格组合；超预算回退贪心。

    pool_key_ids[p] 为第 p 池各实例对应的全局物理工件编号（分支 B 用）。
    """
    P = len(sizes)
    accept = sorted(
        (c for c in combos if c["decision"] == "accept"), key=_combo_key)
    full = [(1 << n) - 1 for n in sizes]

    def score(picked: list[dict[str, Any]]) -> tuple:
        if not picked:
            return (0, 0.0, 0.0, 0)
        worst = min(c["guard_margin_mm"] for c in picked)
        return (
            -len(picked),
            sum(c["center_deviation_mm"] for c in picked),
            -worst,
            sum(c["cross_transitions"] for c in picked),
        )

    greedy = greedy_pack(accept, P, target, sizes)
    best_picked: list[dict[str, Any]] = greedy
    best_key = score(greedy) if greedy else (0, 0.0, 0.0, 0)
    nodes = 0
    exact = True
    fallback_reason: str | None = None

    # 池0 各实例可参与的合格组合（锚定分支用）
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for c in accept:
        by_anchor.setdefault(c["indices"][0], []).append(c)

    def evaluate(picked):
        nonlocal best_picked, best_key
        key = score(picked)
        if picked and (not best_picked or key < best_key):
            best_picked, best_key = list(picked), key

    def dfs(avail, keys_used, picked):
        nonlocal nodes, exact, fallback_reason
        nodes += 1
        if nodes > SOLVER_NODE_BUDGET:
            exact = False
            fallback_reason = (
                f"分支定界节点数超过预算 {SOLVER_NODE_BUDGET}，"
                "保留同序贪心最优解，结果可能非全局最优")
            return
        evaluate(picked)
        chosen_n = len(picked)
        if chosen_n >= target:
            return
        free_counts = [a.bit_count() for a in avail]
        max_add = min(min(free_counts), target - chosen_n)
        if max_add == 0:
            return

        # 可行组合（实例与物理工件均空闲）的统计量，用于字典序下界剪枝
        feasible = [c for c in accept
                    if all(avail[p] >> c["indices"][p] & 1
                           for p in range(P))
                    and (c["key_mask"] & keys_used) == 0]
        if not feasible:
            return
        min_dev = min(c["center_deviation_mm"] for c in feasible)
        max_margin = max(c["guard_margin_mm"] for c in feasible)

        # 计数上界严格不及当前最优 -> 剪枝
        if chosen_n + max_add < -best_key[0]:
            return
        # 计数至多持平：按字典序下界（偏差 / 最差余量）剪枝
        if chosen_n + max_add == -best_key[0]:
            add = -best_key[0] - chosen_n
            if add > 0:
                cur_worst = (min(c["guard_margin_mm"] for c in picked)
                             if picked else math.inf)
                bound_dev = (sum(c["center_deviation_mm"] for c in picked)
                             + add * min_dev)
                bound_margin = -min(cur_worst, max_margin)
                # 偏差下界更优才可能改进；偏差持平时最差余量下界严格更差
                # 才剪枝（余量持平时跨批次数第 4 序仍可能改进，保守保留）
                if (bound_dev > best_key[1]
                        or (bound_dev == best_key[1]
                            and bound_margin > best_key[2])):
                    return

        # 锚定池 0 的最小剩余实例，消除装配排列对称
        anchor = (avail[0] & -avail[0]).bit_length() - 1
        # 分支 A：锚点实例参与某一合格装配
        for c in by_anchor.get(anchor, []):
            if (all(avail[p] >> c["indices"][p] & 1 for p in range(P))
                    and (c["key_mask"] & keys_used) == 0):
                nxt = list(avail)
                for p in range(P):
                    nxt[p] &= ~(1 << c["indices"][p])
                dfs(tuple(nxt), keys_used | c["key_mask"], picked + [c])
                if not exact:
                    return
        # 分支 B：锚点实例本轮不使用（其物理工件在其它池中也一并搁置，
        # 保证同一物理工件不会借道其它池进入本解）
        nxt = list(avail)
        nxt[0] &= ~(1 << anchor)
        if nxt[0] and pool_key_ids is not None:
            anchor_key = 1 << pool_key_ids[0][anchor]
            dfs(tuple(nxt), keys_used | anchor_key, picked)
        elif nxt[0]:
            dfs(tuple(nxt), keys_used, picked)

    dfs(tuple(full), 0, [])
    return sorted(best_picked, key=_combo_key), {
        "exact": exact, "nodes": nodes,
        "fallback_reason": fallback_reason,
        "accept_combos": len(accept),
    }


# ------------------------------------------------------------- 蒙特卡洛复核

def mc_assembly_gap(instances: list[dict[str, Any]], seed: int,
                    samples: int, ordinal_by_key: dict[tuple, int],
                    cache: dict[tuple, np.ndarray | None]) -> dict[str, Any]:
    """固定种子 MC：各实例独立测量误差列求和得到组合间隙误差分布。"""
    rng_seed = np.random.SeedSequence(seed)
    children = rng_seed.spawn(len(ordinal_by_key) + 1)

    total = np.zeros(samples)
    per_instance = []
    for ins in instances:
        key = (ins["batch_id"], ins["serial"])
        if key not in cache:
            col = None
            if ins["has_plan"]:
                corr = np.asarray(ins["corr"], dtype=float)
                u_rest = np.asarray(ins["u_rest_mm"], dtype=float)
                u_res = np.asarray(ins["u_res_mm"], dtype=float)
                signs = np.asarray(ins["signs"], dtype=float)
                rng = np.random.default_rng(children[ordinal_by_key[key]])
                d = len(u_rest)
                eps = (rng.standard_normal((samples, d))
                       @ _cholesky_psd(corr).T) * u_rest[None, :]
                half = u_res * math.sqrt(3.0)
                eps += rng.uniform(-half, half, size=(samples, d))
                col = eps @ signs
            cache[key] = col
        col = cache[key]
        if col is not None:
            total += col
            per_instance.append({
                "batch_id": ins["batch_id"], "serial": ins["serial"],
                "std_error_mm": float(col.std(ddof=1)),
            })
    return {
        "samples": samples,
        "seed": seed,
        "std_mm": float(total.std(ddof=1)),
        "instance_error_std_mm": per_instance,
    }


# ------------------------------------------------------------- 结果组装

def _assembly_report(idx: tuple[int, ...], pools: list[dict[str, Any]],
                     combo: dict[str, Any] | None, order_no: int,
                     locked: bool) -> dict[str, Any]:
    instances = [pools[p]["instances"][idx[p]] for p in range(len(pools))]
    if combo is None:
        gap = sum(i["gap_part"] for i in instances)
        u = math.sqrt(sum(i["u2"] for i in instances))
        combo = {"gap_mm": gap, "u_mm": u,
                 "batches": tuple(sorted({i["batch_id"] for i in instances})),
                 "cross_transitions": 0, "center_deviation_mm": None}
    members = [{
        "pool": pools[p]["name"],
        "batch_id": ins["batch_id"],
        "batch_name": ins["batch_name"],
        "serial": ins["serial"],
        "dimensions": ins["dimensions"],
        "measured_mm": {d: v["value_mm"] for d, v in ins["measured"].items()},
        "submitted": {d: {"value": v["value"], "unit": v["unit"]}
                      for d, v in ins["measured"].items()},
        "corrected_mm": ins["corrected_mm"],
        "gap_part_mm": ins["gap_part"],
        "propagated_variance_mm2": ins["u2"],
    } for p, ins in enumerate(instances)]
    return {
        "assembly_no": order_no,
        "locked": locked,
        "members": members,
        "source_batches": list(combo["batches"]),
        "distinct_batches": len(combo["batches"]),
        "cross_batch_transitions": combo["cross_transitions"],
        "gap_mm": combo["gap_mm"],
        "combined_std_uncertainty_mm": combo["u_mm"],
    }


def _build_snapshot(batch_rows: list[Any], payload, target_mm: dict[str, float],
                    guard: dict[str, Any], seed: int, mc_samples: int,
                    rules_resolved: dict[str, Any],
                    parent_version_id: int | None) -> dict[str, Any]:
    return {
        "source_batches": [{
            "batch_id": r.id,
            "name": r.name,
            "chain_id": r.chain_id,
            "created_at": r.created_at.isoformat(),
            "frozen": bool(r.frozen),
            "measurement_plan_id": r.measurement_plan_id,
            "frozen_rows": r.rows_json,
            "plan_snapshot": (
                r.measurement_json.get("plan_snapshot")
                if r.measurement_json else None),
        } for r in batch_rows],
        "target_gap": {
            "submitted": payload.target_gap.model_dump(mode="json"),
            "normalized_mm": target_mm,
        },
        "guard_band": guard,
        "output_coverage_factor": getattr(payload, "output_coverage_factor", 2.0),
        "random_seed": seed,
        "monte_carlo_samples": mc_samples,
        "pool_mapping": [{"name": p.name, "dimensions": list(p.dimensions)}
                         for p in payload.pools],
        "rules": rules_resolved,
        "assembly_count": payload.assembly_count,
        "cross_batch_limit": payload.cross_batch_limit,
        "parent_version_id": parent_version_id,
        "freeze_policy": "源批次冻结行、测量方案快照、保护带设置与随机种子"
                         "随任务版本一次性冻结，后续批次/方案数据不改写",
    }


# ------------------------------------------------------------- 主编排

def _guard_dict(gb, k_out: float) -> dict[str, Any]:
    return {
        "mode": gb.mode,
        "multiple": gb.multiple if gb.mode == "multiple" else None,
        "fixed_mm": (to_mm(gb.fixed, gb.unit.value)
                     if gb.mode == "fixed" else None),
        "submitted": gb.model_dump(mode="json"),
        "effective_multiple": (gb.multiple if gb.mode == "multiple"
                               else None),
        "applied_to": f"组合扩展不确定度 U = {k_out}·u_C；"
                      "multiple 时 w = multiple × U",
    }


def _remaining_pools(pools, locked_indices: list[tuple[int, ...]]):
    """从各池移除已锁定实例，返回受限池（带剩余原因统计）。

    同时移除已锁定装配在其它池中占用的同一物理工件
    (batch_id, serial)——一个工件在整个任务中只能使用一次。
    """
    locked_keys: set[tuple[int, str]] = set()
    used: list[set[int]] = [set() for _ in pools]
    for idx in locked_indices:
        for p, i in enumerate(idx):
            used[p].add(i)
            ins = pools[p]["instances"][i]
            locked_keys.add((ins["batch_id"], ins["serial"]))
    remaining = []
    for p, pool in enumerate(pools):
        keep = []
        removed_keys: list[str] = []
        for i, ins in enumerate(pool["instances"]):
            if i in used[p]:
                continue
            if (ins["batch_id"], ins["serial"]) in locked_keys:
                removed_keys.append(f"{ins['batch_id']}:{ins['serial']}")
                continue
            keep.append(ins)
        remaining.append({
            "name": pool["name"],
            "dimensions": pool["dimensions"],
            "instances": keep,
            "excluded": pool["excluded"],
            "batch_ids": sorted({i["batch_id"] for i in keep}),
            "removed_locked": sorted(
                f'{pool["instances"][i]["batch_id"]}:'
                f'{pool["instances"][i]["serial"]}' for i in used[p]),
            "removed_same_workpiece": sorted(set(removed_keys)),
        })
    return remaining


def _assign_key_ids(pools) -> int:
    """给各池实例分配全局物理工件编号 key_id（同名同批视为同一工件）。"""
    key_of: dict[tuple[int, str], int] = {}
    for pool in pools:
        for ins in pool["instances"]:
            k = (ins["batch_id"], ins["serial"])
            if k not in key_of:
                key_of[k] = len(key_of)
            ins["key_id"] = key_of[k]
    return len(key_of)


def _diagnostics(combos, pools, remaining_pools, prune, truncated,
                 required, locked_n, target_mm, guard, k_out) -> dict[str, Any]:
    accept = [c for c in combos if c["decision"] == "accept"]
    fixed_w = guard.get("fixed_mm")
    restricted = []
    for p in remaining_pools:
        reasons = []
        if locked_n:
            reasons.append(
                f"{locked_n} 个已锁定装配占用了 {len(p['removed_locked'])} 个"
                "实例")
        for ex in p["excluded"]:
            reasons.append(
                f"{ex['batch_id']}:{ex['serial']} 缺测 "
                f"{ex['missing_dimension_ids']}（已排除）")
        restricted.append({
            "pool": p["name"],
            "available_instances": len(p["instances"]),
            "batch_ids": p["batch_ids"],
            "removed_locked_keys": p["removed_locked"],
            "removed_same_workpiece_keys": p.get("removed_same_workpiece", []),
            "reasons": reasons,
        })
        for key in p.get("removed_same_workpiece", []):
            restricted[-1]["reasons"].append(
                f"工件 {key} 已被已锁定装配在其它零件池占用，"
                "同一物理工件不可跨池重复使用")

    missing_serials: list[dict[str, Any]] = []
    for p in pools:
        for ex in p["excluded"]:
            missing_serials.append({
                "pool": p["name"],
                "batch_id": ex["batch_id"],
                "serial": ex["serial"],
                "reason": ex["reason"],
                "missing_dimension_ids": ex["missing_dimension_ids"],
            })

    # 触发规则：统计限制下的可行组合数
    rules = []
    for key, label in (
        ("same_batch_group", "同批分组规则剪枝"),
        ("cross_batch_limit", "跨批限制剪枝"),
        ("forbidden_match", "禁配关系剪枝"),
        ("same_workpiece_across_pools", "同一物理工件跨池重复剪枝"),
    ):
        if prune.get(key):
            rules.append({"rule": label, "pruned_partial_combinations":
                          prune[key]})
    if truncated:
        rules.append({
            "rule": f"组合枚举上限 {COMBO_ENUM_CAP}",
            "pruned_partial_combinations": None,
            "note": "枚举截断：合格判定统计与求解仅基于前序枚举，"
                    "可减少来源批次或实例数后重试",
        })
    if not accept:
        rules.append({
            "rule": "目标间隙 + 保护带",
            "note": (f"目标区间 [{target_mm['lower']}, {target_mm['upper']}] mm，"
                     + (f"固定保护带 w={fixed_w} mm" if fixed_w is not None
                        else f"保护带 w = {guard['multiple']}×{k_out}·u_C")),
        })
    elif len(accept) < required:
        rules.append({
            "rule": "实例不相交约束",
            "note": f"合格组合 {len(accept)} 个，但每个实例只能使用一次，"
                    f"最多只能完成 {_max_disjoint_count(combos, pools)} 个"
                    "互不相交的合格装配",
        })

    return {
        "restricted_pools": restricted,
        "missing_serials": missing_serials,
        "triggered_rules": rules,
        "candidate_combo_counts": {
            "total_feasible": len(combos),
            "accept": prune.get("decision_accept", 0),
            "indeterminate": prune.get("decision_indeterminate", 0),
            "reject": prune.get("decision_reject", 0),
        },
    }


def _max_disjoint_count(combos, pools) -> int:
    """贪心估计最多互不相交合格装配数（诊断展示用）。"""
    P = len(pools)
    sizes = [len(p["instances"]) for p in pools]
    accept = sorted((c for c in combos if c["decision"] == "accept"),
                    key=_combo_key)
    picked = greedy_pack(accept, P, max(sizes), sizes)
    return len(picked)


def run_task(nc: NormalizedChain, batch_rows: list[Any], payload, *,
             seed: int, mc_samples: int,
             locked_assemblies: list[list[dict[str, Any]]] | None = None,
             parent_version_id: int | None = None,
             ) -> dict[str, Any]:
    """构建池 -> 规则校验 -> 枚举 -> 求解 -> MC 复核 -> 组装结果。

    locked_assemblies：版本重排时已确认锁定的装配（含历史版本累积），
    每个元素为 [{pool, batch_id, serial}]；这些实例先从池中移除。
    """
    built = build_pools(nc, batch_rows, payload)
    pools0 = built["pools"]
    class_of, forbidden = validate_rules(payload, pools0)

    target_mm = {
        "lower": to_mm(payload.target_gap.lower,
                       payload.target_gap.unit.value),
        "upper": to_mm(payload.target_gap.upper,
                       payload.target_gap.unit.value),
    }
    center = 0.5 * (target_mm["lower"] + target_mm["upper"])
    gb = payload.guard_band or GuardBandSpec()
    guard = _guard_dict(gb, payload.output_coverage_factor)
    cross_limit = (payload.cross_batch_limit
                   if payload.cross_batch_limit is not None else len(pools0))

    # ---- 解析已锁定装配为实例下标（同批 / 禁配一致性在版本入口校验）
    locked_indices: list[tuple[int, ...]] = []
    locked_records = locked_assemblies or []
    pool_pos = {p["name"]: q for q, p in enumerate(pools0)}
    for lock in locked_records:
        idx = [-1] * len(pools0)
        for m in lock:
            q = pool_pos[m["pool"]]
            hit = [k for k, ins in enumerate(pools0[q]["instances"])
                   if ins["batch_id"] == m["batch_id"]
                   and ins["serial"] == m["serial"]]
            if not hit:
                raise AssemblyError(
                    f"锁定装配引用的实例 {m['batch_id']}:{m['serial']} "
                    f"不在池 {m['pool']} 的可用实例中（可能缺测被排除）")
            idx[q] = hit[0]
        if any(i < 0 for i in idx):
            raise AssemblyError("锁定装配未覆盖全部零件池")
        locked_indices.append(tuple(idx))

    pools = _remaining_pools(pools0, locked_indices)
    remaining_need = payload.assembly_count - len(locked_indices)
    _assign_key_ids(pools0)
    _assign_key_ids(pools)

    combos, prune, truncated = enumerate_combos(
        pools, class_of, cross_limit, forbidden,
        target_mm["lower"], target_mm["upper"], center,
        payload.output_coverage_factor, guard)
    # 回填每个可行组合占用的全局物理工件位掩码
    for c in combos:
        mask = 0
        for p, i in enumerate(c["indices"]):
            mask |= 1 << pools[p]["instances"][i]["key_id"]
        c["key_mask"] = mask
    sizes = [len(p["instances"]) for p in pools]
    if remaining_need > min(sizes, default=0):
        raise AssemblyError(
            f"扣除 {len(locked_indices)} 个已锁定装配后，剩余需求 "
            f"{remaining_need} 超过最小零件池容量 "
            f"{min(sizes, default=0)}")
    pool_key_ids = [[ins["key_id"] for ins in p["instances"]] for p in pools]
    picked, solver_info = solve(
        combos, sizes, remaining_need, pool_key_ids=pool_key_ids)
    picked = picked[:remaining_need]

    # ---- MC 复核（每实例独立误差列；无方案实例为零）
    all_used: list[tuple[int, ...]] = list(locked_indices)
    for c in picked:
        all_used.append(tuple(
            pools0_pos_of(pools0, pools, p, c["indices"][p])
            for p in range(len(pools))))

    ordinal: dict[tuple, int] = {}
    for tup in all_used:
        for p, i in enumerate(tup):
            ins = pools0[p]["instances"][i]
            ordinal.setdefault((ins["batch_id"], ins["serial"]), len(ordinal))
    mc_cache: dict[tuple, np.ndarray | None] = {}

    assemblies = []
    no = 1
    for tup in locked_indices:
        instances = [pools0[p]["instances"][i] for p, i in enumerate(tup)]
        gap = sum(i["gap_part"] for i in instances)
        u = math.sqrt(sum(i["u2"] for i in instances))
        w = (guard["fixed_mm"] if guard["fixed_mm"] is not None
             else guard["multiple"] * payload.output_coverage_factor * u)
        dec = _decide(gap, target_mm["lower"], target_mm["upper"], w)
        combo_like = {
            "gap_mm": gap, "u_mm": u,
            "batches": tuple(sorted({i["batch_id"] for i in instances})),
            "cross_transitions": len({i["batch_id"] for i in instances}) - 1,
        }
        rep = _assembly_report(tup, pools0, combo_like, no, locked=True)
        mc = mc_assembly_gap(instances, seed, mc_samples, ordinal, mc_cache)
        rep.update({
            "decision": dec,
            "guard_band_mm": w,
            "center_deviation_mm": abs(gap - center),
            "guard_margin_mm": (
                min(gap - (target_mm["lower"] + w),
                    (target_mm["upper"] - w) - gap)
                if dec == "accept" else None),
            "monte_carlo": mc,
        })
        assemblies.append(rep)
        no += 1

    for c in picked:
        tup = tuple(pools0_pos_of(pools0, pools, p, c["indices"][p])
                    for p in range(len(pools)))
        rep = _assembly_report(tup, pools0, c, no, locked=False)
        instances = [pools0[p]["instances"][i] for p, i in enumerate(tup)]
        rep.update({
            "decision": c["decision"],
            "guard_band_mm": c["guard_band_mm"],
            "center_deviation_mm": c["center_deviation_mm"],
            "guard_margin_mm": c["guard_margin_mm"],
            "monte_carlo": mc_assembly_gap(
                instances, seed, mc_samples, ordinal, mc_cache),
        })
        assemblies.append(rep)
        no += 1

    accepted = [a for a in assemblies if a["decision"] == "accept"]
    status = ("solved" if len(accepted) >= payload.assembly_count
              else ("partial" if accepted else "infeasible"))

    objective = {
        "qualified_assemblies": len(accepted),
        "required_count": payload.assembly_count,
        "total_center_deviation_mm": sum(
            a["center_deviation_mm"] or 0.0 for a in accepted),
        "worst_guard_margin_mm": (
            min(a["guard_margin_mm"] for a in accepted)
            if accepted else None),
        "total_cross_batch_transitions": sum(
            a["cross_batch_transitions"] for a in accepted),
        "priority": ["合格装配数最大化", "中心偏差总和最小",
                     "最差保护余量最大", "总跨批次数最小"],
    }
    diag = _diagnostics(
        combos, pools, pools, prune, truncated,
        remaining_need, len(locked_indices), target_mm, guard,
        payload.output_coverage_factor)
    if status == "solved":
        diag["triggered_rules"] = [
            r for r in diag["triggered_rules"]
            if r["rule"] not in ("实例不相交约束",)]

    rules_resolved = {
        "cross_batch_limit": cross_limit,
        "same_batch_classes": {
            pools0[q]["name"]: c for q, c in sorted(class_of.items())},
        "forbidden_match_edges": [
            {"pool_a": pools0[a]["name"],
             "instance_a": f'{pools0[a]["instances"][ai]["batch_id"]}:'
                           f'{pools0[a]["instances"][ai]["serial"]}',
             "pool_b": pools0[b]["name"],
             "instance_b": f'{pools0[b]["instances"][bj]["batch_id"]}:'
                           f'{pools0[b]["instances"][bj]["serial"]}'}
            for (a, ai), (b, bj) in
            (sorted(edge) for edge in forbidden)],
        # 原样冻结调用方提交的规则，供后续版本精确重放
        "submitted_same_batch_groups": [
            {"pools": list(g.pools)} for g in payload.same_batch_groups],
        "submitted_forbidden_matches": [
            f.model_dump(mode="json") for f in payload.forbidden_matches],
    }
    snapshot = _build_snapshot(
        batch_rows, payload, target_mm, guard, seed, mc_samples,
        rules_resolved, parent_version_id)

    return {
        "status": status,
        "required_count": payload.assembly_count,
        "qualified_assemblies": len(accepted),
        "locked_assemblies": len(locked_indices),
        "assemblies": assemblies,
        "objective": objective,
        "diagnostics": diag,
        "pools": [{
            "name": p["name"],
            "dimensions": p["dimensions"],
            "available": len(p["instances"]),
            "batch_ids": p["batch_ids"],
            "excluded_instances": p["excluded"],
        } for p in pools0],
        "solver": {
            "exact": solver_info["exact"],
            "nodes_visited": solver_info["nodes"],
            "candidate_combos_enumerated": len(combos),
            "enumeration_truncated": truncated,
            "enum_cap": COMBO_ENUM_CAP,
            "node_budget": SOLVER_NODE_BUDGET,
            "fallback_reason": solver_info["fallback_reason"],
        },
        "monte_carlo": {"samples": mc_samples, "seed": seed,
                        "policy": "每实例独立测量误差列（实例内方案 ρ 相关，"
                                  "SeedSequence 按 (批次,序号) 派生）；"
                                  "组合误差为各实例列之和"},
        "formulas": ASSEMBLY_FORMULAS,
        "snapshot": snapshot,
    }


def pools0_pos_of(pools0, remaining, p: int, rem_index: int) -> int:
    """剩余池实例 -> 原池实例下标（remaining 由 _remaining_pools 保序）。"""
    name = remaining[p]["name"]
    q0 = next(q for q, x in enumerate(pools0) if x["name"] == name)
    target_serial = remaining[p]["instances"][rem_index]["serial"]
    target_batch = remaining[p]["instances"][rem_index]["batch_id"]
    return next(k for k, ins in enumerate(pools0[q0]["instances"])
                if ins["serial"] == target_serial
                and ins["batch_id"] == target_batch)

