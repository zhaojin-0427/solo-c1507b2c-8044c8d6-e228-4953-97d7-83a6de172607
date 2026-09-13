"""孔系整改方案搜索：候选钻孔 / 连接件规格 / 孔位修正组合枚举与排序。

候选作用（单位已在端点层换算为 mm 内部表示）：

* drill_options：放大某匹配位某侧孔的名义直径与偏差（重新钻孔）；
* fastener_options：固定螺栓型匹配位更换连接件直径规格；
* correction_options：允许在给定上界内修正 A 侧孔位（平移孔中心）。

求解口径与版本分析完全同源（同一固定种子、同样本数）：孔位修正在几何上
等效于给该位圆盘增加自由方向半径（修正矢量可自由定向到最差匹配位法线），
故 WC 用 ``extra_radius`` 精确处理；MC 中把同一修正预算作为逐样本可自由
定向的径向附加量（保守且确定性）。

排序键（越小越优）：失败概率（固定种子 MC）→ 最大孔位改动（mm）→
孔径放大总量（mm，名义直径增量）；现状（零改动）方案始终参与复核。
采纳（select）后由端点冻结为新版本孔系（放大孔径 / 螺栓规格 / 孔位匹配
关系 / 随机种子全部固化）。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any

import numpy as np

from .hole_schemas import RemedySearchRequest
from .holes import (
    HoleModel,
    _ls_start_scalar,
    _subgrad_scalar,
    _vc_allowance,
    monte_carlo,
    worst_case,
)


class RemedyError(ValueError):
    """整改搜索的可定位错误（端点转 422）。"""


def _mm(value, factor: float) -> float:
    return value * factor


def validate_request(model: HoleModel, payload: RemedySearchRequest) -> dict[str, Any]:
    """把候选表换算到 mm 并复核引用对象（未知匹配位 / 侧别 / 锁定冲突 -> 422）。"""
    ids = set(model.mate_ids)
    unknown_locks = [m for m in payload.locked_mates if m not in ids]
    unknown_fastener_locks = [m for m in payload.locked_fasteners if m not in ids]
    if unknown_locks:
        raise RemedyError(
            f"锁定孔位引用了不存在的匹配位: {sorted(unknown_locks)}；"
            f"孔系匹配位: {model.mate_ids}")
    if unknown_fastener_locks:
        raise RemedyError(
            f"锁定连接件引用了不存在的匹配位: {sorted(unknown_fastener_locks)}")
    if len(set(payload.locked_mates)) != len(payload.locked_mates):
        raise RemedyError("locked_mates 存在重复匹配位")
    if len(set(payload.locked_fasteners)) != len(payload.locked_fasteners):
        raise RemedyError("locked_fasteners 存在重复匹配位")

    f = model.factor
    drills: dict[str, list[dict]] = {}
    for opt in payload.drill_options:
        if opt.mate_id not in ids:
            raise RemedyError(
                f"候选钻孔引用了不存在的匹配位 {opt.mate_id!r}")
        i = model.mate_ids.index(opt.mate_id)
        feat = model.features_a[i] if opt.side == "a" else model.features_b[i]
        if feat.kind != "hole":
            raise RemedyError(
                f"匹配位 {opt.mate_id} 侧 {opt.side.upper()} 不是孔要素，"
                "不能重新钻孔")
        nom = _mm(opt.nominal_diameter, f)
        es, ei = _mm(opt.diameter_upper_deviation, f), \
            _mm(opt.diameter_lower_deviation, f)
        if nom <= feat.nom + 1e-12:
            raise RemedyError(
                f"匹配位 {opt.mate_id} 侧 {opt.side.upper()} 候选钻孔名义孔径 "
                f"{nom:g} 不大于现状名义孔径 {feat.nom:g}，重新钻孔必须放大孔径")
        drills.setdefault((opt.side, opt.mate_id), []).append(
            {"nominal": nom, "es": es, "ei": ei})

    fasteners: dict[str, list[dict]] = {}
    for opt in payload.fastener_options:
        if opt.mate_id not in ids:
            raise RemedyError(
                f"候选连接件引用了不存在的匹配位 {opt.mate_id!r}")
        i = model.mate_ids.index(opt.mate_id)
        if opt.mate_id in payload.locked_fasteners:
            raise RemedyError(
                f"匹配位 {opt.mate_id} 已锁定连接件，不能再给候选规格")
        up, lo = _mm(opt.diameter_upper, f), _mm(opt.diameter_lower, f)
        fasteners.setdefault(opt.mate_id, []).append(
            {"upper": up, "lower": lo})
    # 连接件候选只对固定螺栓（feature_b=None 或双孔浮动）有效：
    # 销配合的连接件是 B 零件本身，不在本搜索范围。
    for mid, opts in fasteners.items():
        i = model.mate_ids.index(mid)
        if model.mate_kinds[i] == "fixed" and model.ext_sides[i] != "B":
            raise RemedyError(
                f"匹配位 {mid} 的外要素在 A 侧（销零件），更换连接件规格"
                "不适用；请用候选钻孔放大孔")
        # 去重并按上极限排序
        uniq = {(round(o["upper"], 9), round(o["lower"], 9)): o
                for o in opts}
        fasteners[mid] = sorted(uniq.values(), key=lambda o: o["upper"])

    corrections: dict[str, float] = {}
    for opt in payload.correction_options:
        if opt.mate_id not in ids:
            raise RemedyError(
                f"孔位修正引用了不存在的匹配位 {opt.mate_id!r}")
        if opt.mate_id in payload.locked_mates:
            raise RemedyError(
                f"匹配位 {opt.mate_id} 已锁定孔位，不能同时允许孔位修正")
        if model.features_a[model.mate_ids.index(opt.mate_id)].kind != "hole":
            raise RemedyError(
                f"匹配位 {opt.mate_id} A 侧不是孔，不能做孔位修正")
        corrections[opt.mate_id] = _mm(opt.max_shift, f)

    return {"drills": drills, "fasteners": fasteners,
            "corrections": corrections}


# ----------------------------------------------------- 候选模型派生

def _apply_candidate(model: HoleModel, drill: dict, fastener: dict,
                     ) -> tuple[HoleModel, np.ndarray, float, float]:
    """在模型上施加孔径 / 螺栓选择，返回 (新模型, 修正半径向量, 孔径放大, 孔位改动)。"""
    m = replace(model,
                features_a=[replace(x) for x in model.features_a],
                features_b=[replace(x) for x in model.features_b],
                bolt_nom=list(model.bolt_nom),
                bolt_es=list(model.bolt_es),
                bolt_ei=list(model.bolt_ei),
                bolt_lo=list(model.bolt_lo),
                bolt_hi=list(model.bolt_hi))
    radius = np.zeros(m.n)
    enlarge = 0.0
    correction_shift = 0.0
    for (side, mid), choice in drill.items():
        if choice is None:
            continue
        i = m.mate_ids.index(mid)
        feat = m.features_a[i] if side == "a" else m.features_b[i]
        enlarge += max(0.0, choice["nominal"] - feat.nom)
        feat.nom, feat.es, feat.ei = choice["nominal"], choice["es"], choice["ei"]
    for mid, choice in fastener.items():
        if choice is None:
            continue
        i = m.mate_ids.index(mid)
        mid_d = 0.5 * (choice["upper"] + choice["lower"])
        if m.mate_kinds[i] == "float":
            # 浮动连接件：写入螺栓极限（rad 与抽样都用直径极限）
            m.bolt_nom[i] = mid_d
            m.bolt_es[i] = choice["upper"] - mid_d
            m.bolt_ei[i] = choice["lower"] - mid_d
            m.bolt_lo[i] = choice["lower"]
            m.bolt_hi[i] = choice["upper"]
        else:
            # 固定螺栓（无 feature_b 的合成销）：直接改 B 侧销直径
            pin = m.features_b[i] if m.ext_sides[i] == "B" else m.features_a[i]
            pin.nom = mid_d
            pin.es = choice["upper"] - mid_d
            pin.ei = mid_d - choice["lower"]
    return m, radius, enlarge, correction_shift


def _candidate_iter(model: HoleModel, tables: dict, include_correction: bool,
                    locked_mates: set[str]):
    """枚举 (drill 选择, fastener 选择) 组合；孔位修正作为每候选的可加预算。

    每个匹配位的钻孔候选含 None（保持现状）；候选数受 max_candidates 截断。
    """
    drill_groups: list[tuple[str, str, list[dict | None]]] = []
    for (side, mid), opts in sorted(tables["drills"].items()):
        drill_groups.append((side, mid, [None] + opts))
    fastener_groups: list[tuple[str, list[dict | None]]] = []
    for mid, opts in sorted(tables["fasteners"].items()):
        fastener_groups.append((mid, [None] + opts))

    d_keys = [(s, m) for s, m, _ in drill_groups]
    f_keys = [m for m, _ in fastener_groups]
    d_choices = [g[2] for g in drill_groups]
    f_choices = [g[1] for g in fastener_groups]
    for d_combo in itertools.product(*d_choices) if d_choices else [()]:
        for f_combo in itertools.product(*f_choices) if f_choices else [()]:
            drill = {k: v for k, v in zip(d_keys, d_combo)}
            fastener = {k: v for k, v in zip(f_keys, f_combo)}
            yield drill, fastener


# ----------------------------------------------------- 孔位修正（确定性）

def _shifted_model(model: HoleModel, shifts: dict[str, np.ndarray]) -> HoleModel:
    """把 A 侧名义孔位平移 shifts[mate_id]=(sx,sy) mm，返回新模型。"""
    m = replace(model, features_a=[replace(x) for x in model.features_a])
    for i, mid in enumerate(m.mate_ids):
        if mid in shifts:
            sx, sy = shifts[mid]
            m.features_a[i].x += float(sx)
            m.features_a[i].y += float(sy)
    return m


def _correction_shifts(model: HoleModel, budget: dict[str, float],
                       locked: set[str]) -> dict[str, np.ndarray]:
    """从最坏边界最优位姿导出每个可修正匹配位的确定性平移矢量。

    圆盘约束 |R(θ)pB+t−pA−s_i| ≤ a_i：把 A 孔朝 B 侧实际中心方向移 s_i，
    在预算内取 s_i = min(b_i, 间隙不足量)·u_i，其中 u_i 为最优位姿下
    B 中心相对 A 中心的单位方向。这样最坏边界的修正收益被逐点实现，
    采纳后重算的版本与候选结果同源、同结论。
    """
    rad, _ = _vc_allowance(model)
    PA = np.array([[f.x, f.y] for f in model.features_a])
    PB = np.array([[f.x, f.y] for f in model.features_b])
    # 预算作为自由半径只用于找"如果允许修正，位姿应朝哪走"的方向，
    # 真正的收益仍由平移后的名义孔位决定（不夸大）。
    inflated = rad.copy()
    for i, mid in enumerate(model.mate_ids):
        if mid in budget and mid not in locked:
            inflated[i] += budget[mid]
    x_ls = _ls_start_scalar(PA, PB, PA, model.theta_max_rad)
    x, g, gv = _subgrad_scalar(x_ls, PA, PB, PA, inflated,
                               model.theta_max_rad, 400)
    th = x[2]
    c, s = math.cos(th), math.sin(th)
    posed_B = np.column_stack([c * PB[:, 0] - s * PB[:, 1] + x[0],
                               s * PB[:, 0] + c * PB[:, 1] + x[1]])
    shifts: dict[str, np.ndarray] = {}
    for i, mid in enumerate(model.mate_ids):
        if mid not in budget or mid in locked:
            continue
        v = posed_B[i] - PA[i]
        dist = float(np.hypot(v[0], v[1]))
        b = budget[mid]
        if dist <= 1e-12 or b <= 0.0:
            continue
        # 朝 B 侧方向移动，幅度不超过预算；间隙已够（dist≤rad）时无需修正
        need = max(0.0, dist - rad[i])
        mag = min(b, need)
        if mag > 0.0:
            shifts[mid] = np.array([v[0] / dist * mag, v[1] / dist * mag])
    return shifts


# ----------------------------------------------------- 主体搜索

def search_remedies(model: HoleModel, payload: RemedySearchRequest) -> dict[str, Any]:
    tables = validate_request(model, payload)
    n_samples = payload.mc_samples or model.mc_samples
    seed = payload.random_seed if payload.random_seed is not None else model.seed

    locked_mates = set(payload.locked_mates)
    correction_budget = dict(tables["corrections"])
    if not payload.include_correction:
        correction_budget = {}

    def evaluate(m: HoleModel):
        """对最终模型（孔径/螺栓/孔位已固化）做最坏边界 + 固定种子 MC。

        MC 直接调用分析模块（与版本采纳后重算完全同源），
        保证候选声称的失败概率等于采纳后新版本的失败概率。
        """
        wc = worst_case(m)
        mc = monte_carlo(m, n_samples, seed=seed)
        return wc, mc["failure_probability"]

    candidates = []
    count = 0
    truncate = False
    for drill, fastener in _candidate_iter(
            model, tables, payload.include_correction, locked_mates):
        if count >= payload.max_candidates:
            truncate = True
            break
        base_m, _, enlarge, _ = _apply_candidate(model, drill, fastener)
        # 每个 (钻孔, 连接件) 组合给两种变体：不修正 / 使用声明的孔位修正；
        # 零改动零修正的现状方案因此必然在候选集合中。
        use_correction = bool(payload.include_correction and correction_budget)
        variants = [False, True] if use_correction else [False]
        for use_shift in variants:
            if count >= payload.max_candidates:
                truncate = True
                break
            shifts: dict[str, np.ndarray] = {}
            shift_records = []
            shift_max = 0.0
            if use_shift:
                shifts = _correction_shifts(
                    base_m, correction_budget, locked_mates)
                for mid, v in shifts.items():
                    mag = float(np.hypot(v[0], v[1]))
                    shift_max = max(shift_max, mag)
                    shift_records.append({
                        "mate_id": mid,
                        "shift_x_mm": float(v[0]),
                        "shift_y_mm": float(v[1]),
                        "shift_magnitude_mm": mag,
                        "max_allowed_mm": correction_budget[mid]})
            eval_m = _shifted_model(base_m, shifts) if shifts else base_m
            wc, fail = evaluate(eval_m)
            count += 1
            candidates.append({
                "drill": [
                    {"mate_id": mid, "side": side,
                     "nominal_diameter_mm": c["nominal"],
                     "diameter_upper_deviation_mm": c["es"],
                     "diameter_lower_deviation_mm": c["ei"]}
                    for (side, mid), c in drill.items() if c is not None],
                "fasteners": [
                    {"mate_id": mid,
                     "diameter_upper_mm": c["upper"],
                     "diameter_lower_mm": c["lower"]}
                    for mid, c in fastener.items() if c is not None],
                "uses_hole_correction": use_shift,
                "hole_correction_budget_mm": [
                    {"mate_id": mid, "max_shift_mm": v}
                    for mid, v in sorted(correction_budget.items())
                    if use_shift and mid not in locked_mates],
                "hole_correction_vectors_mm": shift_records,
                "max_hole_shift_mm": shift_max,
                "hole_enlargement_total_mm": enlarge,
                "worst_case_feasible": bool(wc["feasible"]),
                "worst_case_margin_mm":
                    wc["best_pose"]["worst_margin_mm"],
                "failure_probability": fail,
                "assembly_success_rate": 1.0 - fail,
                "limiting_mate":
                    wc["limiting_mate_at_best_pose"]["mate_id"],
            })
        if truncate:
            break

    candidates.sort(key=lambda c: (
        c["failure_probability"],
        c["max_hole_shift_mm"],
        c["hole_enlargement_total_mm"],
        -(1 if c["worst_case_feasible"] else 0),
    ))
    for rank, c in enumerate(candidates, start=1):
        c["rank"] = rank

    baseline = next((c for c in candidates
                     if not c["drill"] and not c["fasteners"]
                     and c["max_hole_shift_mm"] == 0.0), None)
    return {
        "status": "ok",
        "samples": n_samples,
        "random_seed": seed,
        "candidate_count": len(candidates),
        "candidate_cap": payload.max_candidates,
        "truncated": truncate,
        "ranking_note":
            "排序键（越小越优）：固定种子 MC 失败概率 → 最大孔位改动(mm) "
            "→ 孔径放大总量(mm)；零改动现状方案始终参与复核；"
            "孔位修正以最坏边界最优位姿导出的确定性平移矢量固化，"
            "采纳后新版本失败概率与本结果一致",
        "baseline": baseline,
        "candidates": candidates,
        "shared_sampling": {
            "policy": "所有候选用同种子、同样本数经分析模块蒙特卡洛评定，"
                      "候选间差异只来自孔径/螺栓/孔位修正选择",
            "seed": seed,
            "samples": n_samples,
        },
    }


# ----------------------------------------------------- 采纳：冻结为新版本输入

def freeze_payload(model: HoleModel, candidate: dict, name: str, note: str,
                   parent_payload, mc_samples: int | None = None,
                   seed: int | None = None):
    """把选中候选应用到父版本提交，返回可直接建新版本的 HolePatternCreate。

    放大孔径 / 螺栓规格写入对应要素；孔位修正按求解出的确定性平移矢量
    写入 A 侧名义孔坐标（采纳后新版本重算结果与候选同源）；
    匹配关系保持冻结；mc_samples/seed 缺省沿用父版本。
    """
    from .hole_schemas import HolePatternCreate

    data = parent_payload.model_dump(mode="json") if hasattr(
        parent_payload, "model_dump") else dict(parent_payload)
    data["name"] = name
    data["note"] = note
    if mc_samples is not None:
        data["mc_samples"] = mc_samples
    if seed is not None:
        data["random_seed"] = seed
    drill_map = {(d["side"], d["mate_id"]): d for d in candidate["drill"]}
    fast_map = {d["mate_id"]: d for d in candidate["fasteners"]}
    shift_map = {v["mate_id"]: v for v in candidate.get(
        "hole_correction_vectors_mm", [])}
    inv = 1.0 / model.factor
    mate_index = {mid: i for i, mid in enumerate(model.mate_ids)}
    for m in data["mates"]:
        i = mate_index[m["id"]]
        # 采纳孔位修正：把确定性平移矢量（mm）写入 A 侧名义坐标（提交单位）
        sv = shift_map.get(m["id"])
        if sv is not None:
            m["feature_a"]["x"] = m["feature_a"].get("x", 0.0) \
                + sv["shift_x_mm"] * inv
            m["feature_a"]["y"] = m["feature_a"].get("y", 0.0) \
                + sv["shift_y_mm"] * inv
        for side in ("a", "b"):
            key = f"feature_{side}"
            d = drill_map.get((side, m["id"]))
            if d is not None and m.get(key):
                m[key]["nominal_diameter"] = d["nominal_diameter_mm"] * inv
                m[key]["diameter_upper_deviation"] = \
                    d["diameter_upper_deviation_mm"] * inv
                m[key]["diameter_lower_deviation"] = \
                    d["diameter_lower_deviation_mm"] * inv
        fb = fast_map.get(m["id"])
        if fb is None:
            continue
        up, lo = fb["diameter_upper_mm"] * inv, fb["diameter_lower_mm"] * inv
        if model.mate_kinds[i] == "float":
            # 浮动连接件：更换螺栓极限
            m["bolt_diameter_upper"] = up
            m["bolt_diameter_lower"] = lo
        elif m.get("feature_b") and m["feature_b"]["kind"] == "pin":
            # 固定销配合：换 B 侧销直径（对称偏差）
            m["feature_b"]["nominal_diameter"] = 0.5 * (up + lo)
            m["feature_b"]["diameter_upper_deviation"] = up - 0.5 * (up + lo)
            m["feature_b"]["diameter_lower_deviation"] = 0.5 * (up + lo) - lo
        else:
            # 固定螺栓（无 feature_b）：改螺栓直径极限
            m["bolt_diameter_upper"] = up
            m["bolt_diameter_lower"] = lo
    data["random_seed"] = model.seed
    data["adopted_remedy"] = {
        "drill": candidate["drill"],
        "fasteners": candidate["fasteners"],
        "hole_correction_budget_mm": candidate["hole_correction_budget_mm"],
        "hole_correction_vectors_mm": candidate.get(
            "hole_correction_vectors_mm", []),
        "frozen": "采纳结果冻结输入孔系、匹配关系与随机种子",
    }
    return HolePatternCreate.model_validate(data)
