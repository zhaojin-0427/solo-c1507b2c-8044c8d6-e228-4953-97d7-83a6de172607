"""孔系装配分析：Pydantic 校验、最坏边界 / MC 求解、版本冻结与整改采纳。"""
import math

import numpy as np
import pytest

from app.hole_optimizer import RemedyError, search_remedies
from app.hole_schemas import (
    DatumFeatureSpec,
    DatumFrameSpec,
    HolePatternCreate,
    RemedySearchRequest,
)
from app.holes import (
    HoleError,
    analyze,
    build_model,
    monte_carlo,
    solve_pose,
    worst_case,
)


# ------------------------------------------------------------- 测试构造器

def hole_feat(x, y, **kw):
    base = dict(
        kind="hole", x=x, y=y, nominal_diameter=10.0,
        diameter_upper_deviation=0.1, diameter_lower_deviation=0.0,
        position_tolerance=0.2, material_condition="MMC",
        distribution="normal", diameter_std_dev=0.02,
        position_std_dev=0.02)
    base.update(kw)
    return base


def float_mate(mid, x, y, bolt_up=9.75, bolt_lo=9.65, off=(0.0, 0.0)):
    return {
        "id": mid,
        "feature_a": hole_feat(x, y),
        "feature_b": hole_feat(x + off[0], y + off[1]),
        "bolt_diameter_upper": bolt_up,
        "bolt_diameter_lower": bolt_lo,
    }


def flange_payload(name="flange", mates=None, **kw):
    mates = mates or [float_mate("M1", 0, 0), float_mate("M2", 50, 0),
                     float_mate("M3", 50, 50), float_mate("M4", 0, 50)]
    base = dict(name=name, mates=mates, mc_samples=3000, random_seed=7,
                theta_search_deg=2.0, theta_grid_points=41,
                mc_theta_points=5)
    base.update(kw)
    return HolePatternCreate(**base)


# ------------------------------------------------------------- Pydantic 校验

def test_inverted_diameter_limits_rejected():
    f = hole_feat(0, 0, diameter_lower_deviation=0.5)
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[
            {"id": "M1", "feature_a": f,
             "feature_b": hole_feat(0, 0),
             "bolt_diameter_upper": 9.7, "bolt_diameter_lower": 9.6}])
    assert "倒置" in str(ei.value)


def test_inverted_bolt_limits_rejected():
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[
            {"id": "M1", "feature_a": hole_feat(0, 0),
             "feature_b": hole_feat(0, 0),
             "bolt_diameter_upper": 9.6, "bolt_diameter_lower": 9.7}])
    assert "螺栓直径极限倒置" in str(ei.value)


def test_duplicate_mate_id_rejected():
    with pytest.raises(Exception) as ei:
        flange_payload("x", mates=[float_mate("M1", 0, 0),
                                   float_mate("M1", 10, 0)])
    assert "id 重复" in str(ei.value)


def test_missing_feature_b_rejected():
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[{"id": "M1",
                                            "feature_a": hole_feat(0, 0)}])
    assert "匹配缺失" in str(ei.value)


def test_two_pins_rejected():
    pin = hole_feat(0, 0, kind="pin", nominal_diameter=9.8,
                    diameter_upper_deviation=0.05,
                    diameter_lower_deviation=-0.05)
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[
            {"id": "M1", "feature_a": pin, "feature_b": dict(pin)}])
    assert "外要素" in str(ei.value)


def test_two_holes_without_bolt_rejected():
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[
            {"id": "M1", "feature_a": hole_feat(0, 0),
             "feature_b": hole_feat(0, 0)}])
    assert "螺栓直径极限" in str(ei.value)


def test_bolt_with_pin_side_rejected():
    pin = hole_feat(0, 0, kind="pin", nominal_diameter=9.8,
                    diameter_upper_deviation=0.05,
                    diameter_lower_deviation=-0.05)
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", mates=[
            {"id": "M1", "feature_a": hole_feat(0, 0),
             "feature_b": pin,
             "bolt_diameter_upper": 9.7, "bolt_diameter_lower": 9.6}])
    assert "螺栓直径极限只能与双孔" in str(ei.value)


def test_position_datum_ref_unknown_rejected():
    f = hole_feat(0, 0, position_datum_refs=["Z"])
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", frame_a=DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="edge",
                             constrains=["tx", "ty"],
                             normal_x=0.0, normal_y=1.0),
            DatumFeatureSpec(label="B", order=2, kind="edge",
                             constrains=["rz"],
                             normal_x=1.0, normal_y=0.0,
                             angular_half_range_deg=0.5)]),
            mates=[{"id": "M1", "feature_a": f,
                    "feature_b": hole_feat(0, 0),
                    "bolt_diameter_upper": 9.7,
                    "bolt_diameter_lower": 9.6}])
    assert "不存在的基准" in str(ei.value) and "M1" in str(ei.value)


def test_position_datum_ref_skip_order_rejected():
    """引用必须是优先次序的连续前缀：只引 B（次序2）为跳级。"""
    f = hole_feat(0, 0, position_datum_refs=["B"])
    with pytest.raises(Exception) as ei:
        HolePatternCreate(name="x", frame_a=DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="edge",
                             constrains=["tx", "ty"],
                             normal_x=0.0, normal_y=1.0),
            DatumFeatureSpec(label="B", order=2, kind="edge",
                             constrains=["rz"],
                             normal_x=1.0, normal_y=0.0,
                             angular_half_range_deg=0.5)]),
            mates=[{"id": "M1", "feature_a": f,
                    "feature_b": hole_feat(0, 0),
                    "bolt_diameter_upper": 9.7,
                    "bolt_diameter_lower": 9.6}])
    assert "连续前缀" in str(ei.value)


def test_degenerate_frame_parallel_edges_rejected():
    with pytest.raises(Exception) as ei:
        DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="edge",
                             constrains=["tx"],
                             normal_x=1.0, normal_y=0.0),
            DatumFeatureSpec(label="B", order=2, kind="edge",
                             constrains=["ty"],
                             normal_x=1.0, normal_y=0.0),
            DatumFeatureSpec(label="C", order=3, kind="edge",
                             constrains=["rz"],
                             normal_x=0.0, normal_y=1.0,
                             angular_half_range_deg=0.5)])
    assert "基准退化" in str(ei.value)


def test_frame_underconstrained_rejected():
    with pytest.raises(Exception) as ei:
        DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="edge",
                             constrains=["tx"],
                             normal_x=1.0, normal_y=0.0)])
    assert "欠约束" in str(ei.value)


def test_frame_order_must_be_contiguous():
    with pytest.raises(Exception) as ei:
        DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="edge",
                             constrains=["tx", "ty"],
                             normal_x=0.0, normal_y=1.0),
            DatumFeatureSpec(label="B", order=3, kind="edge",
                             constrains=["rz"],
                             normal_x=1.0, normal_y=0.0,
                             angular_half_range_deg=0.5)])
    assert "连续" in str(ei.value)


def test_size_datum_requires_lever_for_rz():
    with pytest.raises(Exception) as ei:
        DatumFrameSpec(datums=[
            DatumFeatureSpec(label="A", order=1, kind="size",
                             constrains=["tx", "ty"],
                             nominal_diameter=10.0,
                             diameter_upper_deviation=0.05,
                             diameter_lower_deviation=-0.05),
            DatumFeatureSpec(label="B", order=2, kind="size",
                             constrains=["rz"],
                             nominal_diameter=10.0,
                             diameter_upper_deviation=0.05,
                             diameter_lower_deviation=-0.05)])
    assert "lever_radius" in str(ei.value)


# ------------------------------------------------------------- 求解器

def test_worst_case_feasible_flange():
    res = analyze(flange_payload())
    wc = res["worst_case"]
    # 名义对齐，VC 径向余量 = (10-9.75) - 0.2 = 0.05
    assert wc["feasible"]
    assert abs(wc["best_pose"]["worst_margin_mm"] - 0.05) < 1e-6
    assert wc["best_pose"]["tx_mm"] == pytest.approx(0.0, abs=1e-6)
    assert wc["first_interference_at_nominal"]["interferes"] is False
    # 四个匹配位同构
    for pm in wc["per_mate"]:
        assert pm["vc_radial_allowance_mm"] == pytest.approx(0.05, abs=1e-9)
        assert "nominal_clearance" in pm["contributions_mm"]


def test_global_offset_absorbed_by_translation():
    p = flange_payload("off", mates=[
        float_mate("M1", 0, 0, off=(1.0, 0.0)),
        float_mate("M2", 50, 0, off=(1.0, 0.0)),
        float_mate("M3", 50, 50, off=(1.0, 0.0)),
        float_mate("M4", 0, 50, off=(1.0, 0.0))])
    wc = analyze(p)["worst_case"]
    # B 整体 +1：装配变换 t = -1 完全吸收（符号约定 R·pB+t=pA）
    assert wc["feasible"]
    assert wc["best_pose"]["tx_mm"] == pytest.approx(-1.0, abs=1e-6)


def test_pattern_scale_infeasible_worst_case():
    """B 孔位相对 A 缩放（非线性错位），余量不足时最坏边界不可行。"""
    def scaled(mid, x, y, s=0.004):
        return float_mate(mid, x, y, off=(s * x, s * y))
    p = flange_payload("scale", mates=[
        scaled("M1", 0, 0), scaled("M2", 50, 0),
        scaled("M3", 50, 50), scaled("M4", 0, 50)])
    # M2/M3 错位 0.2~0.28，远超 0.05
    wc = analyze(p)["worst_case"]
    assert not wc["feasible"]
    assert wc["best_pose"]["worst_margin_mm"] < 0
    # 外围孔（非原点 M1）在最优位姿下仍有负余量
    assert any(pm["mate_id"] in {"M2", "M3", "M4"}
               and pm["margin_at_best_pose_mm"] < 0
               for pm in wc["per_mate"])


def test_monte_carlo_deterministic_and_bounded():
    p = flange_payload(mc_samples=4000)
    r1 = analyze(p)["monte_carlo"]
    r2 = analyze(p)["monte_carlo"]
    assert r1["assembly_success_rate"] == r2["assembly_success_rate"]
    assert r1["random_seed"] == 7
    assert 0.0 <= r1["assembly_success_rate"] <= 1.0
    pr = r1["feasible_pose_range"]
    for axis in ("tx_mm", "ty_mm", "theta_deg"):
        assert pr[axis]["min"] <= pr[axis]["mean"] <= pr[axis]["max"] + 1e-12
    assert sum(c["count"] for c in r1["first_interference"]) >= 0


def test_monte_carlo_reports_failures_for_tight_pattern():
    """紧配合 + 孔位偏差：出现失败且最先干涉统计指向受限匹配位。"""
    tight = [
        {"id": "P1",
         "feature_a": hole_feat(0, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.1,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.012,
                                position_std_dev=0.03),
         "feature_b": hole_feat(0.06, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.1,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.012,
                                position_std_dev=0.03),
         "bolt_diameter_upper": 9.9, "bolt_diameter_lower": 9.85},
        {"id": "P2",
         "feature_a": hole_feat(40, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.1,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.012,
                                position_std_dev=0.03),
         "feature_b": hole_feat(40.06, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.1,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.012,
                                position_std_dev=0.03),
         "bolt_diameter_upper": 9.9, "bolt_diameter_lower": 9.85},
    ]
    p = flange_payload("tight", mates=tight, mc_samples=6000,
                       theta_search_deg=1.0, theta_grid_points=21,
                       mc_theta_points=3)
    mc = analyze(p)["monte_carlo"]
    assert 0.0 < mc["failure_probability"] < 1.0
    total_fail = sum(c["count"] for c in mc["first_interference"])
    assert total_fail > 0


def test_fixed_pin_allowance_formula():
    """固定销：0.5(D_min-d_max) - 0.5(t_hole+t_pin)。"""
    pin = hole_feat(0, 0, kind="pin", nominal_diameter=9.8,
                    diameter_upper_deviation=0.05,
                    diameter_lower_deviation=-0.05,
                    position_tolerance=0.1, material_condition="MMC",
                    distribution="uniform", diameter_std_dev=None)
    h = hole_feat(0, 0, nominal_diameter=10.0,
                  diameter_upper_deviation=0.1, diameter_lower_deviation=0.0,
                  position_tolerance=0.1, material_condition="RFS",
                  distribution="uniform", diameter_std_dev=None)
    p = HolePatternCreate(name="pin", mc_samples=2000, random_seed=3,
                          mates=[{"id": "P1", "feature_a": h,
                                  "feature_b": pin}])
    wc = worst_case(build_model(p))
    # 0.5*(10.0-9.85) - 0.1 = 0.075 - 0.1 = -0.025
    assert wc["best_pose"]["worst_margin_mm"] == pytest.approx(-0.025, abs=1e-9)
    assert not wc["feasible"]


def test_solver_matches_bruteforce_disc():
    """两圆盘公共交集：子梯度解与闭式（圆心距-半径）一致。"""
    PA = np.array([[0.0, 0.0], [10.0, 0.0]])
    PB = PA.copy()
    T = np.array([[0.0, 0.0], [10.1, 0.0]])
    rad = np.array([0.1, 0.1])
    sol = solve_pose(PA, PB, T, rad, theta_max=0.01, theta_scan=11)
    # 圆心错位 0.1、半径各 0.1：公共交集切比雪夫中心 t=0.05，半径余量 0.05
    assert sol["feasible"]
    assert sol["tx"] == pytest.approx(0.05, abs=1e-4)
    assert sol["margin"] == pytest.approx(0.05, abs=1e-4)
    T2 = np.array([[0.0, 0.0], [10.3, 0.0]])
    sol2 = solve_pose(PA, PB, T2, rad, theta_max=0.01, theta_scan=11)
    # 中心距 0.3 > 半径和 0.2：两圆盘在 t=0.15 相切，最差缺口 0.05
    assert not sol2["feasible"]
    assert sol2["margin"] == pytest.approx(-0.05, abs=2e-3)


def test_unit_conversion_inches():
    """英寸输入：内部 mm 结果按 25.4 缩放。"""
    def in_mate(mid, x, y):
        f = hole_feat(x, y, nominal_diameter=0.4,
                      diameter_upper_deviation=0.004,
                      diameter_lower_deviation=0.0,
                      position_tolerance=0.008,
                      diameter_std_dev=0.0008,
                      position_std_dev=0.0008)
        return {"id": mid, "feature_a": f, "feature_b": dict(f),
                "bolt_diameter_upper": 0.39, "bolt_diameter_lower": 0.386}
    p = HolePatternCreate(name="inch", unit="in", mc_samples=2000,
                          mates=[in_mate("M1", 0, 0),
                                 in_mate("M2", 2.0, 0)],
                          theta_search_deg=2.0)
    model = build_model(p)
    assert model.features_a[0].nom == pytest.approx(0.4 * 25.4)
    wc = worst_case(model)
    # (0.4-0.39)*25.4 - 0.008*25.4 = 0.254 - 0.2032 = 0.0508
    assert wc["best_pose"]["worst_margin_mm"] == pytest.approx(0.0508, abs=1e-6)


# ----------------------------------------------------- 基准框架参与模型

def test_size_datum_shift_relieves_position():
    """MMC 尺寸基准存在间隙时，基准偏移计入有效位置度（MC 成功率提升）。"""
    frame = DatumFrameSpec(datums=[
        DatumFeatureSpec(label="A", order=1, kind="size",
                         constrains=["tx", "ty"],
                         nominal_diameter=10.0,
                         diameter_upper_deviation=0.2,
                         diameter_lower_deviation=0.0,
                         material_condition="MMC",
                         distribution="uniform"),
        DatumFeatureSpec(label="B", order=2, kind="edge",
                         constrains=["rz"],
                         normal_x=1.0, normal_y=0.0,
                         angular_half_range_deg=0.2)])

    def mate(mid, x, y, refs):
        return {"id": mid, "feature_a": hole_feat(x, y,
                position_tolerance=0.05, position_datum_refs=refs,
                material_condition="MMC"),
                "feature_b": hole_feat(x, y, position_tolerance=0.05),
                "bolt_diameter_upper": 9.75, "bolt_diameter_lower": 9.65}

    p = HolePatternCreate(name="datum", frame_a=frame, mc_samples=4000,
                          random_seed=11, theta_search_deg=2.0,
                          mates=[mate("M1", 0, 0, ["A"]),
                                 mate("M2", 40, 0, ["A"])])
    model = build_model(p)
    mc = monte_carlo(model, 4000, seed=11)
    # 引用 MMB 基准 A：基准孔间隙最多 0.2 计入偏移，位置度实际被放宽
    assert mc["assembly_success_rate"] > 0.9


# ------------------------------------------------------------- 整改搜索

def _remedy_model():
    # 名义有 0.06 错位、紧余量的两孔法兰：现状部分失败
    tight = [
        {"id": "P1",
         "feature_a": hole_feat(0, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.08,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.01,
                                position_std_dev=0.02),
         "feature_b": hole_feat(0.08, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.08,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.01,
                                position_std_dev=0.02),
         "bolt_diameter_upper": 9.9, "bolt_diameter_lower": 9.85},
        {"id": "P2",
         "feature_a": hole_feat(40, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.08,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.01,
                                position_std_dev=0.02),
         "feature_b": hole_feat(40.08, 0, nominal_diameter=10.0,
                                diameter_upper_deviation=0.05,
                                diameter_lower_deviation=0.0,
                                position_tolerance=0.08,
                                material_condition="RFS",
                                distribution="normal",
                                diameter_std_dev=0.01,
                                position_std_dev=0.02),
         "bolt_diameter_upper": 9.9, "bolt_diameter_lower": 9.85},
    ]
    p = flange_payload("remedy", mates=tight, mc_samples=4000,
                       theta_search_deg=1.0, theta_grid_points=21,
                       mc_theta_points=3)
    return build_model(p)


def test_remedy_baseline_present_and_ranked():
    model = _remedy_model()
    req = RemedySearchRequest(name="s1", mc_samples=2000,
                              drill_options=[], fastener_options=[],
                              correction_options=[])
    res = search_remedies(model, req)
    assert res["baseline"] is not None
    assert res["baseline"]["max_hole_shift_mm"] == 0.0
    assert res["candidates"][0]["rank"] == 1


def test_remedy_unknown_mate_rejected():
    model = _remedy_model()
    req = RemedySearchRequest(
        name="s", drill_options=[
            {"mate_id": "NOPE", "side": "a", "nominal_diameter": 11.0,
             "diameter_upper_deviation": 0.05,
             "diameter_lower_deviation": 0.0}])
    with pytest.raises(RemedyError) as ei:
        search_remedies(model, req)
    assert "NOPE" in str(ei.value)


def test_remedy_drill_must_enlarge():
    model = _remedy_model()
    req = RemedySearchRequest(
        name="s", drill_options=[
            {"mate_id": "P1", "side": "a", "nominal_diameter": 10.0,
             "diameter_upper_deviation": 0.05,
             "diameter_lower_deviation": 0.0}])
    with pytest.raises(RemedyError) as ei:
        search_remedies(model, req)
    assert "放大" in str(ei.value)


def test_remedy_lock_conflict_rejected():
    model = _remedy_model()
    req = RemedySearchRequest(
        name="s", locked_mates=["P1"],
        correction_options=[{"mate_id": "P1", "max_shift": 0.2}])
    with pytest.raises(RemedyError) as ei:
        search_remedies(model, req)
    assert "锁定" in str(ei.value)


def test_remedy_improves_failure_probability():
    """放大孔 / 缩小螺栓 / 孔位修正都应降低失败概率（现状最差）。"""
    model = _remedy_model()
    req = RemedySearchRequest(
        name="s", mc_samples=4000,
        drill_options=[
            {"mate_id": "P1", "side": "a", "nominal_diameter": 10.6,
             "diameter_upper_deviation": 0.05,
             "diameter_lower_deviation": 0.0},
            {"mate_id": "P2", "side": "a", "nominal_diameter": 10.6,
             "diameter_upper_deviation": 0.05,
             "diameter_lower_deviation": 0.0}],
        correction_options=[{"mate_id": "P1", "max_shift": 0.2},
                            {"mate_id": "P2", "max_shift": 0.2}])
    res = search_remedies(model, req)
    base_fail = res["baseline"]["failure_probability"]
    top = res["candidates"][0]
    assert top["failure_probability"] <= base_fail
    # 排序：失败概率相等时按最大孔位改动、孔径放大总量升序
    keys = [(c["failure_probability"], c["max_hole_shift_mm"],
             c["hole_enlargement_total_mm"]) for c in res["candidates"]]
    assert keys == sorted(keys)


# ----------------------------------------------------- 数据处理回归

def test_bolt_diameter_samples_cover_full_interval():
    """螺栓极限 9~11 均匀分布：样本必须覆盖完整区间（不得退化成上限）。"""
    from app.holes import sample_realization
    fixed = hole_feat(0, 0, nominal_diameter=10.0,
                      diameter_upper_deviation=0.0,
                      diameter_lower_deviation=0.0,
                      position_tolerance=0.0, material_condition="RFS",
                      distribution="uniform",
                      diameter_std_dev=None, position_std_dev=None)
    # 浮动螺栓（双孔）
    p_float = HolePatternCreate(name="b-float", mc_samples=20000, random_seed=7,
        mates=[{"id": "M1", "feature_a": dict(fixed), "feature_b": dict(fixed),
                "bolt_diameter_upper": 11.0, "bolt_diameter_lower": 9.0}])
    m = build_model(p_float)
    bolt = sample_realization(m, 20000, seed=7)["bolt"][:, 0]
    assert bolt.min() < 9.2 and bolt.max() > 10.8
    assert bolt.mean() == pytest.approx(10.0, abs=0.05)
    rate = analyze(p_float)["monte_carlo"]["assembly_success_rate"]
    # 孔固定 Φ10：螺栓 ≤10 才能装入，成功率约 1/2
    assert 0.40 < rate < 0.60

    # 固定螺栓（无 feature_b）同样覆盖区间
    p_fixed = HolePatternCreate(name="b-fixed", mc_samples=20000, random_seed=7,
        mates=[{"id": "M1", "feature_a": dict(fixed),
                "bolt_diameter_upper": 11.0, "bolt_diameter_lower": 9.0}])
    m2 = build_model(p_fixed)
    pin = sample_realization(m2, 20000, seed=7)["diameters"][("B", 0)]
    assert pin.min() < 9.2 and pin.max() > 10.8
    assert pin.mean() == pytest.approx(10.0, abs=0.05)


def test_correction_shift_is_deterministic_and_frozen():
    """孔位修正导出确定性平移矢量：采纳后重算失败概率与候选一致。"""
    from app.hole_optimizer import freeze_payload, search_remedies

    def mate(mid, ax, ay, bx, by):
        f = dict(kind="hole", nominal_diameter=10.0,
                 diameter_upper_deviation=0.0, diameter_lower_deviation=0.0,
                 position_tolerance=0.0, material_condition="RFS",
                 distribution="uniform")
        return {"id": mid,
                "feature_a": {**f, "x": ax, "y": ay},
                "feature_b": {**f, "x": bx, "y": by},
                "bolt_diameter_upper": 10.0, "bolt_diameter_lower": 10.0}

    p = HolePatternCreate(name="corr", mc_samples=2000, random_seed=7,
        mates=[mate("M1", 0, 0, 0.2, 0), mate("M2", 40, 0, 39.8, 0)])
    model = build_model(p)
    req = RemedySearchRequest(
        name="fix", mc_samples=1000, random_seed=7,
        correction_options=[{"mate_id": "M1", "max_shift": 0.25},
                            {"mate_id": "M2", "max_shift": 0.25}])
    res = search_remedies(model, req)
    assert res["baseline"]["failure_probability"] == 1.0
    top = res["candidates"][0]
    assert top["uses_hole_correction"]
    assert top["failure_probability"] == 0.0
    vec = {v["mate_id"]: (v["shift_x_mm"], v["shift_y_mm"])
           for v in top["hole_correction_vectors_mm"]}
    assert vec["M1"][0] == pytest.approx(0.2, abs=1e-9)
    assert vec["M2"][0] == pytest.approx(-0.2, abs=1e-9)
    assert top["max_hole_shift_mm"] == pytest.approx(0.2, abs=1e-9)

    # 冻结为新版本输入并重算：失败概率必须保持 0
    new_payload = freeze_payload(model, top, "corr-adopted", "", p,
                                 mc_samples=1000, seed=7)
    assert new_payload.mc_samples == 1000
    assert new_payload.random_seed == 7
    xs = [m.feature_a.x for m in new_payload.mates]
    assert xs[0] == pytest.approx(0.2, abs=1e-9)
    assert xs[1] == pytest.approx(39.8, abs=1e-9)
    adopted = analyze(new_payload)
    assert adopted["monte_carlo"]["failure_probability"] == \
        top["failure_probability"]


# ------------------------------------------------------------- API

def test_api_create_get_freeze_and_422(client):
    payload = flange_payload().model_dump(mode="json")
    payload["name"] = "api-flange"
    r = client.post("/hole-patterns", json=payload)
    assert r.status_code == 201, r.text
    j = r.json()
    vid = j["version_id"]
    assert j["frozen"] is True
    got = client.get(f"/hole-versions/{vid}").json()
    assert got["result"] == j["result"]  # 冻结复现
    assert client.get("/hole-versions/9999").status_code == 404

    bad = dict(payload)
    bad["name"] = "api-bad"
    bad["mates"][0]["feature_a"]["diameter_lower_deviation"] = 0.9
    assert client.post("/hole-patterns", json=bad).status_code == 422


def test_api_versions_line_and_remedy_adopt(client):
    payload = flange_payload().model_dump(mode="json")
    payload["name"] = "api-line"
    j = client.post("/hole-patterns", json=payload).json()
    pid, vid = j["pattern_id"], j["version_id"]

    # 新版本
    v2_payload = dict(payload)
    v2_payload["note"] = "v2"
    r = client.post(f"/hole-patterns/{pid}/versions", json=v2_payload)
    assert r.status_code == 201, r.text
    assert r.json()["version_no"] == 2
    assert client.post(f"/hole-patterns/{pid}/versions",
                       json={**v2_payload, "parent_version_id": 9999}
                       ).status_code == 404
    assert client.post("/hole-patterns/9999/versions",
                       json=v2_payload).status_code == 404

    # 整改搜索
    rem = {"name": "rem", "mc_samples": 2000,
           "drill_options": [
               {"mate_id": "M1", "side": "a", "nominal_diameter": 10.5,
                "diameter_upper_deviation": 0.1,
                "diameter_lower_deviation": 0.0}],
           "correction_options": [{"mate_id": "M2", "max_shift": 0.3}]}
    rr = client.post(f"/hole-versions/{vid}/remedies", json=rem)
    assert rr.status_code == 201, rr.text
    rj = rr.json()
    rid = rj["remedy_id"]
    assert rj["result"]["baseline"] is not None
    assert client.get(f"/hole-versions/{vid}/remedies").status_code == 200
    assert client.get(f"/hole-remedies/{rid}").status_code == 200

    # 采纳 rank 1 -> 新版本 3，种子冻结
    sr = client.post(f"/hole-remedies/{rid}/select", json={"rank": 1})
    assert sr.status_code == 200, sr.text
    sj = sr.json()
    assert sj["version_no"] == 3
    assert sj["random_seed"] == payload["random_seed"]
    assert sj["adopted_from"]["rank"] == 1
    # 重复采纳拒绝
    assert client.post(f"/hole-remedies/{rid}/select",
                       json={"rank": 2}).status_code == 422
    # rank 越界
    assert client.post(f"/hole-remedies/{rid}/select",
                       json={"rank": 999}).status_code == 422
    assert client.post("/hole-remedies/9999/select",
                       json={"rank": 1}).status_code == 404

    lst = client.get("/hole-patterns").json()["hole_patterns"]
    me = next(p for p in lst if p["pattern_id"] == pid)
    assert [v["version_no"] for v in me["versions"]] == [1, 2, 3]


def test_api_remedy_unknown_mate_422(client):
    payload = flange_payload().model_dump(mode="json")
    payload["name"] = "api-rem-bad"
    vid = client.post("/hole-patterns", json=payload).json()["version_id"]
    r = client.post(f"/hole-versions/{vid}/remedies", json={
        "name": "x",
        "fastener_options": [{"mate_id": "GHOST",
                              "diameter_upper": 9.0,
                              "diameter_lower": 8.9}]})
    assert r.status_code == 422 and "GHOST" in r.text


def test_api_adopt_keeps_correction_samples_and_seed(client):
    """采纳孔位修正：新版本保留修正效果，且 mc_samples/seed 取整改请求值。"""
    f = dict(kind="hole", nominal_diameter=10.0,
             diameter_upper_deviation=0.0, diameter_lower_deviation=0.0,
             position_tolerance=0.0, material_condition="RFS",
             distribution="uniform")

    def mate(mid, ax, bx):
        return {"id": mid,
                "feature_a": {**f, "x": ax[0], "y": ax[1]},
                "feature_b": {**f, "x": bx[0], "y": bx[1]},
                "bolt_diameter_upper": 10.0, "bolt_diameter_lower": 10.0}

    payload = {"name": "api-adopt", "mc_samples": 5000, "random_seed": 7,
               "mates": [mate("M1", (0, 0), (0.2, 0)),
                         mate("M2", (40, 0), (39.8, 0))]}
    vid = client.post("/hole-patterns", json=payload).json()["version_id"]
    rem = {"name": "fix", "mc_samples": 1000, "random_seed": 7,
           "correction_options": [{"mate_id": "M1", "max_shift": 0.25},
                                  {"mate_id": "M2", "max_shift": 0.25}]}
    rj = client.post(f"/hole-versions/{vid}/remedies", json=rem).json()
    top = rj["result"]["candidates"][0]
    assert top["failure_probability"] == 0.0
    sr = client.post(f"/hole-remedies/{rj['remedy_id']}/select",
                     json={"rank": top["rank"]})
    assert sr.status_code == 200, sr.text
    v = sr.json()
    assert v["mc_samples"] == 1000
    assert v["result"]["monte_carlo"]["samples"] == 1000
    assert v["result"]["monte_carlo"]["random_seed"] == 7
    assert v["result"]["monte_carlo"]["failure_probability"] == \
        top["failure_probability"]
    # 修正量已固化为 A 侧名义孔位
    xs = [m["feature_a"]["x"] for m in v["submitted_input"]["mates"]]
    assert xs[0] == pytest.approx(0.2, abs=1e-9)
    assert xs[1] == pytest.approx(39.8, abs=1e-9)
    # 重新 GET 冻结版本结果不变
    got = client.get(f"/hole-versions/{v['version_id']}").json()
    assert got["result"]["monte_carlo"]["failure_probability"] == 0.0
    assert got["mc_samples"] == 1000
