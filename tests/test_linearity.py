"""量具线性与偏倚研究测试。

覆盖：Pydantic 校验（单位 / 标准件编号 / 重复组合 / 覆盖范围）、逐点平均
偏倚与置信区间、以标准不确定度加权的偏倚–参考值回归（截距 / 斜率 / 残差 /
系数协方差 / 失配检验）、参考点不足与覆盖过窄不外推、复制研究排除异常
读数比较结论、草稿→定稿→采用状态机、采用研究接入测量方案后的逆回归
修正与不确定度传播（GUM/MC、量程外不修正、快照冻结、历史结果不变）。
"""
import math

import numpy as np
import pytest

from app.linearity import (
    apply_linear_bias_model,
    bias_model_snapshot,
    compare_results,
    run_study,
    t_critical,
)
from app.linearity_schemas import LinearityStudyCreate


# ------------------------------------------------------------- 造数辅助

def _point(pid, serial, x, biases=(-0.002, 0.0, 0.002), *,
           u_ref=0.001, operators=("A", "B", "A"), unit="mm",
           value_unit=None, order_start=None):
    """构造一个三读数核查点（偏倚扰动在调用处给定）。"""
    readings = []
    base = order_start if order_start is not None else 1
    for r, d in enumerate(biases):
        readings.append({
            "replicate": r + 1,
            "value": x + d if value_unit is None else x + d,
            "operator": operators[r],
            "measurement_order": base + r,
        })
    return {
        "point_id": pid, "standard_serial": serial,
        "reference_value": x,
        "reference_unit": unit,
        "reference_std_uncertainty": u_ref,
        "readings": readings,
    }


def _study_payload(xs=(49.1, 49.55, 50.0, 50.45, 50.9), *,
                   wl=49.0, wu=51.0, bias_fn=None, process_tolerance=0.08,
                   u_ref=0.0002, noise=0.0005, name="lin", min_cov=0.5,
                   unit="mm", extra=None):
    if bias_fn is None:
        def bias_fn(x):
            return -0.02 + 0.0005 * (x - 50.0)
    points = []
    order = 1
    for i, x in enumerate(xs, 1):
        center_bias = bias_fn(x)
        p = _point(f"P{i}", f"GB-S-{i}", x,
                   biases=(center_bias - noise, center_bias,
                           center_bias + noise),
                   u_ref=u_ref, order_start=order)
        if unit != "mm":
            p["reference_unit"] = unit
        points.append(p)
        order += 3
    payload = {
        "name": name, "dimension_id": "L1", "unit": unit,
        "working_range_lower": wl, "working_range_upper": wu,
        "process_tolerance": process_tolerance,
        "min_span_coverage": min_cov, "points": points,
    }
    if extra:
        payload.update(extra)
    return payload


def _make_chain(client, payload=None):
    chain = payload or {
        "name": "lin-chain",
        "closure_lower_limit": 0.05,
        "closure_upper_limit": 0.60,
        "dimensions": [
            {"id": "L1", "start": "a", "end": "b", "nominal": 50,
             "upper_deviation": 0.08, "lower_deviation": 0.0,
             "std_dev": 0.02, "distribution": "normal"},
            {"id": "L2", "start": "b", "end": "c", "nominal": 30.3,
             "upper_deviation": 0.05, "lower_deviation": -0.05,
             "distribution": "uniform"},
            {"id": "L3", "start": "c", "end": "a", "nominal": 80,
             "upper_deviation": 0.0, "lower_deviation": -0.15,
             "std_dev": 0.04, "distribution": "normal", "direction": -1},
        ],
        "mc_samples": 20000, "random_seed": 7,
    }
    r = client.post("/chains", json=chain)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


# ------------------------------------------------------------- 统计核心

def test_t_critical_values():
    assert t_critical(0.975, math.inf) == pytest.approx(1.96, abs=0.001)
    assert t_critical(0.975, 10) == pytest.approx(2.228, abs=0.002)
    assert t_critical(0.975, 2) == pytest.approx(4.303, abs=0.003)
    assert t_critical(0.95, math.inf) == pytest.approx(1.645, abs=0.001)


def test_weighted_regression_recovers_known_line():
    p = LinearityStudyCreate(**_study_payload())
    res = run_study(p)
    reg = res["regression"]
    # b(x) = -0.02 + 0.0005·(x−50) = -0.045 + 0.0005·x
    assert reg["slope"] == pytest.approx(0.0005, abs=2e-4)
    assert reg["intercept_mm"] == pytest.approx(-0.045, abs=3e-3)
    assert reg["slope_block"]["significant"]
    assert reg["intercept"]["significant"]
    # GUM 协方差对称、半正定且对角为正
    cov = np.array(reg["coefficient_covariance"]
                   ["gum_including_reference_mm2"])
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-18
    assert cov[0, 0] > 0 and cov[1, 1] > 0
    # 残差在噪声量级；加权 R² 高；失配检验不拒绝线性模型
    assert reg["weighted_r_squared"] > 0.9
    assert reg["lack_of_fit"]["p_value"] > 0.05
    assert reg["lack_of_fit"]["fits_declared_uncertainties"] is True
    # 适用量程即参考值 min/max
    assert reg["applicable_range_mm"] == pytest.approx([49.1, 50.9])


def test_gum_covariance_matches_manual_propagation():
    p = LinearityStudyCreate(**_study_payload())
    res = run_study(p)
    reg = res["regression"]
    xs = np.array([pt["reference_value"]
                   for pt in p.model_dump(mode="json")["points"]])
    b = np.array([q["mean_bias_mm"] for q in res["points"]])
    u = np.array([q["combined_std_uncertainty_mm"] for q in res["points"]])
    X = np.column_stack([np.ones(len(xs)), xs])
    W = np.diag(1 / u ** 2)
    A = np.linalg.inv(X.T @ W @ X) @ X.T @ W
    cov = A @ np.diag(u ** 2) @ A.T
    assert np.allclose(
        reg["coefficient_covariance"]["gum_including_reference_mm2"], cov)
    # 权重即逆方差时 Cov_β = (XᵀWX)⁻¹
    assert np.allclose(cov, np.linalg.inv(X.T @ W @ X))


def test_per_point_bias_ci_and_significance():
    # 偏倚 b(x)=0.005·(x−50)：端点 ±0.0045 显著、中点为 0 不显著
    payload = _study_payload(
        u_ref=0.001, noise=0.002,
        bias_fn=lambda x: 0.005 * (x - 50.0))
    p = LinearityStudyCreate(**payload)
    res = run_study(p)
    for q, pt in zip(res["points"], payload["points"]):
        x = pt["reference_value"]
        expected_bias = 0.005 * (x - 50.0)
        assert q["mean_bias_mm"] == pytest.approx(expected_bias, abs=3e-3)
        lo, hi = q["confidence_interval_mm"]
        assert lo <= q["mean_bias_mm"] <= hi
        assert q["combined_std_uncertainty_mm"] > 0
        # u_ref 与读数均值标准差方和根
        u_rep = 0.002 / math.sqrt(3)
        assert q["combined_std_uncertainty_mm"] == pytest.approx(
            math.sqrt(0.001 ** 2 + u_rep ** 2), rel=1e-9)
    sig = [q["bias_significant"] for q in res["points"]]
    assert sig[0] and sig[-1]
    assert not sig[2]


def test_reference_uncertainty_widens_se_and_can_remove_significance():
    big = LinearityStudyCreate(**_study_payload(u_ref=0.02))
    small = LinearityStudyCreate(**_study_payload(u_ref=0.0001))
    rb, rs = run_study(big), run_study(small)
    assert rb["regression"]["slope_block"]["std_error"] > \
        rs["regression"]["slope_block"]["std_error"]
    assert rs["regression"]["slope_block"]["significant"] is True
    assert rb["regression"]["slope_block"]["significant"] is False


def test_zero_bias_study_not_significant():
    p = LinearityStudyCreate(**_study_payload(bias_fn=lambda x: 0.0))
    res = run_study(p)
    assert res["regression"]["slope_block"]["significant"] is False
    assert res["regression"]["intercept"]["significant"] is False
    assert res["linearity"]["linearity_present"] is False
    assert res["mean_bias"]["bias_significant"] is False


def test_two_points_still_fits_but_no_classic_inference():
    p = LinearityStudyCreate(**_study_payload(xs=(49.2, 50.8)))
    res = run_study(p)
    assert res["regression_available"] is True
    reg = res["regression"]
    assert reg["classic"]["degrees_of_freedom"] == 0
    assert reg["classic"]["coefficient_covariance_mm2"] is None
    # GUM 协方差仍可用
    assert reg["slope_block"]["confidence_interval"] is not None
    assert any("至少使用 5 个标准件" in w for w in res["warnings"])


def test_single_reading_points_unestimable_repeatability():
    payload = _study_payload()
    # 每点只留 1 个读数；测量次序仍全研究唯一
    for pt in payload["points"]:
        pt["readings"] = [pt["readings"][1]]
    p = LinearityStudyCreate(**payload)
    res = run_study(p)
    # 无任何重复：重复性不可估计，权重只反映 u(ref)
    assert any("无重复读数" in w for w in res["warnings"])
    assert res["design"]["pooled_repeatability_std_mm"] is None
    for q in res["points"]:
        assert q["repeatability_basis"] == "unestimable"
        assert q["u_repeatability_of_mean_mm"] is None


def test_mixed_single_and_repeated_uses_pooled_sigma():
    payload = _study_payload()
    # P2 只保留单次读数，其余点三次重复
    payload["points"][1]["readings"] = [payload["points"][1]["readings"][0]]
    p = LinearityStudyCreate(**payload)
    res = run_study(p)
    q2 = res["points"][1]
    assert q2["repeatability_basis"] == "pooled_within_std"
    assert q2["u_repeatability_of_mean_mm"] == pytest.approx(
        res["design"]["pooled_repeatability_std_mm"])


def test_coverage_too_narrow_not_adoptable():
    p = LinearityStudyCreate(**_study_payload(
        xs=(49.9, 50.0, 50.1), wl=40.0, wu=60.0, min_cov=0.5))
    res = run_study(p)
    assert res["coverage"]["coverage_adequate"] is False
    assert res["adoptable"] is False
    assert any("覆盖过窄" in b for b in res["adoption_blockers"])
    # 仍有回归结果（拟合本身可用），但不得外推
    assert res["regression_available"] is True
    assert res["regression"]["applicable_range_mm"] == pytest.approx(
        [49.9, 50.1])


def test_zero_uncertainty_point_equal_weights_fallback():
    payload = _study_payload(u_ref=0.0)
    # 噪声仍给重复性，故 u_g>0；改为全部单次且 u_ref=0 → u_g=0
    for pt in payload["points"]:
        pt["readings"] = [pt["readings"][1]]
        pt["reference_std_uncertainty"] = 0.0
    p = LinearityStudyCreate(**payload)
    res = run_study(p)
    assert "equal_weights" in res["regression"]["weight_policy"]
    assert any("标准不确定度为 0" in w for w in res["warnings"])


# ------------------------------------------------------------- Pydantic 校验

def test_schema_rejects_duplicate_standard_serial():
    payload = _study_payload()
    payload["points"][1]["standard_serial"] = payload["points"][0]["standard_serial"]
    with pytest.raises(Exception) as ei:
        LinearityStudyCreate(**payload)
    assert "标准件编号重复" in str(ei.value)


def test_schema_rejects_duplicate_operator_replicate():
    payload = _study_payload()
    payload["points"][0]["readings"][1]["operator"] = "A"
    payload["points"][0]["readings"][1]["replicate"] = 1
    with pytest.raises(Exception) as ei:
        LinearityStudyCreate(**payload)
    assert "重复序号组合重复" in str(ei.value)


def test_schema_rejects_duplicate_measurement_order():
    payload = _study_payload()
    payload["points"][2]["readings"][0]["measurement_order"] = 1
    with pytest.raises(Exception) as ei:
        LinearityStudyCreate(**payload)
    assert "测量次序" in str(ei.value)


def test_schema_rejects_bad_working_range_and_nonfinite():
    with pytest.raises(Exception) as ei:
        LinearityStudyCreate(**_study_payload(wl=51.0, wu=49.0))
    assert "下限" in str(ei.value)
    payload = _study_payload()
    payload["points"][0]["readings"][0]["value"] = float("nan")
    with pytest.raises(Exception):
        LinearityStudyCreate(**payload)


def test_schema_rejects_negative_reference_uncertainty():
    payload = _study_payload()
    payload["points"][0]["reference_std_uncertainty"] = -0.001
    with pytest.raises(Exception):
        LinearityStudyCreate(**payload)


def test_schema_requires_two_points():
    payload = _study_payload()
    payload["points"] = payload["points"][:1]
    with pytest.raises(Exception):
        LinearityStudyCreate(**payload)


def test_reference_outside_working_range_rejected_by_engine():
    # 跨单位的越界 Pydantic 看不到，引擎换算后必须 422
    payload = _study_payload()
    payload["points"][0]["reference_unit"] = "cm"
    payload["points"][0]["reference_value"] = 6.0  # 60 mm，超出 49~51
    p = LinearityStudyCreate(**payload)  # schema 通过
    with pytest.raises(Exception) as ei:
        run_study(p)
    assert "超出量具工作量程" in str(ei.value)


def test_mixed_units_are_converted():
    # 一个核查点参考值用 in（50 mm = 1.968504 in），其余用 mm
    payload = _study_payload(xs=(50.0, 50.8))
    payload["points"][0]["reference_unit"] = "in"
    payload["points"][0]["reference_value"] = 50.0 / 25.4
    p = LinearityStudyCreate(**payload)
    res = run_study(p)
    assert res["points"][0]["reference_value"]["mm"] == pytest.approx(
        50.0, abs=1e-6)
    assert res["points"][0]["reference_value"]["unit"] == "in"


# ------------------------------------------------------------- 状态机 / API

def _create_study(client, cid, payload=None):
    payload = payload or _study_payload()
    r = client.post(f"/chains/{cid}/linearity-studies", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _full_lifecycle(client, cid):
    j = _create_study(client, cid)
    sid = j["study_id"]
    assert j["status"] == "draft"
    assert client.post(f"/linearity-studies/{sid}/finalize",
                       json={"note": "评审定稿"}).status_code == 200
    r = client.post(f"/linearity-studies/{sid}/adopt",
                    json={"note": "用于 L1 修正"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "adopted"
    assert r.json()["bias_model"]["applicable_range_mm"] == \
        pytest.approx([49.1, 50.9])
    return sid


def test_full_lifecycle_and_state_transitions(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    # 重复定稿 / 重复采用 / 草案跳转均拒绝
    r = client.post(f"/linearity-studies/{sid}/finalize", json={})
    assert r.status_code == 422
    r = client.post(f"/linearity-studies/{sid}/adopt", json={})
    assert r.status_code == 422
    r = client.post(f"/linearity-studies/{sid}/copy", json={})
    assert r.status_code == 422


def test_unknown_dimension_rejected(client):
    cid = _make_chain(client)
    payload = _study_payload()
    payload["dimension_id"] = "L9"
    r = client.post(f"/chains/{cid}/linearity-studies", json=payload)
    assert r.status_code == 422


def test_draft_can_copy_exclude_reading_and_compare(client):
    cid = _make_chain(client)
    j = _create_study(client, cid)
    sid = j["study_id"]
    # 排除 P3 中点读数（制造一个异常：把该读数改成大偏倚后再排除）
    # 直接排除正常读数比较结构即可
    r = client.post(f"/linearity-studies/{sid}/copy", json={
        "name": "lin-copy",
        "exclusions": [{"point_id": "P3", "operator": "B",
                        "replicate": 2, "reason": "测量时碰动标准件"}],
    })
    assert r.status_code == 201, r.text
    cj = r.json()
    assert cj["parent_study_id"] == sid
    assert cj["version_no"] == 2
    assert cj["exclusions"][0]["reason"] == "测量时碰动标准件"
    comp = cj["comparison_with_parent"]
    p3 = next(t for t in comp["points"] if t["point_id"] == "P3")
    assert p3["reading_count_before"] == 3
    assert p3["reading_count_after"] == 2
    assert comp["regression_before"] is not None
    assert comp["regression_after"] is not None
    # 父研究不变
    g = client.get(f"/linearity-studies/{sid}").json()
    assert g["status"] == "draft"
    assert sum(q["reading_count"] for q in g["result"]["points"]) == 15


def test_copy_exclusion_unknown_reading_rejected(client):
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    r = client.post(f"/linearity-studies/{sid}/copy", json={
        "exclusions": [{"point_id": "P3", "operator": "Z", "replicate": 1,
                        "reason": "x"}]})
    assert r.status_code == 422


def test_copy_exclusion_reason_required(client):
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    r = client.post(f"/linearity-studies/{sid}/copy", json={
        "exclusions": [{"point_id": "P3", "operator": "B",
                        "replicate": 2, "reason": "   "}]})
    assert r.status_code == 422


def test_copy_cannot_empty_points(client):
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    # 排除到只剩一个有效点：每点 3 条读数，排除 4 个点的全部读数
    excl = []
    for pid, ops in [("P1", ("A", "B", "A")), ("P2", ("A", "B", "A")),
                     ("P3", ("A", "B", "A")), ("P4", ("A", "B", "A"))]:
        for r_, op in enumerate(ops, 1):
            excl.append({"point_id": pid, "operator": op,
                         "replicate": r_, "reason": "清空"})
    r = client.post(f"/linearity-studies/{sid}/copy",
                    json={"exclusions": excl})
    assert r.status_code == 422


def test_copy_dropping_whole_standard_point_succeeds(client):
    """整点三条读数全部标为异常：其余 4 点仍可拟合，副本 201 且
    comparison 中该点标记 dropped、after 字段为 None（不产生 500）。"""
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    excl = [
        {"point_id": "P3", "operator": "A", "replicate": 1,
         "reason": "测量时碰动标准件"},
        {"point_id": "P3", "operator": "B", "replicate": 2,
         "reason": "测量时碰动标准件"},
        {"point_id": "P3", "operator": "A", "replicate": 3,
         "reason": "测量时碰动标准件"},
    ]
    r = client.post(f"/linearity-studies/{sid}/copy",
                    json={"name": "drop-P3", "exclusions": excl})
    assert r.status_code == 201, r.text
    j = r.json()
    comp = j["comparison_with_parent"]
    p3 = next(t for t in comp["points"] if t["point_id"] == "P3")
    assert p3["dropped"] is True
    assert p3["present_before"] is True
    assert p3["present_after"] is False
    assert p3["mean_bias_mm_after"] is None
    assert p3["bias_significant_after"] is None
    assert p3["confidence_interval_mm_after"] is None
    assert p3["reading_count_before"] == 3
    assert p3["reading_count_after"] == 0
    kept = [t for t in comp["points"] if t["point_id"] != "P3"]
    assert all(t["dropped"] is False and t["present_after"] is True
               and t["reading_count_after"] == 3 for t in kept)
    # 副本结果只含 4 个点、12 条读数，回归仍可用
    assert len(j["result"]["points"]) == 4
    assert j["result"]["design"]["total_readings"] == 12
    assert j["result"]["design"]["excluded_readings"] == 3
    assert j["result"]["regression_available"] is True
    assert comp["regression_after"] is not None
    # 父研究保持 5 点不变
    g = client.get(f"/linearity-studies/{sid}").json()
    assert len(g["result"]["points"]) == 5


def test_copy_is_atomic_failure_writes_no_version(client):
    """复制任一步失败（未知读数 / 排除后点数不足）都不得写入新版本行。"""
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]

    def version_count():
        return len(client.get(
            f"/chains/{cid}/linearity-studies").json()["studies"])

    assert version_count() == 1
    r = client.post(f"/linearity-studies/{sid}/copy", json={
        "exclusions": [{"point_id": "P3", "operator": "ZZ",
                        "replicate": 1, "reason": "x"}]})
    assert r.status_code == 422
    assert version_count() == 1

    excl = []
    for pid in ("P1", "P2", "P3", "P4"):
        for op, r_ in (("A", 1), ("B", 2), ("A", 3)):
            excl.append({"point_id": pid, "operator": op,
                         "replicate": r_, "reason": "clear"})
    r = client.post(f"/linearity-studies/{sid}/copy",
                    json={"exclusions": excl})
    assert r.status_code == 422
    assert version_count() == 1  # 只剩 1 点：拟合不可用，无新版本落库


def test_compare_results_handles_dropped_point_directly():
    p1 = LinearityStudyCreate(**_study_payload())
    r1 = run_study(p1)
    excl = [
        {"point_id": "P3", "operator": "A", "replicate": 1, "reason": "x"},
        {"point_id": "P3", "operator": "B", "replicate": 2, "reason": "x"},
        {"point_id": "P3", "operator": "A", "replicate": 3, "reason": "x"},
    ]
    r2 = run_study(p1, exclusions=excl)
    comp = compare_results(r1, r2)
    assert len(comp["points"]) == 5
    p3 = next(t for t in comp["points"] if t["point_id"] == "P3")
    assert p3["dropped"] is True
    assert p3["mean_bias_mm_after"] is None
    assert p3["bias_significant_after"] is None
    assert p3["reading_count_after"] == 0
    # 不抛异常且其余点照常比较
    assert all(t["reading_count_after"] == 3
               for t in comp["points"] if not t["dropped"])


def test_finalized_study_not_copyable(client):
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    assert client.post(f"/linearity-studies/{sid}/finalize",
                       json={}).status_code == 200
    r = client.post(f"/linearity-studies/{sid}/copy", json={})
    assert r.status_code == 422


def test_adopt_requires_finalize_and_adoptability(client):
    cid = _make_chain(client)
    # 覆盖不足的草案：定稿后仍不能采用
    payload = _study_payload(xs=(49.9, 50.0, 50.1),
                             wl=40.0, wu=60.0, min_cov=0.5)
    sid = _create_study(client, cid, payload)["study_id"]
    r = client.post(f"/linearity-studies/{sid}/adopt", json={})
    assert r.status_code == 422  # 草案
    assert client.post(f"/linearity-studies/{sid}/finalize",
                       json={}).status_code == 200
    r = client.post(f"/linearity-studies/{sid}/adopt", json={})
    assert r.status_code == 422  # 覆盖过窄
    detail = r.json()["detail"]
    assert "覆盖" in detail


def test_snapshot_repeated_get_identical(client):
    cid = _make_chain(client)
    sid = _create_study(client, cid)["study_id"]
    g1 = client.get(f"/linearity-studies/{sid}").json()
    g2 = client.get(f"/linearity-studies/{sid}").json()
    assert g1["result"] == g2["result"]


def test_list_studies(client):
    cid = _make_chain(client)
    _create_study(client, cid)
    r = client.get(f"/chains/{cid}/linearity-studies")
    assert r.status_code == 200
    item = r.json()["studies"][0]
    assert item["reference_points"] == 5
    assert item["total_readings"] == 15
    assert item["regression_available"] is True


# ------------------------------------------------------------- 接入测量方案

def _plan_payload(linearity_studies=None, gauges=None):
    base = [
        {"dimension_id": "L1", "unit": "mm", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
         "repeatability_std": 0.003},
        {"dimension_id": "L2", "unit": "mm", "resolution": 0.005,
         "calibration_expanded_uncertainty": 0.003, "coverage_factor": 2.0,
         "bias_correction": 0.0, "bias_std_uncertainty": 0.0005,
         "repeatability_std": 0.002},
        {"dimension_id": "L3", "unit": "mm", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2.0,
         "bias_correction": -0.001, "bias_std_uncertainty": 0.0015,
         "repeatability_std": 0.004},
    ]
    if gauges:
        base = gauges
    return {"name": "plan-lin", "gauges": base,
            "linearity_studies": linearity_studies or {}}


def _batch(plan_id, rows, *, name="b", seed=42):
    return {
        "name": name, "measurement_plan_id": plan_id,
        "output_coverage_factor": 2.0,
        "guard_band": {"mode": "multiple", "multiple": 1.0},
        "measurement_mc_samples": 80000, "measurement_mc_seed": seed,
        "bootstrap_samples": 1000, "random_seed": 7,
        "rows": rows,
    }


def _rows(values):
    return [{"serial": s, "measurements": [
        {"dimension_id": d, "value": v, "unit": "mm"}
        for d, v in row.items()]} for s, row in values]


def test_plan_references_adopted_study_and_batch_uses_inverse_regression(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json=_plan_payload({"L1": sid}))
    assert r.status_code == 201, r.text
    plan = r.json()["plan"]
    pid = r.json()["plan_id"]
    l1 = next(d for d in plan["dimensions"] if d["dimension_id"] == "L1")
    assert l1["bias_correction_source"] == "linearity_study"
    assert l1["linear_bias_model"]["study_id"] == sid

    x_meas = 50.046
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json=_batch(pid, _rows([
                        ("W1", {"L1": x_meas, "L2": 30.302, "L3": 79.981})])))
    assert r.status_code == 201, r.text
    m = r.json()["measurement"]
    item = next(d for d in m["dimensions"]
                if d["dimension_id"] == "L1")["items"][0]
    model = l1["linear_bias_model"]
    a, b = model["intercept_mm"], model["slope"]
    assert item["corrected_mm"] == pytest.approx((x_meas - a) / (1 + b))
    assert item["in_applicable_range"] is True
    assert item["linear_bias_std_uncertainty_mm"] > 0
    # GUM 与 MC 的扩展不确定度都包含模型项（k=2）
    assert item["expanded_uncertainty_mm"]["gum"] > 2 * math.sqrt(
        0.0036055512754639895 ** 2)  # 旧 u_rest 仅校准+重复


def test_out_of_applicable_range_not_corrected(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    plan = client.post(f"/chains/{cid}/measurement-plans",
                       json=_plan_payload({"L1": sid})).json()["plan"]
    pid = client.get(f"/chains/{cid}/measurement-plans").json()["plans"][0]["id"]
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json=_batch(pid, _rows([
                        ("W1", {"L1": 30.0, "L2": 30.302, "L3": 79.981})])))
    m = r.json()["measurement"]
    item = next(d for d in m["dimensions"]
                if d["dimension_id"] == "L1")["items"][0]
    assert item["in_applicable_range"] is False
    assert item["corrected_mm"] == pytest.approx(30.0)  # 不外推
    assert item["linear_bias_std_uncertainty_mm"] == 0.0
    assert "no_correction_reason" in item
    cl = m["closure"]["items"][0]
    assert cl["linear_bias_variance_mm2"] == 0.0


def test_plan_rejects_non_adopted_and_wrong_dimension_study(client):
    cid = _make_chain(client)
    sid_draft = _create_study(client, cid)["study_id"]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json=_plan_payload({"L1": sid_draft}))
    assert r.status_code == 422
    sid = _full_lifecycle(client, cid)
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json=_plan_payload({"L2": sid}))
    assert r.status_code == 422  # 研究针对 L1


def test_plan_rejects_manual_bias_with_linearity_study(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    gauges = [
        {"dimension_id": "L1", "unit": "mm", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
         "bias_correction": 0.001, "bias_std_uncertainty": 0.0005,
         "repeatability_std": 0.003},
        {"dimension_id": "L2", "unit": "mm", "resolution": 0.005,
         "calibration_expanded_uncertainty": 0.003, "coverage_factor": 2.0,
         "repeatability_std": 0.002},
        {"dimension_id": "L3", "unit": "mm", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2.0,
         "repeatability_std": 0.004},
    ]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json=_plan_payload({"L1": sid}, gauges=gauges))
    assert r.status_code == 422


def test_closure_propagates_linearity_coefficient_and_reference_uncertainty(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    pid = client.post(f"/chains/{cid}/measurement-plans",
                      json=_plan_payload({"L1": sid})).json()["plan_id"]
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json=_batch(pid, _rows([
                        ("W1", {"L1": 50.046, "L2": 30.302, "L3": 79.981})])))
    m = r.json()["measurement"]
    cl = m["closure"]["items"][0]
    # L1 方向系数 +1，线性项方差 = u_lin(L1)²
    l1item = next(d for d in m["dimensions"]
                  if d["dimension_id"] == "L1")["items"][0]
    assert cl["linear_bias_variance_mm2"] == pytest.approx(
        l1item["linear_bias_std_uncertainty_mm"] ** 2)
    assert cl["combined_std_uncertainty_mm"]["gum"] > \
        m["closure"]["gum"]["variance_from_correlated_rest_mm2"] ** 0.5
    # GUM 与 MC 同量级
    assert cl["combined_std_uncertainty_mm"]["monte_carlo"] == pytest.approx(
        cl["combined_std_uncertainty_mm"]["gum"], rel=0.15)
    assert m["closure"]["gum"]["variance_from_linear_bias_mm2"] > 0


def test_batch_frozen_and_seed_reproducible(client):
    cid = _make_chain(client)
    sid = _full_lifecycle(client, cid)
    pid = client.post(f"/chains/{cid}/measurement-plans",
                      json=_plan_payload({"L1": sid})).json()["plan_id"]
    rows = _rows([("W1", {"L1": 50.046, "L2": 30.302, "L3": 79.981})])
    j1 = client.post(f"/chains/{cid}/inspection-batches",
                     json=_batch(pid, rows, name="b1")).json()
    j2 = client.post(f"/chains/{cid}/inspection-batches",
                     json=_batch(pid, rows, name="b2")).json()
    us1 = [it["expanded_uncertainty_mm"]["monte_carlo"]
           for d in j1["measurement"]["dimensions"] for it in d["items"]]
    us2 = [it["expanded_uncertainty_mm"]["monte_carlo"]
           for d in j2["measurement"]["dimensions"] for it in d["items"]]
    assert us1 == us2
    g1 = client.get(f"/inspection-batches/{j1['batch_id']}").json()
    g2 = client.get(f"/inspection-batches/{j1['batch_id']}").json()
    assert g1["measurement"] == g2["measurement"]


def test_historical_plans_and_batches_unchanged(client):
    """不引用线性研究的历史方案 / 批次口径完全不变。"""
    cid = _make_chain(client)
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "legacy",
        "gauges": [
            {"dimension_id": "L1", "unit": "mm", "resolution": 0.01,
             "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
             "bias_correction": 0.002, "bias_std_uncertainty": 0.001,
             "repeatability_std": 0.003}],
        "correlations": []})
    # 需要覆盖全部尺寸
    assert r.status_code == 422
    # 新研究采用后再建旧风格方案并出批，结果与无研究时一致
    _full_lifecycle(client, cid)
    legacy = {
        "name": "legacy-full",
        "gauges": [
            {"dimension_id": "L1", "unit": "mm", "resolution": 0.01,
             "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
             "bias_correction": 0.002, "bias_std_uncertainty": 0.001,
             "repeatability_std": 0.003},
            {"dimension_id": "L2", "unit": "mm", "resolution": 0.005,
             "calibration_expanded_uncertainty": 0.003, "coverage_factor": 2.0,
             "bias_correction": 0.0, "bias_std_uncertainty": 0.0005,
             "repeatability_std": 0.002},
            {"dimension_id": "L3", "unit": "mm", "resolution": 0.01,
             "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2.0,
             "bias_correction": -0.001, "bias_std_uncertainty": 0.0015,
             "repeatability_std": 0.004}],
    }
    r = client.post(f"/chains/{cid}/measurement-plans", json=legacy)
    assert r.status_code == 201
    pid = r.json()["plan_id"]
    m = client.post(f"/chains/{cid}/inspection-batches",
                    json=_batch(pid, _rows([
                        ("W1", {"L1": 50.046, "L2": 30.302,
                                "L3": 79.981})]))).json()["measurement"]
    l1 = next(d for d in m["dimensions"] if d["dimension_id"] == "L1")
    assert l1["bias_correction_source"] == "manual"
    assert l1["linear_bias_model"] is None
    item = l1["items"][0]
    assert item["corrected_mm"] == pytest.approx(50.048)
    assert m["closure"]["gum"]["variance_from_linear_bias_mm2"] == 0.0


# ------------------------------------------------------------- 模型直接评估

def test_apply_linear_bias_model_gradients_and_range():
    model = {
        "intercept_mm": -0.02, "slope": 0.001,
        "coefficient_covariance_mm2": [[4e-6, 1e-7], [1e-7, 9e-9]],
        "applicable_range_mm": [10.0, 50.0],
    }
    ev = apply_linear_bias_model(model, 30.0)
    assert ev["corrected_mm"] == pytest.approx((30.0 + 0.02) / 1.001)
    # 梯度手算
    cov = np.array(model["coefficient_covariance_mm2"])
    grad = np.array([-1 / 1.001, -(30.0 + 0.02) / 1.001 ** 2])
    assert ev["correction_std_uncertainty_mm"] == pytest.approx(
        math.sqrt(grad @ cov @ grad))
    out = apply_linear_bias_model(model, 5.0)
    assert out["correction_applied"] is False
    assert out["corrected_mm"] == 5.0


def test_bias_model_snapshot_requires_regression():
    class Row:
        id = 1
        name = "s"
        dimension_id = "L1"
        result_json = {
            "regression": {
                "intercept_mm": -0.02, "slope": 0.001,
                "coefficient_covariance": {
                    "gum_including_reference_mm2": [[1e-6, 0], [0, 1e-9]]},
                "applicable_range_mm": [10, 50],
                "applicable_range_submitted": [10, 50]},
            "coverage": {"span_coverage_ratio": 0.8},
            "process_tolerance": {"mm": 0.1}}
    snap = bias_model_snapshot(Row())
    assert snap["slope"] == 0.001
    assert snap["study_id"] == 1


def test_compare_results_detects_change():
    p1 = LinearityStudyCreate(**_study_payload())
    r1 = run_study(p1)
    # 排除 P3 一个读数
    r2 = run_study(p1, exclusions=[
        {"point_id": "P3", "operator": "B", "replicate": 2, "reason": "x"}])
    comp = compare_results(r1, r2)
    assert len(comp["points"]) == 5
    p3 = next(t for t in comp["points"] if t["point_id"] == "P3")
    assert p3["reading_count_after"] == 2
    assert isinstance(comp["regression_changed"], bool)
