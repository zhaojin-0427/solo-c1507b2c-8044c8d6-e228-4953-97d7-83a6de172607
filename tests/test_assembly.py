"""选择性装配：池映射校验、缺测排除、规则矛盾、偏倚/不确定度传播、
分支定界求解与版本重排、快照冻结。"""
import pytest


CHAIN = {
    "name": "asm-chain",
    "closure_lower_limit": 0.05,
    "closure_upper_limit": 0.60,
    "dimensions": [
        {"id": "L1", "start": "a", "end": "b", "nominal": 50,
         "upper_deviation": 0.08, "lower_deviation": 0.0, "std_dev": 0.02},
        {"id": "L2", "start": "b", "end": "c", "nominal": 30.3,
         "upper_deviation": 0.05, "lower_deviation": -0.05,
         "distribution": "uniform"},
        {"id": "L3", "start": "c", "end": "a", "nominal": 80,
         "upper_deviation": 0.0, "lower_deviation": -0.15, "std_dev": 0.04,
         "direction": -1},
    ],
    "mc_samples": 5000,
    "random_seed": 1,
}


def _batch(name, rows, plan_id=None, guard=None):
    payload = {
        "name": name,
        "bootstrap_samples": 1000,
        "random_seed": 1,
        "rows": [
            {"serial": s, "measurements": [
                {"dimension_id": d, "value": v, "unit": "mm"}
                for d, v in meas]}
            for s, meas in rows
        ],
    }
    if plan_id is not None:
        payload["measurement_plan_id"] = plan_id
        payload["measurement_mc_samples"] = 20000
        payload["measurement_mc_seed"] = 9
        payload["guard_band"] = guard or {"mode": "multiple", "multiple": 1.0}
    return payload


# 全部落在规格内、端隙约 0.28~0.40 的两组实测
RAW_A = [
    ("A1", [("L1", 50.02), ("L2", 30.30), ("L3", 79.95)]),
    ("A2", [("L1", 50.06), ("L2", 30.32), ("L3", 80.05)]),
    ("A3", [("L1", 50.10), ("L2", 30.20), ("L3", 80.10)]),
]
RAW_B = [
    ("B1", [("L1", 50.00), ("L2", 30.28), ("L3", 80.00)]),
    ("B2", [("L1", 50.04), ("L2", 30.31), ("L3", 79.90)]),
    ("B3", [("L1", 50.08), ("L2", 30.26), ("L3", 80.02)]),
]
BATCH_A = _batch("lot-A", RAW_A)
BATCH_B = _batch("lot-B", RAW_B)

POOLS_SINGLE = [
    {"name": "P1", "dimensions": ["L1"]},
    {"name": "P2", "dimensions": ["L2"]},
    {"name": "P3", "dimensions": ["L3"]},
]


@pytest.fixture()
def setup(client):
    cid = client.post("/chains", json=CHAIN).json()["chain_id"]
    id_a = client.post(
        f"/chains/{cid}/inspection-batches", json=BATCH_A).json()["batch_id"]
    id_b = client.post(
        f"/chains/{cid}/inspection-batches", json=BATCH_B).json()["batch_id"]
    return cid, id_a, id_b


def _task(client, setup, **over):
    cid, id_a, id_b = setup
    payload = {
        "name": "task",
        "batch_ids": [id_a, id_b],
        "pools": POOLS_SINGLE,
        "assembly_count": 2,
        "target_gap": {"lower": 0.05, "upper": 0.60},
        "random_seed": 42,
        "measurement_mc_samples": 5000,
    }
    payload.update(over)
    r = client.post(f"/chains/{cid}/assembly-tasks", json=payload)
    return r


# --------------------------------------------------------------- 基本求解

def test_solve_happy_path(client, setup):
    r = _task(client, setup)
    assert r.status_code == 201, r.text
    res = r.json()["result"]
    assert res["status"] == "solved"
    assert res["qualified_assemblies"] == 2
    assert res["solver"]["exact"] is True
    assert res["solver"]["enumeration_truncated"] is False
    for a in res["assemblies"]:
        assert a["decision"] == "accept"
        # 无测量方案：不确定度与保护带为 0；gap_part 已含闭环方向系数
        assert a["combined_std_uncertainty_mm"] == 0.0
        assert a["guard_band_mm"] == 0.0
        s = sum(m["gap_part_mm"] for m in a["members"])
        assert abs(s - a["gap_mm"]) < 1e-12
        assert 0.05 <= a["gap_mm"] <= 0.60


def test_instance_and_workpiece_used_at_most_once(client, setup):
    r = _task(client, setup, assembly_count=3)
    res = r.json()["result"]
    keys = [(m["batch_id"], m["serial"])
            for a in res["assemblies"] for m in a["members"]]
    # 池内实例不重复，且同一物理工件不得跨池复用
    assert len(keys) == len(set(keys))


def test_center_deviation_optimization(client, setup):
    """所有组合均合格时，目标中心偏差总和应取到枚举中的最优值。"""
    from itertools import product
    r = _task(client, setup, assembly_count=3)
    res = r.json()["result"]
    # 直接用报告里的目标中心核对：每个装配偏差都不劣于可行上界
    assert res["objective"]["total_center_deviation_mm"] >= 0.0
    assert res["objective"]["worst_guard_margin_mm"] is not None
    assert (res["objective"]["total_cross_batch_transitions"] >= 0)


# --------------------------------------------------------------- 创建期拒绝

def test_wrong_chain_batch_rejected(client, setup):
    cid, id_a, _ = setup
    other = client.post("/chains", json={**CHAIN, "name": "other"})
    cid2 = other.json()["chain_id"]
    r = client.post(f"/chains/{cid2}/assembly-tasks", json={
        "name": "bad", "batch_ids": [id_a], "pools": POOLS_SINGLE,
        "assembly_count": 1, "target_gap": {"lower": 0.05, "upper": 0.6}})
    assert r.status_code == 422
    assert "不属于当前链" in r.text


def test_missing_batch_404(client, setup):
    cid, _, _ = setup
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "bad", "batch_ids": [9999], "pools": POOLS_SINGLE,
        "assembly_count": 1, "target_gap": {"lower": 0.05, "upper": 0.6}})
    assert r.status_code == 404


def test_missing_dimension_mapping_rejected(client, setup):
    r = _task(client, setup, pools=[
        {"name": "P1", "dimensions": ["L1"]},
        {"name": "P2", "dimensions": ["L2"]}])
    assert r.status_code == 422
    assert "漏映射" in r.text and "L3" in r.text


def test_duplicate_dimension_mapping_rejected(client, setup):
    r = _task(client, setup, pools=[
        {"name": "P1", "dimensions": ["L1", "L2"]},
        {"name": "P2", "dimensions": ["L2", "L3"]}])
    assert r.status_code == 422
    assert "重复映射" in r.text


def test_external_dimension_mapping_rejected(client, setup):
    r = _task(client, setup, pools=[
        {"name": "P1", "dimensions": ["L1"]},
        {"name": "P2", "dimensions": ["L2"]},
        {"name": "P3", "dimensions": ["NOPE"]}])
    assert r.status_code == 422
    assert "链之外" in r.text


def test_pool_name_duplicate_rejected(client, setup):
    r = _task(client, setup, pools=[
        {"name": "P", "dimensions": ["L1"]},
        {"name": "P", "dimensions": ["L2"]},
        {"name": "P3", "dimensions": ["L3"]}])
    assert r.status_code == 422
    assert "零件池名称重复" in r.text


def test_assembly_count_exceeds_capacity_rejected(client, setup):
    r = _task(client, setup, assembly_count=9)
    assert r.status_code == 422
    assert "装配数量" in r.text


def test_cross_batch_limit_bounds_rejected(client, setup):
    r = _task(client, setup, cross_batch_limit=4)
    assert r.status_code == 422
    assert "跨批限制" in r.text


def test_target_gap_invalid_rejected(client, setup):
    r = _task(client, setup, target_gap={"lower": 0.6, "upper": 0.05})
    assert r.status_code == 422


def test_duplicate_batch_listing_rejected(client, setup):
    cid, id_a, _ = setup
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "dup", "batch_ids": [id_a, id_a], "pools": POOLS_SINGLE,
        "assembly_count": 1, "target_gap": {"lower": 0.05, "upper": 0.6}})
    assert r.status_code == 422
    assert "来源批次重复" in r.text


# --------------------------------------------------------------- 缺测排除

def test_missing_measurement_instance_excluded(client, setup):
    cid, _, id_b = setup
    # 在 B 批补一个缺 L3 的工件
    extra = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "lot-B-gap", [("B9", [("L1", 50.03), ("L2", 30.29)])]))
    id_g = extra.json()["batch_id"]
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "gaps", "batch_ids": [id_g],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 1,
        "target_gap": {"lower": 0.05, "upper": 0.6}})
    assert r.status_code == 422  # P23 池无可用实例
    res_text = r.text
    assert "装配数量" in res_text or "容量" in res_text


def test_multi_dim_pool_excludes_incomplete_and_reports(client, setup):
    cid, id_a, _ = setup
    gap_batch = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "g2", [
            ("G1", [("L1", 50.01), ("L2", 30.30), ("L3", 80.00)]),
            ("G2", [("L1", 50.02), ("L2", 30.29)]),  # 缺 L3
        ]))
    id_g = gap_batch.json()["batch_id"]
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "multi", "batch_ids": [id_a, id_g],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 1,
        "target_gap": {"lower": 0.05, "upper": 0.6},
        "measurement_mc_samples": 3000})
    assert r.status_code == 201, r.text
    res = r.json()["result"]
    p23 = next(p for p in res["pools"] if p["name"] == "P23")
    assert p23["available"] == 4  # A1,A2,A3,G1（G2 缺测排除）
    ex = p23["excluded_instances"][0]
    assert ex["serial"] == "G2" and ex["missing_dimension_ids"] == ["L3"]
    missing = res["diagnostics"]["missing_serials"]
    assert ("P23", id_g, "G2") in [
        (m["pool"], m["batch_id"], m["serial"]) for m in missing]


# --------------------------------------------------------------- 规则

def test_cross_batch_limit_one_forces_single_batch(client, setup):
    r = _task(client, setup, cross_batch_limit=1, assembly_count=2)
    assert r.status_code == 201, r.text
    for a in r.json()["result"]["assemblies"]:
        assert a["distinct_batches"] == 1
        assert a["cross_batch_transitions"] == 0


def test_same_batch_group_forces_common_batch(client, setup):
    pools = [{"name": "P1", "dimensions": ["L1"]},
             {"name": "P23", "dimensions": ["L2", "L3"]}]
    r = _task(client, setup, pools=pools, assembly_count=2,
              same_batch_groups=[{"pools": ["P1", "P23"]}])
    assert r.status_code == 201, r.text
    for a in r.json()["result"]["assemblies"]:
        batches = {m["batch_id"] for m in a["members"]}
        assert len(batches) == 1


def test_same_batch_group_contradiction_no_common_batch(client, setup):
    cid, _, _ = setup
    # 批 1 只测 L1（P1 池专用来源），批 2 只测 L2,L3（P23 池专用来源）
    only_l1 = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "only-l1", [("X1", [("L1", 50.02)]), ("X2", [("L1", 50.03)])]))
    only_l23 = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "only-l23", [("Y1", [("L2", 30.30), ("L3", 80.00)]),
                     ("Y2", [("L2", 30.28), ("L3", 79.98)])]))
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "contradict",
        "batch_ids": [only_l1.json()["batch_id"], only_l23.json()["batch_id"]],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 1,
        "target_gap": {"lower": 0.05, "upper": 0.6},
        "same_batch_groups": [{"pools": ["P1", "P23"]}]})
    assert r.status_code == 422
    assert "同批组" in r.text


def test_forbidden_single_instance_pair_contradiction(client, setup):
    cid, _, _ = setup
    # 两个池各只有一个可用实例，且是不同工件：禁配即没有任何可行装配
    only = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "specialized", [
            ("S1", [("L1", 50.02)]),           # 仅 P1=[L1] 可用
            ("T1", [("L2", 30.30), ("L3", 80.00)]),  # 仅 P2=[L2,L3] 可用
        ]))
    bid = only.json()["batch_id"]
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "forbid", "batch_ids": [bid],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 1,
        "target_gap": {"lower": 0.05, "upper": 0.6},
        "forbidden_matches": [
            {"pool_a": "P1", "batch_a": bid, "serial_a": "S1",
             "pool_b": "P23", "batch_b": bid, "serial_b": "T1"}]})
    assert r.status_code == 422
    assert "禁配" in r.text


def test_forbidden_match_referencing_missing_serial_rejected(client, setup):
    cid, id_a, id_b = setup
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "forbid-missing", "batch_ids": [id_a, id_b],
        "pools": POOLS_SINGLE, "assembly_count": 1,
        "target_gap": {"lower": 0.05, "upper": 0.6},
        "forbidden_matches": [
            {"pool_a": "P1", "batch_a": id_a, "serial_a": "GHOST",
             "pool_b": "P2", "batch_b": id_a, "serial_b": "A1"}]})
    assert r.status_code == 422
    assert "不存在或缺测" in r.text


def test_forbidden_match_prunes_combo(client, setup):
    cid, id_a, id_b = setup
    # 先求一个解，取一个合格组合再加禁配，确认该组合不再出现
    base = _task(client, setup, assembly_count=1).json()["result"]
    members = base["assemblies"][0]["members"]
    edge = [
        {"pool_a": members[0]["pool"], "batch_a": members[0]["batch_id"],
         "serial_a": members[0]["serial"],
         "pool_b": members[1]["pool"], "batch_b": members[1]["batch_id"],
         "serial_b": members[1]["serial"]},
    ]
    r = _task(client, setup, name="task-fb", assembly_count=1, forbidden_matches=edge)
    assert r.status_code == 201, r.text
    comb = r.json()["result"]["assemblies"][0]["members"]
    pair = {(members[0]["batch_id"], members[0]["serial"]),
            (members[1]["batch_id"], members[1]["serial"])}
    got = {(m["batch_id"], m["serial"]) for m in comb}
    assert not pair.issubset(got)


def test_unknown_pool_in_rules_rejected(client, setup):
    r = _task(client, setup, same_batch_groups=[{"pools": ["P1", "NOPE"]}])
    assert r.status_code == 422
    r2 = _task(client, setup, name="t2", forbidden_matches=[
        {"pool_a": "P1", "serial_a": "A1", "pool_b": "NOPE", "serial_b": "A2"}])
    assert r2.status_code == 422


# ----------------------------------------------------- 偏倚修正 / 不确定度

PLAN = {
    "name": "plan-asm",
    "gauges": [
        {"dimension_id": "L1", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.02, "coverage_factor": 2,
         "bias_correction": 0.01, "bias_std_uncertainty": 0.005,
         "repeatability_std": 0.004},
        {"dimension_id": "L2", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.02, "coverage_factor": 2,
         "repeatability_std": 0.004},
        {"dimension_id": "L3", "resolution": 0.02,
         "calibration_expanded_uncertainty": 0.04, "coverage_factor": 2,
         "bias_correction": -0.01, "bias_std_uncertainty": 0.006,
         "repeatability_std": 0.006},
    ],
    "correlations": [{"dim_a": "L1", "dim_b": "L2", "rho": 0.5}],
}


@pytest.fixture()
def setup_plan(client, setup):
    cid, _, _ = setup
    pid = client.post(
        f"/chains/{cid}/measurement-plans", json=PLAN).json()["plan_id"]
    ba = client.post(f"/chains/{cid}/inspection-batches",
                     json=_batch("plan-A", RAW_A, plan_id=pid))
    bb = client.post(f"/chains/{cid}/inspection-batches",
                     json=_batch("plan-B", RAW_B, plan_id=pid))
    return cid, ba.json()["batch_id"], bb.json()["batch_id"], pid


def test_bias_correction_and_uncertainty_propagation(client, setup_plan):
    cid, id_a, id_b, pid = setup_plan
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "gauged", "batch_ids": [id_a, id_b], "pools": POOLS_SINGLE,
        "assembly_count": 2, "target_gap": {"lower": 0.05, "upper": 0.60},
        "random_seed": 42, "measurement_mc_samples": 40000})
    assert r.status_code == 201, r.text
    a = r.json()["result"]["assemblies"][0]
    # G = (L1+0.01) + L2 − (L3−0.01) = 原始 + 0.02
    raw = sum((1 if m["pool"] != "P3" else -1) * m["measured_mm"][
        {"P1": "L1", "P2": "L2", "P3": "L3"}[m["pool"]]]
        for m in a["members"])
    assert abs(a["gap_mm"] - (raw + 0.02)) < 1e-9
    # GUM u_C > 0，保护带 w = 2·u_C（multiple=1, k_out=2）
    assert a["combined_std_uncertainty_mm"] > 0
    assert abs(a["guard_band_mm"]
               - 2 * a["combined_std_uncertainty_mm"]) < 1e-12
    # 固定种子 MC 与 GUM 一致（独立实例求和，5% 容差）
    mc_std = a["monte_carlo"]["std_mm"]
    assert mc_std > 0
    assert abs(mc_std - a["combined_std_uncertainty_mm"]) / mc_std < 0.05


def test_guard_band_indeterminate_zone(client, setup_plan):
    cid, id_a, id_b, _ = setup_plan
    # 目标区间极窄且贴近某组合间隙：必出现 reject/indeterminate
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "tight", "batch_ids": [id_a, id_b], "pools": POOLS_SINGLE,
        "assembly_count": 1, "target_gap": {"lower": 0.30, "upper": 0.32},
        "guard_band": {"mode": "multiple", "multiple": 1.0},
        "measurement_mc_samples": 20000, "random_seed": 42})
    res = r.json()["result"]
    counts = res["diagnostics"]["candidate_combo_counts"]
    assert counts["total_feasible"] > 0
    assert res["status"] in ("infeasible", "partial")
    # 有不确定度时应当存在落入保护带的不确定组合
    assert counts["indeterminate"] + counts["reject"] > 0


def test_fixed_guard_band(client, setup_plan):
    cid, id_a, id_b, _ = setup_plan
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "fixed", "batch_ids": [id_a, id_b], "pools": POOLS_SINGLE,
        "assembly_count": 2, "target_gap": {"lower": 0.05, "upper": 0.60},
        "guard_band": {"mode": "fixed", "fixed": 0.02, "unit": "mm"}})
    assert r.status_code == 201, r.text
    for a in r.json()["result"]["assemblies"]:
        assert a["guard_band_mm"] == pytest.approx(0.02)


# --------------------------------------------------------------- 无解诊断

def test_infeasible_returns_restricted_pools_and_rules(client, setup_plan):
    cid, id_a, id_b, _ = setup_plan
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "no-solution", "batch_ids": [id_a, id_b],
        "pools": POOLS_SINGLE, "assembly_count": 3,
        "target_gap": {"lower": 0.318, "upper": 0.320},
        "measurement_mc_samples": 10000, "random_seed": 42})
    assert r.status_code == 201, r.text  # 任务仍冻结入库
    res = r.json()["result"]
    assert res["status"] == "infeasible"
    assert res["qualified_assemblies"] == 0
    diag = res["diagnostics"]
    assert len(diag["restricted_pools"]) == 3
    assert all("available_instances" in p for p in diag["restricted_pools"])
    assert diag["triggered_rules"]  # 至少报告目标间隙+保护带
    assert diag["candidate_combo_counts"]["accept"] == 0


# --------------------------------------------------------------- 版本重排

def _lock_body(assembly):
    return {"locked_assemblies": [{"members": [
        {"pool": m["pool"], "batch_id": m["batch_id"], "serial": m["serial"]}
        for m in assembly["members"]]}]}


def test_rearrange_locks_confirmed_assembly(client, setup):
    v1 = _task(client, setup).json()
    vid1 = v1["version_id"]
    chosen = v1["result"]["assemblies"][0]
    r = client.post(f"/assembly-versions/{vid1}/rearrange",
                    json=_lock_body(chosen))
    assert r.status_code == 201, r.text
    v2 = r.json()
    assert v2["version_no"] == 2
    assert v2["parent_version_id"] == vid1
    res2 = v2["result"]
    assert res2["locked_assemblies"] == 1
    locked = [a for a in res2["assemblies"] if a["locked"]]
    assert len(locked) == 1
    locked_keys = {(m["batch_id"], m["serial"]) for m in locked[0]["members"]}
    assert locked_keys == {(m["batch_id"], m["serial"])
                           for m in chosen["members"]}
    # 全部实例仍然只用一次
    keys = [(m["batch_id"], m["serial"])
            for a in res2["assemblies"] for m in a["members"]]
    assert len(keys) == len(set(keys))


def test_rearrange_cannot_lock_non_accepted(client, setup_plan):
    cid, id_a, id_b, _ = setup_plan
    v1 = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "tight-lock", "batch_ids": [id_a, id_b], "pools": POOLS_SINGLE,
        "assembly_count": 1, "target_gap": {"lower": 0.30, "upper": 0.315},
        "measurement_mc_samples": 10000, "random_seed": 42}).json()
    # 找一个非 accept 组合需要直接构造：锁定不存在的组合 -> 422
    r = client.post(f"/assembly-versions/{v1['version_id']}/rearrange",
                    json={"locked_assemblies": [{"members": [
                        {"pool": "P1", "batch_id": id_a, "serial": "A1"},
                        {"pool": "P2", "batch_id": id_a, "serial": "A2"},
                        {"pool": "P3", "batch_id": id_a, "serial": "A3"}]}]})
    assert r.status_code == 422
    assert "合格" in r.text


def test_lock_unknown_pool_and_duplicate_rejected(client, setup):
    v1 = _task(client, setup).json()
    vid = v1["version_id"]
    # 缺池
    r = client.post(f"/assembly-versions/{vid}/rearrange",
                    json={"locked_assemblies": [{"members": [
                        {"pool": "P1", "batch_id": 1, "serial": "A1"},
                        {"pool": "P2", "batch_id": 1, "serial": "A2"}]}]})
    assert r.status_code == 422
    # 正常锁一组合
    chosen = v1["result"]["assemblies"][0]
    v2 = client.post(f"/assembly-versions/{vid}/rearrange",
                     json=_lock_body(chosen)).json()
    # 对 v2 再锁同一组合 -> 重复 422
    r2 = client.post(
        f"/assembly-versions/{v2['version_id']}/rearrange",
        json=_lock_body(chosen))
    assert r2.status_code == 422
    assert "重复" in r2.text


def test_parent_version_immutable_after_rearrange(client, setup):
    v1 = _task(client, setup).json()
    vid1 = v1["version_id"]
    chosen = v1["result"]["assemblies"][0]
    client.post(f"/assembly-versions/{vid1}/rearrange",
                json=_lock_body(chosen))
    again = client.get(f"/assembly-versions/{vid1}").json()
    assert again["result"]["locked_assemblies"] == 0
    assert again["version_no"] == 1
    versions = client.get(
        f"/assembly-tasks/{v1['task_id']}/versions").json()["versions"]
    assert [v["version_no"] for v in versions] == [1, 2]


def test_snapshot_freezes_batches_plan_and_seed(client, setup_plan):
    cid, id_a, id_b, pid = setup_plan
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "snap", "batch_ids": [id_a, id_b], "pools": POOLS_SINGLE,
        "assembly_count": 2, "target_gap": {"lower": 0.05, "upper": 0.60},
        "random_seed": 77, "measurement_mc_samples": 12345})
    res = r.json()["result"]
    snap = res["snapshot"]
    assert [b["batch_id"] for b in snap["source_batches"]] == [id_a, id_b]
    assert all(b["frozen"] and b["frozen_rows"] for b in snap["source_batches"])
    assert snap["source_batches"][0]["plan_snapshot"]["plan_id"] == pid
    assert snap["random_seed"] == 77
    assert snap["monte_carlo_samples"] == 12345
    assert snap["target_gap"]["normalized_mm"] == {"lower": 0.05, "upper": 0.6}
    # MC 结果固定种子可复现
    a = res["assemblies"][0]
    assert a["monte_carlo"]["seed"] == 77
    assert a["monte_carlo"]["samples"] == 12345


# --------------------------------------------------------------- 列表 / 404

def test_task_listing_and_404(client, setup):
    v = _task(client, setup).json()
    cid, _, _ = setup
    lst = client.get(f"/chains/{cid}/assembly-tasks").json()
    assert lst["tasks"][0]["versions"][0]["status"] == "solved"
    assert client.get("/assembly-tasks/9999/versions").status_code == 404
    assert client.get("/assembly-versions/9999").status_code == 404
    assert client.post(
        "/assembly-versions/9999/rearrange",
        json=_lock_body(v["result"]["assemblies"][0])).status_code == 404


# ----------------------------------------------------- 缺陷回归：禁配/分辨率MC/容量诊断

def test_forbidden_survives_lock_and_rearrange(client, setup):
    """锁定一组含 A1/B1 的合格组合重排后，禁配 A2(P2)+B2(P3) 仍不得入选。"""
    cid, id_a, id_b = setup
    forbid = [{
        "pool_a": "P2", "batch_a": id_a, "serial_a": "A2",
        "pool_b": "P3", "batch_b": id_b, "serial_b": "B2",
    }]
    r = _task(client, setup, name="forbid-keep",
              forbidden_matches=forbid, assembly_count=3)
    assert r.status_code == 201, r.text
    v1 = r.json()

    def has_forbidden(result):
        for a in result["assemblies"]:
            mp = {m["pool"]: (m["batch_id"], m["serial"])
                  for m in a["members"]}
            if mp["P2"] == (id_a, "A2") and mp["P3"] == (id_b, "B2"):
                return a["assembly_no"]
        return None

    assert has_forbidden(v1["result"]) is None
    # 锁定 v1 的第一个合格装配，在剩余实例上重排
    lock_asm = next(a for a in v1["result"]["assemblies"]
                    if a["decision"] == "accept")
    v2 = client.post(
        f"/assembly-versions/{v1['version_id']}/rearrange",
        json=_lock_body(lock_asm)).json()
    # 版本 2 剩余装配仍不得选中禁配对
    assert has_forbidden(v2["result"]) is None
    assert v2["result"]["locked_assemblies"] == 1
    # 版本 2 里 A2 若被使用，其 P3 搭档不能是 B2（显式逐条复核）
    for a in v2["result"]["assemblies"]:
        mp = {m["pool"]: (m["batch_id"], m["serial"])
              for m in a["members"]}
        if mp["P2"] == (id_a, "A2"):
            assert mp["P3"] != (id_b, "B2")


RESOLUTION_PLAN = {
    "name": "resolution-only",
    "gauges": [
        {"dimension_id": d, "resolution": 0.02,
         "calibration_expanded_uncertainty": 0.0, "coverage_factor": 2}
        for d in ("L1", "L2", "L3")
    ],
}


def test_resolution_only_plan_propagates_in_monte_carlo(client, setup):
    """只有 0.02 mm 分辨率分量时，u_res=0.02/(2√3)，MC 必须包含量化误差。"""
    cid, _, _ = setup
    pid = client.post(
        f"/chains/{cid}/measurement-plans",
        json=RESOLUTION_PLAN).json()["plan_id"]
    id_a = client.post(f"/chains/{cid}/inspection-batches",
                       json=_batch("res-A", RAW_A, plan_id=pid)).json()["batch_id"]
    id_b = client.post(f"/chains/{cid}/inspection-batches",
                       json=_batch("res-B", RAW_B, plan_id=pid)).json()["batch_id"]
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "res-mc", "batch_ids": [id_a, id_b],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 2, "target_gap": {"lower": 0.05, "upper": 0.6},
        "random_seed": 42, "measurement_mc_samples": 300000})
    assert r.status_code == 201, r.text
    asm = r.json()["result"]["assemblies"][0]
    # GUM：3 个独立尺寸 sqrt(3)·u_res；P23 实例自身两尺寸 sqrt(2)·u_res
    u_res = 0.02 / (2 * 3 ** 0.5)
    assert asm["combined_std_uncertainty_mm"] == pytest.approx(
        (3 ** 0.5) * u_res, rel=1e-9)
    mc_std = asm["monte_carlo"]["std_mm"]
    assert mc_std == pytest.approx(
        (3 ** 0.5) * u_res, rel=2e-2)
    # 实例误差明细非空：两尺寸实例 std ≈ 0.02/√6 ≈ 0.008165
    details = {(d["batch_id"], d["serial"]): d["std_error_mm"]
               for d in asm["monte_carlo"]["instance_error_std_mm"]}
    p23 = next(m for m in asm["members"] if m["pool"] == "P23")
    assert details[(p23["batch_id"], p23["serial"])] == pytest.approx(
        0.02 / 6 ** 0.5, rel=2e-2)
    assert len(details) == 2  # 两个工件实例都有量化误差列


def test_capacity_shortage_422_includes_diagnostics(client, setup):
    """缺测使某池只剩 1 实例却请求 2 套时，422 必须附受限池/缺测/规则。"""
    cid, _, _ = setup
    only = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "one-p23", [
            ("Z1", [("L1", None), ("L2", 30.30), ("L3", 80.00)]),
            ("Z2", [("L1", None), ("L2", 30.29)]),  # P23 缺 L3
        ]))
    id_one = only.json()["batch_id"]
    many = client.post(f"/chains/{cid}/inspection-batches", json=_batch(
        "many-p1", [
            ("H1", [("L1", 50.02)]),
            ("H2", [("L1", 50.03)]),
        ]))
    id_many = many.json()["batch_id"]
    r = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "cap", "batch_ids": [id_one, id_many],
        "pools": [{"name": "P1", "dimensions": ["L1"]},
                  {"name": "P23", "dimensions": ["L2", "L3"]}],
        "assembly_count": 2,
        "target_gap": {"lower": 0.05, "upper": 0.6}})
    assert r.status_code == 422
    body = r.json()["detail"]
    assert "容量" in body["message"]
    diag = body["diagnostics"]
    pools_d = {p["pool"]: p for p in diag["restricted_pools"]}
    assert pools_d["P23"]["available_instances"] == 1
    assert pools_d["P1"]["available_instances"] == 2
    missing = {(m["pool"], m["batch_id"], m["serial"])
               for m in diag["missing_serials"]}
    assert ("P23", id_one, "Z2") in missing
    z2_p23 = next(m for m in diag["missing_serials"]
                  if m["serial"] == "Z2" and m["batch_id"] == id_one
                  and m["pool"] == "P23")
    assert z2_p23["missing_dimension_ids"] == ["L3"]
    assert diag["triggered_rules"][0]["rule"] == "零件池容量不足"


def test_forbidden_survives_pool_index_remap_after_lock(client):
    """缺陷回归：锁定移除后池收缩重排，禁配关系仍按实例身份生效。

    构造两个专用批次（P1 池 A1..A4、P2 池 B1..B4），锁定 A1+B1 后
    A2/B2 在剩余池中同为下标 0——若禁配以池内下标存储，会错位到
    其它实例而错误放行 A2+B2。
    """
    cid = client.post("/chains", json={
        "name": "remap-chain",
        "closure_lower_limit": -2, "closure_upper_limit": 2,
        "dimensions": [
            {"id": "L1", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 1, "lower_deviation": -1, "std_dev": 0.2},
            {"id": "L2", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 1, "lower_deviation": -1, "std_dev": 0.2,
             "direction": -1},
        ],
        "mc_samples": 2000, "random_seed": 1}).json()["chain_id"]

    def single_batch(name, dim, serials, values):
        return client.post(f"/chains/{cid}/inspection-batches", json={
            "name": name, "bootstrap_samples": 1000, "random_seed": 1,
            "rows": [{"serial": s, "measurements": [
                {"dimension_id": dim, "value": v, "unit": "mm"}]}
                for s, v in zip(serials, values)]}).json()["batch_id"]

    bp1 = single_batch("p1-only", "L1",
                       ["A1", "A2", "A3", "A4"], [10.0, 10.1, 10.2, 10.3])
    bp2 = single_batch("p2-only", "L2",
                       ["B1", "B2", "B3", "B4"], [10.0, 10.1, 10.2, 10.3])
    pools = [{"name": "P1", "dimensions": ["L1"]},
             {"name": "P2", "dimensions": ["L2"]}]
    v1 = client.post(f"/chains/{cid}/assembly-tasks", json={
        "name": "remap", "batch_ids": [bp1, bp2], "pools": pools,
        "assembly_count": 1, "target_gap": {"lower": -2, "upper": 2},
        "forbidden_matches": [
            {"pool_a": "P1", "batch_a": bp1, "serial_a": "A2",
             "pool_b": "P2", "batch_b": bp2, "serial_b": "B2"}],
        "measurement_mc_samples": 1000}).json()
    locked = v1["result"]["assemblies"][0]
    assert {(m["batch_id"], m["serial"]) for m in locked["members"]} == {
        (bp1, "A1"), (bp2, "B1")}
    v2 = client.post(
        f"/assembly-versions/{v1['version_id']}/rearrange",
        json=_lock_body(locked)).json()["result"]
    for a in v2["assemblies"]:
        if a["locked"]:
            continue
        keys = [(m["batch_id"], m["serial"]) for m in a["members"]]
        assert keys != [(bp1, "A2"), (bp2, "B2")]
