"""API 端到端：持久化、方案分支不覆盖基线、批量调整、成本搜索、追溯。"""
import pytest


def _create(client, payload):
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_persist(client, simple_chain_payload):
    j = _create(client, simple_chain_payload)
    cid = j["chain_id"]
    got = client.get(f"/chains/{cid}").json()
    assert got["name"] == "test-chain"
    assert got["random_seed"] == 7
    methods = got["result"]["results"]
    assert set(methods) == {"worst_case", "rss", "monte_carlo"}
    # 单边公差原始表示保留
    l1 = got["submitted_input"]["dimensions"][0]
    assert l1["lower_deviation"] == 0.0


def test_404(client):
    assert client.get("/chains/9999").status_code == 404


def test_scenario_does_not_overwrite_baseline(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    before = client.get(f"/chains/{cid}").json()["result"]
    r = client.post(f"/chains/{cid}/scenarios", json={
        "name": "tight",
        "tolerance_overrides": {"L1": {"upper_deviation": 0.03,
                                       "lower_deviation": 0.0}},
        "std_dev_overrides": {"L3": 0.02},
    })
    assert r.status_code == 201, r.text
    sj = r.json()
    assert sj["baseline_preserved"] is True
    # L1 收紧后 RSS 超差率下降
    cmp_ = sj["comparison"]["rss"]
    assert (cmp_["scenario"]["reject_probability"]
            <= cmp_["baseline"]["reject_probability"])
    # 基线结果未改变
    after = client.get(f"/chains/{cid}").json()["result"]
    assert (after["results"]["rss"]["reject_probability"]
            == before["results"]["rss"]["reject_probability"])
    # 方案列表 / 详情
    lst = client.get(f"/chains/{cid}/scenarios").json()["scenarios"]
    assert len(lst) == 1 and lst[0]["name"] == "tight"
    detail = client.get(f"/scenarios/{sj['scenario_id']}").json()
    assert detail["kind"] == "branch"


def test_scenario_unknown_dimension_rejected(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/scenarios", json={
        "name": "bad",
        "tolerance_overrides": {"NOPE": {"upper_deviation": 0.01,
                                         "lower_deviation": 0.0}},
    })
    assert r.status_code == 422


def test_scenario_negative_std_dev_rejected_before_save(
        client, simple_chain_payload):
    """缺陷回归：方案提交负 σ 必须在保存前 422，不能落库后显示正 σ。"""
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/scenarios", json={
        "name": "negative-sigma",
        "std_dev_overrides": {"L2": -0.01},
    })
    assert r.status_code == 422
    assert "标准差" in r.text
    # 没有产生任何方案分支
    scs = client.get(f"/chains/{cid}/scenarios").json()["scenarios"]
    assert all(s["name"] != "negative-sigma" for s in scs)
    # 零仍合法（零方差固定尺寸）
    r0 = client.post(f"/chains/{cid}/scenarios", json={
        "name": "zero-sigma", "std_dev_overrides": {"L2": 0.0}})
    assert r0.status_code == 201, r0.text
    assert r0.json()["overrides"]["L2"]["std_dev"]["std_dev"] == 0.0


def test_batch_adjust(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/batch-adjust", json={
        "name": "70%", "tolerance_scale": 0.7, "std_dev_scale": 0.7,
    })
    assert r.status_code == 201, r.text
    j = r.json()
    assert {d["dimension_id"] for d in j["adjustment"]["dimensions"]} == {
        "L1", "L2", "L3"}
    cmp_ = j["comparison"]["rss"]
    assert cmp_["delta"]["sigma_mm"] < 0
    # 均匀分布 σ 随公差带理论重算的策略
    policies = {d["dimension_id"]: d["sigma_policy"]
                for d in j["adjustment"]["dimensions"]}
    assert policies["L2"] == "scaled_explicit"


def test_batch_target_subset(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/batch-adjust", json={
        "name": "only-L1", "target_dimensions": ["L1"],
        "tolerance_scale": 0.5,
    })
    j = r.json()
    assert [d["dimension_id"] for d in j["adjustment"]["dimensions"]] == ["L1"]
    policies = {d["dimension_id"]: d["sigma_policy"]
                for d in j["adjustment"]["dimensions"]}
    assert policies["L1"] == "kept_user_value"


def test_gap_probability_endpoint(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/gap-probability", json={
        "lower": 0.25, "upper": 0.58})
    assert r.status_code == 200
    p = r.json()["probabilities"]
    assert abs(p["rss"] - p["monte_carlo"]) < 0.01
    assert p["worst_case"] == 1.0
    # 比 WC 界更窄的区间，WC 记 0
    r2 = client.post(f"/chains/{cid}/gap-probability", json={
        "lower": 0.3, "upper": 0.5})
    assert r2.json()["probabilities"]["worst_case"] == 0.0


def test_gap_probability_in_inches(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    # 0.25mm ≈ 0.0098425 in ; 0.58mm ≈ 0.0228346 in
    r = client.post(f"/chains/{cid}/gap-probability", json={
        "lower": 0.25 / 25.4, "upper": 0.58 / 25.4, "unit": "in"})
    p_mm = client.post(f"/chains/{cid}/gap-probability",
                       json={"lower": 0.25, "upper": 0.58}).json()
    p_in = r.json()
    assert abs(p_in["probabilities"]["rss"]
               - p_mm["probabilities"]["rss"]) < 1e-9


def test_explicit_std_dev_uniform_moves_mc_like_rss(client, simple_chain_payload):
    """缺陷回归：只调低均匀尺寸的显式 σ，RSS 与 MC σ 必须同步减小。"""
    cid = _create(client, simple_chain_payload)["chain_id"]
    base = client.get(f"/chains/{cid}").json()["result"]["results"]
    r = client.post(f"/chains/{cid}/scenarios", json={
        "name": "tighter-spacer-sigma",
        "std_dev_overrides": {"L2": 0.01},   # 原理论 σ = 0.05/√3 ≈ 0.0289
    })
    assert r.status_code == 201, r.text
    sc = r.json()["result"]["results"]
    assert sc["rss"]["sigma_mm"] < base["rss"]["sigma_mm"]
    assert sc["monte_carlo"]["sigma_mm"] < base["monte_carlo"]["sigma_mm"]
    # MC 与 RSS 对同一显式 σ 必须口径一致
    assert abs(sc["monte_carlo"]["sigma_mm"] - sc["rss"]["sigma_mm"]) / \
        sc["rss"]["sigma_mm"] < 0.02
    # 方案上下文中的区间概率重放同样使用显式 σ
    gp = client.post(
        f"/chains/{cid}/gap-probability?scenario_id={r.json()['scenario_id']}",
        json={"lower": 0.25, "upper": 0.58}).json()
    assert gp["probabilities"]["rss"] == gp["probabilities"]["monte_carlo"] or \
        abs(gp["probabilities"]["rss"] - gp["probabilities"]["monte_carlo"]) < 0.01


def _zero_variance_payload():
    return {
        "name": "fixed-gap",
        "closure_lower_limit": 0.1,
        "closure_upper_limit": 0.5,
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10.2,
             "upper_deviation": 0.0, "lower_deviation": 0.0,
             "std_dev": 0.0, "distribution": "normal"},
            {"id": "B", "start": "b", "end": "a", "nominal": 10.0,
             "upper_deviation": 0.0, "lower_deviation": 0.0,
             "std_dev": 0.0, "distribution": "normal", "direction": -1},
        ],
        "mc_samples": 5000,
        "random_seed": 3,
    }


def test_zero_variance_chain_results(client):
    cid = _create(client, _zero_variance_payload())["chain_id"]
    res = client.get(f"/chains/{cid}").json()["result"]["results"]
    assert res["rss"]["sigma_mm"] == 0.0
    assert res["monte_carlo"]["sigma_mm"] == 0.0
    assert res["monte_carlo"]["lower_bound_mm"] == pytest.approx(0.2, abs=1e-9)
    assert res["monte_carlo"]["upper_bound_mm"] == pytest.approx(0.2, abs=1e-9)
    assert res["rss"]["mean_gap_mm"] == pytest.approx(0.2, abs=1e-9)


def test_zero_variance_gap_probability_deterministic(client):
    """缺陷回归：σ=0 的固定间隙链查询区间概率返回确定 0/1，不再 500。"""
    cid = _create(client, _zero_variance_payload())["chain_id"]
    # 固定间隙 0.2 被 [0.15, 0.25] 覆盖 -> 三方法全 1
    inside = client.post(f"/chains/{cid}/gap-probability",
                         json={"lower": 0.15, "upper": 0.25})
    assert inside.status_code == 200, inside.text
    p = inside.json()["probabilities"]
    assert p == {"rss": 1.0, "monte_carlo": 1.0, "worst_case": 1.0}
    assert inside.json()["degenerate_fixed_gap_mm"] == pytest.approx(
        0.2, abs=1e-9)
    # 区间不含固定间隙 -> 0
    outside = client.post(f"/chains/{cid}/gap-probability",
                          json={"lower": 0.21, "upper": 0.30})
    assert outside.status_code == 200
    assert outside.json()["probabilities"] == {
        "rss": 0.0, "monte_carlo": 0.0, "worst_case": 0.0}
    # 端点闭区间：恰好等于 0.2 也算落入
    edge = client.post(f"/chains/{cid}/gap-probability",
                       json={"lower": 0.2, "upper": 0.2})
    assert edge.json()["probabilities"]["rss"] == 1.0
    assert edge.json()["probabilities"]["monte_carlo"] == 1.0
    # 单侧无界
    tail = client.post(f"/chains/{cid}/gap-probability",
                       json={"lower": 0.2})
    assert tail.json()["probabilities"]["rss"] == 1.0


def test_cost_targets(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.post(f"/chains/{cid}/cost-targets", json={
        "target_reject_rate": 0.0001,
        "tightening_cost": {"L1": 10.0, "L3": 6.0},
        "default_cost": 2.0,
        "scale_levels": [0.5, 0.7, 0.9, 1.0],
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["candidates"]) >= 1
    costs = [c["total_cost"] for c in j["candidates"]]
    assert costs == sorted(costs)
    for c in j["candidates"]:
        assert c["mc_reject_rate"] <= 0.0001
        for item in c["combination"]:
            if item["dimension_id"] == "L1":
                assert item["unit_cost_per_mm"] == 10.0
            if item["dimension_id"] == "L2":
                assert item["unit_cost_per_mm"] == 2.0


def test_cost_targets_requires_limits(client):
    payload = {
        "name": "no-limits",
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02},
            {"id": "B", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02,
             "direction": -1},
        ],
        "mc_samples": 3000,
    }
    cid = _create(client, payload)["chain_id"]
    r = client.post(f"/chains/{cid}/cost-targets",
                    json={"target_reject_rate": 0.01})
    assert r.status_code == 422
    assert "上下限" in r.text


def test_traceability(client, simple_chain_payload):
    cid = _create(client, simple_chain_payload)["chain_id"]
    r = client.get(f"/chains/{cid}/traceability")
    assert r.status_code == 200
    j = r.json()
    assert "rss" in j["formulas_by_method"]
    assert any("ρ_ij" in f for f in j["formulas_by_method"]["rss"])
    assert j["traceability"]["monte_carlo"]["seed"] == 7
    ids = [d["dimension_id"] for d in j["normalized_inputs"]]
    assert ids == ["L1", "L2", "L3"]
