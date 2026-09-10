"""API 端到端：持久化、方案分支不覆盖基线、批量调整、成本搜索、追溯。"""


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
    assert policies["L2"] == "scaled"


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
