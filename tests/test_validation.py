"""校验规则测试：闭合性、重复边、相关矩阵、偏差自洽、单边/混合单位。"""
import pytest


def _closed_payload():
    return {
        "name": "v",
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02},
            {"id": "B", "start": "b", "end": "c", "nominal": 5,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02},
            {"id": "C", "start": "c", "end": "a", "nominal": 15,
             "upper_deviation": 0.2, "lower_deviation": -0.2, "std_dev": 0.03,
             "direction": -1},
        ],
        "mc_samples": 5000,
    }


def test_open_chain_rejected(client):
    p = _closed_payload()
    p["dimensions"][2]["start"] = "a"   # a-c 反向边，度数不平衡
    p["dimensions"][2]["end"] = "c"
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "方向不闭合" in r.text


def test_duplicate_edge_rejected(client):
    p = _closed_payload()
    p["dimensions"][2]["id"] = "C2"
    p["dimensions"][2]["start"] = "a"
    p["dimensions"][2]["end"] = "b"   # 与 A 同方向平行边 -> 重复
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "重复边" in r.text


def test_antiparallel_edges_form_valid_loop(client):
    """a→b 与 b→a 是两件配合的合法闭合环。"""
    p = {
        "name": "two-part",
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02},
            {"id": "B", "start": "b", "end": "a", "nominal": 9.8,
             "upper_deviation": 0.05, "lower_deviation": -0.05,
             "std_dev": 0.02, "direction": -1},
        ],
        "mc_samples": 3000,
    }
    r = client.post("/chains", json=p)
    assert r.status_code == 201, r.text
    res = r.json()["result"]
    dims = {d["dimension_id"]: d for d in res["normalized_inputs"]}
    assert dims["A"]["sign"] == 1
    assert dims["B"]["sign"] == -1
    assert res["results"]["worst_case"]["nominal_gap_mm"] == pytest.approx(
        0.2, abs=1e-9)


def test_duplicate_dimension_id_rejected(client):
    p = _closed_payload()
    p["dimensions"][2]["id"] = "A"
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "id 重复" in r.text


def test_invalid_correlation_matrix_rejected(client):
    p = _closed_payload()
    p["correlations"] = [
        {"dim_a": "A", "dim_b": "B", "rho": 0.9},
        {"dim_a": "A", "dim_b": "C", "rho": 0.9},
        {"dim_a": "B", "dim_b": "C", "rho": -0.9},
    ]
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "半正定" in r.text


def test_correlation_out_of_range_rejected(client):
    p = _closed_payload()
    p["correlations"] = [{"dim_a": "A", "dim_b": "B", "rho": 1.5}]
    r = client.post("/chains", json=p)
    assert r.status_code == 422


def test_correlation_unknown_dimension_rejected(client):
    p = _closed_payload()
    p["correlations"] = [{"dim_a": "A", "dim_b": "ZZ", "rho": 0.2}]
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "不存在的尺寸" in r.text


def test_deviations_order_rejected(client):
    p = _closed_payload()
    p["dimensions"][0]["upper_deviation"] = -0.2
    p["dimensions"][0]["lower_deviation"] = 0.1
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "下偏差" in r.text


def test_normal_requires_sigma(client):
    p = _closed_payload()
    del p["dimensions"][0]["std_dev"]
    r = client.post("/chains", json=p)
    assert r.status_code == 422
    assert "std_dev" in r.text


def test_unilateral_flag(client):
    from app.schemas import DimensionInput
    uni = DimensionInput(
        id="x", start="a", end="b", nominal=10,
        upper_deviation=0.1, lower_deviation=0.0, std_dev=0.02,
    )
    bi = DimensionInput(
        id="y", start="a", end="b", nominal=10,
        upper_deviation=0.1, lower_deviation=-0.1, std_dev=0.02,
    )
    assert uni.is_unilateral is True
    assert bi.is_unilateral is False


def test_inch_unit_normalized(client):
    p = _closed_payload()
    p["dimensions"][1]["unit"] = "in"
    p["dimensions"][1]["nominal"] = 5 / 25.4
    p["dimensions"][1]["upper_deviation"] = 0.1 / 25.4
    p["dimensions"][1]["lower_deviation"] = -0.1 / 25.4
    p["dimensions"][1]["std_dev"] = 0.02 / 25.4
    r = client.post("/chains", json=p)
    assert r.status_code == 201, r.text
    dims = {d["dimension_id"]: d for d in r.json()["result"]["normalized_inputs"]}
    assert dims["B"]["normalized_mm"]["nominal"] == pytest.approx(5.0, abs=1e-9)
    assert dims["B"]["normalized_mm"]["half_width"] == pytest.approx(0.1, abs=1e-9)
    assert dims["B"]["original_representation"]["unit"] == "in"
