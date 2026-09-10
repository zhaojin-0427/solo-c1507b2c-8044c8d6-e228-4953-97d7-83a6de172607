"""测试夹具：每个测试模块使用独立的临时 SQLite 数据库。"""
import os
import tempfile

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["TOLCHAIN_DB"] = _tmp.name


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app import database as db
    from app.main import app

    # 确保全新建表
    if os.path.exists(_tmp.name):
        os.remove(_tmp.name)
    db._engine = None
    db.get_engine()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def simple_chain_payload():
    return {
        "name": "test-chain",
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
        "mc_samples": 50000,
        "random_seed": 7,
    }
