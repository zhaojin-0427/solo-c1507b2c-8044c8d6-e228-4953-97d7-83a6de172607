"""SQLite 持久层（SQLAlchemy Core + ORM）。

存储：
* chains        基线尺寸链（输入 + 规范化 + 结果 JSON）
* scenarios     方案分支（属于某条链，永不覆盖基线）
* scenario_runs 方案/批量调整的每次计算结果（便于追溯）
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DB_PATH = os.environ.get(
    "TOLCHAIN_DB", os.path.join(os.path.dirname(__file__), "..", "tolchain.db")
)
DB_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChainRow(Base):
    __tablename__ = "chains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    normalized_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ScenarioRow(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(40), default="branch")  # branch/batch
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    overrides_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


_engine: Any = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DB_URL, connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(_engine)
    return _engine


def session_factory() -> Session:
    return Session(get_engine())


# ------------------------------------------------------------- CRUD 辅助

def save_chain(name: str, description: str, request_data: dict,
               normalized: dict, result: dict,
               mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        row = ChainRow(
            name=name,
            description=description,
            request_json=request_data,
            normalized_json=normalized,
            result_json=result,
            mc_samples=mc_samples,
            random_seed=seed,
        )
        s.add(row)
        s.commit()
        return row.id


def get_chain(chain_id: int) -> ChainRow | None:
    with session_factory() as s:
        row = s.get(ChainRow, chain_id)
        if row is not None:
            s.expunge(row)
        return row


def list_chains() -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(select(ChainRow).order_by(ChainRow.id)).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "mc_samples": r.mc_samples,
                "random_seed": r.random_seed,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def save_scenario(chain_id: int, kind: str, name: str, note: str,
                  overrides: dict, result: dict) -> int:
    with session_factory() as s:
        row = ScenarioRow(
            chain_id=chain_id, kind=kind, name=name, note=note,
            overrides_json=overrides, result_json=result,
        )
        s.add(row)
        s.commit()
        return row.id


def get_scenario(scenario_id: int) -> ScenarioRow | None:
    with session_factory() as s:
        row = s.get(ScenarioRow, scenario_id)
        if row is not None:
            s.expunge(row)
        return row


def list_scenarios(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(ScenarioRow)
            .where(ScenarioRow.chain_id == chain_id)
            .order_by(ScenarioRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "kind": r.kind,
                "name": r.name,
                "note": r.note,
                "overrides": r.overrides_json,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
