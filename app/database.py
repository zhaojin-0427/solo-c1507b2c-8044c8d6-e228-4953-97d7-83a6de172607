"""SQLite 持久层（SQLAlchemy Core + ORM）。

存储：
* chains             基线尺寸链（输入 + 规范化 + 结果 JSON）
* scenarios          方案分支（属于某条链，永不覆盖基线）
* measurement_plans  不可变测量方案（量具误差声明 + 合成结果）
* inspection_batches 来料检验批次（冻结；可含测量方案快照与判定报告）
* thermal_analyses   热分析版本（冻结基线之上的热参数 + 工况 + 结果）
* thermal_proposals  热分析方案搜索结果（候选材料/垫片/基准温度，冻结）
* gage_rr_studies    量具 R&R 研究（单尺寸 ANOVA 交叉表 + 结果，冻结）
* networks           多闭环公差网络（版本线；共享尺寸池的多个功能要求）
* network_versions   网络版本（尺寸池/闭环/相关矩阵快照 + 结果，冻结）
* network_scenarios  网络方案分支搜索结果（锁定尺寸 + 批量调整，冻结）
* hole_patterns      孔系装配对象（版本线；一对零件的孔/销/螺栓与基准框架）
* hole_versions      孔系版本（匹配位/基准框架快照 + WC/MC 结果，冻结）
* hole_remedies      孔系整改搜索结果（候选钻孔/连接件/孔位修正；采纳后冻结）
* wear_studies       服役磨损研究（冻结基线链上的磨损曲线/节点/限值，冻结）
* wear_maintenances  磨损维护编排搜索结果（继续/垫片/更换方案；选定后冻结）
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
    UniqueConstraint,
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


class MeasurementPlanRow(Base):
    """不可变测量方案：创建后只允许读取，新版本另建行。"""

    __tablename__ = "measurement_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 原始请求与规范化合成结果（mm）
    request_json: Mapped[dict] = mapped_column(JSON)
    combined_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class InspectionBatchRow(Base):
    """来料检验批次：创建后即冻结，只允许读取。"""

    __tablename__ = "inspection_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 提交的原始测量行（值 + 单位 + 缺测标记）
    rows_json: Mapped[list] = mapped_column(JSON)
    # 创建时一次性算好的统计报告，保证多次读取内容不变
    report_json: Mapped[dict] = mapped_column(JSON)
    comparison_json: Mapped[dict] = mapped_column(JSON)
    bootstrap_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    # 量具误差判定：引用的测量方案与冻结的判定报告（含方案快照）
    measurement_plan_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    measurement_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    frozen: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AssemblyTaskRow(Base):
    """选择性装配任务（版本线）：首个版本创建时建行。"""

    __tablename__ = "assembly_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AssemblyVersionRow(Base):
    """装配任务版本：创建即冻结（含来源批次 / 测量方案 / 种子快照）。"""

    __tablename__ = "assembly_versions"
    __table_args__ = (
        UniqueConstraint("task_id", "version_no", name="uq_task_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("assembly_tasks.id", ondelete="CASCADE"), index=True
    )
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ThermalAnalysisRow(Base):
    """热分析版本：从冻结基线链建立，创建即冻结（基线不被覆盖）。"""

    __tablename__ = "thermal_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 原始请求（逐尺寸热参数 / 工况 / 相关矩阵原样冻结）
    request_json: Mapped[dict] = mapped_column(JSON)
    # 归一化热模型快照（T0 °C / α K^-1 / 工况温度）
    model_json: Mapped[dict] = mapped_column(JSON)
    # 创建时一次性算好的逐工况结果（含固定种子 MC）
    result_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    frozen: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ThermalProposalRow(Base):
    """热分析方案搜索结果：候选表 / 种子随请求冻结，重复读取结果不变。"""

    __tablename__ = "thermal_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thermal_analysis_id: Mapped[int] = mapped_column(
        ForeignKey("thermal_analyses.id", ondelete="CASCADE"), index=True
    )
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GageRrStudyRow(Base):
    """量具 R&R 研究：创建即冻结（交叉表、ANOVA 结果与 bootstrap 配置固化）。"""

    __tablename__ = "gage_rr_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    dimension_id: Mapped[str] = mapped_column(String(200), index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 原始请求（交叉表 / 单位 / 过程公差原样冻结）与创建时算好的完整结果
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    bootstrap_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    frozen: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class NetworkRow(Base):
    """多闭环公差网络（版本线）：首个版本创建时建行。"""

    __tablename__ = "networks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 首版本的来源基线链（只读引用，基线不被改写）；纯尺寸池网络为 None
    source_chain_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class NetworkVersionRow(Base):
    """网络版本：尺寸池 / 闭环 / 相关矩阵快照与结果随版本冻结，不回改。"""

    __tablename__ = "network_versions"
    __table_args__ = (
        UniqueConstraint("network_id", "version_no", name="uq_network_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(
        ForeignKey("networks.id", ondelete="CASCADE"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    # 提交的原始请求（含来源链 id 与追加尺寸原样）
    request_json: Mapped[dict] = mapped_column(JSON)
    # 归一化快照（尺寸池 mm / 闭环路径系数 / 相关矩阵 / 种子 / 来源）
    snapshot_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class NetworkScenarioRow(Base):
    """网络方案分支搜索结果：锁定尺寸与批量调整候选随请求冻结。"""

    __tablename__ = "network_scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(
        ForeignKey("networks.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[int] = mapped_column(
        ForeignKey("network_versions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ProcessPlanRow(Base):
    """工序尺寸方案（版本线）：一次零件加工路线独立成版，首版创建时建行。"""

    __tablename__ = "process_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 首版引用的冻结基线链（只读快照；无则 None）
    source_chain_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ProcessPlanVersionRow(Base):
    """工序方案版本：设计链快照/工序路线/传递矩阵/种子随版本冻结，不回改。"""

    __tablename__ = "process_plan_versions"
    __table_args__ = (
        UniqueConstraint("plan_id", "version_no",
                         name="uq_process_plan_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("process_plans.id", ondelete="CASCADE"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    snapshot_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ProcessSolutionRow(Base):
    """反算结果：锁定/档位/步进候选随请求冻结；选定后冻结选中候选。"""

    __tablename__ = "process_solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("process_plans.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[int] = mapped_column(
        ForeignKey("process_plan_versions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    selected_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_note: Mapped[str] = mapped_column(Text, default="")
    frozen: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class HolePatternRow(Base):
    """孔系装配对象（版本线）：一对零件的孔 / 销 / 螺栓与基准框架。"""

    __tablename__ = "hole_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class HoleVersionRow(Base):
    """孔系版本：匹配位 / 基准框架 / 抽样配置与结果随版本冻结，不回改。"""

    __tablename__ = "hole_versions"
    __table_args__ = (
        UniqueConstraint("pattern_id", "version_no",
                         name="uq_hole_pattern_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pattern_id: Mapped[int] = mapped_column(
        ForeignKey("hole_patterns.id", ondelete="CASCADE"), index=True
    )
    version_no: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    snapshot_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class HoleRemedyRow(Base):
    """孔系整改搜索结果：候选钻孔 / 连接件 / 孔位修正随请求冻结；采纳后固化。"""

    __tablename__ = "hole_remedies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pattern_id: Mapped[int] = mapped_column(
        ForeignKey("hole_patterns.id", ondelete="CASCADE"), index=True
    )
    version_id: Mapped[int] = mapped_column(
        ForeignKey("hole_versions.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    selected_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_version_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    selected_note: Mapped[str] = mapped_column(Text, default="")
    frozen: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class WearStudyRow(Base):
    """服役磨损研究：冻结基线链上独立成版，创建即冻结（基线不被覆盖）。"""

    __tablename__ = "wear_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # 原始请求（逐尺寸磨损曲线 / 计算节点 / 封闭环限值原样冻结）
    request_json: Mapped[dict] = mapped_column(JSON)
    # 归一化磨损模型快照（曲线 mm / 速率 σ / 相关矩阵 / 节点）
    model_json: Mapped[dict] = mapped_column(JSON)
    # 创建时一次性算好的逐节点结果（含固定种子 MC 与首次越界分布）
    result_json: Mapped[dict] = mapped_column(JSON)
    mc_samples: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    frozen: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class WearMaintenanceRow(Base):
    """磨损维护编排搜索结果：候选表 / 种子随请求冻结；选定方案后固化。"""

    __tablename__ = "wear_maintenances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_id: Mapped[int] = mapped_column(
        ForeignKey("wear_studies.id", ondelete="CASCADE"), index=True
    )
    chain_id: Mapped[int] = mapped_column(
        ForeignKey("chains.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    request_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON)
    selected_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_note: Mapped[str] = mapped_column(Text, default="")
    frozen: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


_engine: Any = None


def _migrate(engine) -> None:
    """旧库轻量迁移：inspection_batches 补测量方案相关列。"""
    from sqlalchemy import inspect, text

    cols = {c["name"] for c in inspect(engine).get_columns("inspection_batches")}
    with engine.begin() as conn:
        if "measurement_plan_id" not in cols:
            conn.execute(text(
                "ALTER TABLE inspection_batches "
                "ADD COLUMN measurement_plan_id INTEGER"
            ))
        if "measurement_json" not in cols:
            conn.execute(text(
                "ALTER TABLE inspection_batches "
                "ADD COLUMN measurement_json JSON"
            ))


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DB_URL, connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(_engine)
        _migrate(_engine)
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


# ------------------------------------------------------- 测量方案 CRUD

def save_measurement_plan(chain_id: int, name: str, note: str,
                          request: dict, combined: dict) -> int:
    with session_factory() as s:
        row = MeasurementPlanRow(
            chain_id=chain_id, name=name, note=note,
            request_json=request, combined_json=combined,
        )
        s.add(row)
        s.commit()
        return row.id


def get_measurement_plan(plan_id: int) -> MeasurementPlanRow | None:
    with session_factory() as s:
        row = s.get(MeasurementPlanRow, plan_id)
        if row is not None:
            s.expunge(row)
        return row


def list_measurement_plans(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(MeasurementPlanRow)
            .where(MeasurementPlanRow.chain_id == chain_id)
            .order_by(MeasurementPlanRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "name": r.name,
                "note": r.note,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ------------------------------------------------------- 来料检验批次 CRUD

def save_inspection_batch(chain_id: int, name: str, note: str,
                          rows: list[dict], report: dict, comparison: dict,
                          bootstrap_samples: int, seed: int,
                          measurement_plan_id: int | None = None,
                          measurement: dict | None = None) -> int:
    with session_factory() as s:
        row = InspectionBatchRow(
            chain_id=chain_id, name=name, note=note,
            rows_json=rows, report_json=report, comparison_json=comparison,
            bootstrap_samples=bootstrap_samples, random_seed=seed, frozen=1,
            measurement_plan_id=measurement_plan_id,
            measurement_json=measurement,
        )
        s.add(row)
        s.commit()
        return row.id


def get_inspection_batch(batch_id: int) -> InspectionBatchRow | None:
    with session_factory() as s:
        row = s.get(InspectionBatchRow, batch_id)
        if row is not None:
            s.expunge(row)
        return row


def list_inspection_batches(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(InspectionBatchRow)
            .where(InspectionBatchRow.chain_id == chain_id)
            .order_by(InspectionBatchRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "name": r.name,
                "note": r.note,
                "bootstrap_samples": r.bootstrap_samples,
                "random_seed": r.random_seed,
                "measurement_plan_id": r.measurement_plan_id,
                "frozen": bool(r.frozen),
                "rows_total": len(r.rows_json),
                "complete_rows": r.report_json["sample_summary"]["complete_rows"],
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ------------------------------------------------------- 选择性装配 CRUD

def save_assembly_first_version(chain_id: int, name: str, note: str,
                                request: dict, result: dict) -> tuple[int, int]:
    """事务性创建任务行与版本 1，返回 (task_id, version_id)。"""
    with session_factory() as s:
        task = AssemblyTaskRow(chain_id=chain_id, name=name, note=note)
        s.add(task)
        s.flush()
        version = AssemblyVersionRow(
            task_id=task.id, chain_id=chain_id, version_no=1,
            parent_version_id=None, name=name, note=note,
            request_json=request, result_json=result,
        )
        s.add(version)
        s.commit()
        return task.id, version.id


def save_assembly_version(task_id: int, chain_id: int, version_no: int,
                          parent_version_id: int, name: str, note: str,
                          request: dict, result: dict) -> int:
    with session_factory() as s:
        version = AssemblyVersionRow(
            task_id=task_id, chain_id=chain_id, version_no=version_no,
            parent_version_id=parent_version_id, name=name, note=note,
            request_json=request, result_json=result,
        )
        s.add(version)
        s.commit()
        return version.id


def get_assembly_version(version_id: int) -> AssemblyVersionRow | None:
    with session_factory() as s:
        row = s.get(AssemblyVersionRow, version_id)
        if row is not None:
            s.expunge(row)
        return row


def get_assembly_task(task_id: int) -> AssemblyTaskRow | None:
    with session_factory() as s:
        row = s.get(AssemblyTaskRow, task_id)
        if row is not None:
            s.expunge(row)
        return row


def list_assembly_versions(task_id: int) -> list[AssemblyVersionRow]:
    with session_factory() as s:
        rows = s.scalars(
            select(AssemblyVersionRow)
            .where(AssemblyVersionRow.task_id == task_id)
            .order_by(AssemblyVersionRow.version_no)
        ).all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def list_assembly_tasks(chain_id: int) -> list[dict]:
    with session_factory() as s:
        tasks = s.scalars(
            select(AssemblyTaskRow)
            .where(AssemblyTaskRow.chain_id == chain_id)
            .order_by(AssemblyTaskRow.id)
        ).all()
        out = []
        for t in tasks:
            versions = s.scalars(
                select(AssemblyVersionRow)
                .where(AssemblyVersionRow.task_id == t.id)
                .order_by(AssemblyVersionRow.version_no)
            ).all()
            out.append({
                "task_id": t.id,
                "chain_id": t.chain_id,
                "name": t.name,
                "note": t.note,
                "created_at": t.created_at.isoformat(),
                "versions": [
                    {
                        "version_id": v.id,
                        "version_no": v.version_no,
                        "parent_version_id": v.parent_version_id,
                        "note": v.note,
                        "status": v.result_json["status"],
                        "required_count": v.result_json["required_count"],
                        "qualified_assemblies":
                            v.result_json["qualified_assemblies"],
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
            })
        return out


# ------------------------------------------------------------- 热分析 CRUD

def save_thermal_analysis(chain_id: int, name: str, note: str,
                          request: dict, model_snapshot: dict, result: dict,
                          mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        row = ThermalAnalysisRow(
            chain_id=chain_id, name=name, note=note,
            request_json=request, model_json=model_snapshot,
            result_json=result, mc_samples=mc_samples, random_seed=seed,
            frozen=1,
        )
        s.add(row)
        s.commit()
        return row.id


def get_thermal_analysis(analysis_id: int) -> ThermalAnalysisRow | None:
    with session_factory() as s:
        row = s.get(ThermalAnalysisRow, analysis_id)
        if row is not None:
            s.expunge(row)
        return row


def list_thermal_analyses(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(ThermalAnalysisRow)
            .where(ThermalAnalysisRow.chain_id == chain_id)
            .order_by(ThermalAnalysisRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "name": r.name,
                "note": r.note,
                "mc_samples": r.mc_samples,
                "random_seed": r.random_seed,
                "frozen": bool(r.frozen),
                "conditions": [
                    c["analytic"]["name"] for c in r.result_json["conditions"]
                ],
                "worst_reject_probability":
                    r.result_json["summary"]["worst_reject_probability"],
                "minimum_spec_margin_mm":
                    r.result_json["summary"]["minimum_spec_margin_mm"],
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def save_thermal_proposal(analysis_id: int, chain_id: int, name: str,
                          note: str, request: dict, result: dict) -> int:
    with session_factory() as s:
        row = ThermalProposalRow(
            thermal_analysis_id=analysis_id, chain_id=chain_id,
            name=name, note=note, request_json=request, result_json=result)
        s.add(row)
        s.commit()
        return row.id


def get_thermal_proposal(proposal_id: int) -> ThermalProposalRow | None:
    with session_factory() as s:
        row = s.get(ThermalProposalRow, proposal_id)
        if row is not None:
            s.expunge(row)
        return row


def list_thermal_proposals(analysis_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(ThermalProposalRow)
            .where(ThermalProposalRow.thermal_analysis_id == analysis_id)
            .order_by(ThermalProposalRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "thermal_analysis_id": r.thermal_analysis_id,
                "name": r.name,
                "note": r.note,
                "candidate_count": len(
                    r.result_json.get("proposals", [])),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ------------------------------------------------------- 量具 R&R 研究 CRUD

def save_gage_rr_study(chain_id: int, dimension_id: str, name: str, note: str,
                       request: dict, result: dict,
                       bootstrap_samples: int, seed: int) -> int:
    with session_factory() as s:
        row = GageRrStudyRow(
            chain_id=chain_id, dimension_id=dimension_id, name=name,
            note=note, request_json=request, result_json=result,
            bootstrap_samples=bootstrap_samples, random_seed=seed, frozen=1,
        )
        s.add(row)
        s.commit()
        return row.id


def get_gage_rr_study(study_id: int) -> GageRrStudyRow | None:
    with session_factory() as s:
        row = s.get(GageRrStudyRow, study_id)
        if row is not None:
            s.expunge(row)
        return row


def list_gage_rr_studies(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(GageRrStudyRow)
            .where(GageRrStudyRow.chain_id == chain_id)
            .order_by(GageRrStudyRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "dimension_id": r.dimension_id,
                "name": r.name,
                "note": r.note,
                "bootstrap_samples": r.bootstrap_samples,
                "random_seed": r.random_seed,
                "frozen": bool(r.frozen),
                "total_gage_std_mm": r.result_json["total_gage_std_mm"],
                "ndc": r.result_json["ndc"],
                "dominant_source": r.result_json["dominant_source"]["component"],
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ------------------------------------------------------- 多闭环网络 CRUD

def save_network_first_version(name: str, note: str,
                               source_chain_id: int | None,
                               request: dict, snapshot: dict, result: dict,
                               mc_samples: int, seed: int) -> tuple[int, int]:
    """事务性创建网络行与版本 1，返回 (network_id, version_id)。"""
    with session_factory() as s:
        net = NetworkRow(name=name, note=note, source_chain_id=source_chain_id)
        s.add(net)
        s.flush()
        version = NetworkVersionRow(
            network_id=net.id, version_no=1, parent_version_id=None,
            name=name, note=note, request_json=request,
            snapshot_json=snapshot, result_json=result,
            mc_samples=mc_samples, random_seed=seed,
        )
        s.add(version)
        s.commit()
        return net.id, version.id


def save_network_version(network_id: int, version_no: int,
                         parent_version_id: int | None, name: str, note: str,
                         request: dict, snapshot: dict, result: dict,
                         mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        version = NetworkVersionRow(
            network_id=network_id, version_no=version_no,
            parent_version_id=parent_version_id, name=name, note=note,
            request_json=request, snapshot_json=snapshot,
            result_json=result, mc_samples=mc_samples, random_seed=seed,
        )
        s.add(version)
        s.commit()
        return version.id


def get_network(network_id: int) -> NetworkRow | None:
    with session_factory() as s:
        row = s.get(NetworkRow, network_id)
        if row is not None:
            s.expunge(row)
        return row


def get_network_version(version_id: int) -> NetworkVersionRow | None:
    with session_factory() as s:
        row = s.get(NetworkVersionRow, version_id)
        if row is not None:
            s.expunge(row)
        return row


def list_network_versions(network_id: int) -> list[NetworkVersionRow]:
    with session_factory() as s:
        rows = s.scalars(
            select(NetworkVersionRow)
            .where(NetworkVersionRow.network_id == network_id)
            .order_by(NetworkVersionRow.version_no)
        ).all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def list_networks() -> list[dict]:
    with session_factory() as s:
        nets = s.scalars(select(NetworkRow).order_by(NetworkRow.id)).all()
        out = []
        for net in nets:
            versions = s.scalars(
                select(NetworkVersionRow)
                .where(NetworkVersionRow.network_id == net.id)
                .order_by(NetworkVersionRow.version_no)
            ).all()
            out.append({
                "network_id": net.id,
                "name": net.name,
                "note": net.note,
                "source_chain_id": net.source_chain_id,
                "created_at": net.created_at.isoformat(),
                "versions": [
                    {
                        "version_id": v.id,
                        "version_no": v.version_no,
                        "parent_version_id": v.parent_version_id,
                        "note": v.note,
                        "loop_ids": [loop["loop_id"]
                                     for loop in v.snapshot_json["loops"]],
                        "joint_pass_rate_monte_carlo":
                            v.result_json["joint"]
                            ["joint_pass_rate_monte_carlo"],
                        "all_loops_in_spec_worst_case":
                            v.result_json["joint"]
                            ["all_loops_in_spec_worst_case"],
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
            })
        return out


def save_network_scenario(network_id: int, version_id: int, name: str,
                          note: str, request: dict, result: dict) -> int:
    with session_factory() as s:
        row = NetworkScenarioRow(
            network_id=network_id, version_id=version_id, name=name,
            note=note, request_json=request, result_json=result,
        )
        s.add(row)
        s.commit()
        return row.id


def get_network_scenario(scenario_id: int) -> NetworkScenarioRow | None:
    with session_factory() as s:
        row = s.get(NetworkScenarioRow, scenario_id)
        if row is not None:
            s.expunge(row)
        return row


def list_network_scenarios(version_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(NetworkScenarioRow)
            .where(NetworkScenarioRow.version_id == version_id)
            .order_by(NetworkScenarioRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "network_id": r.network_id,
                "version_id": r.version_id,
                "name": r.name,
                "note": r.note,
                "candidate_count": len(r.result_json.get("candidates", [])),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ------------------------------------------------------- 工序尺寸方案 CRUD

def save_process_plan_first_version(name: str, note: str,
                                    source_chain_id: int | None,
                                    request: dict, snapshot: dict,
                                    mc_samples: int, seed: int) -> tuple[int, int]:
    """事务性创建工序方案行与版本 1，返回 (plan_id, version_id)。"""
    with session_factory() as s:
        plan = ProcessPlanRow(name=name, note=note,
                              source_chain_id=source_chain_id)
        s.add(plan)
        s.flush()
        version = ProcessPlanVersionRow(
            plan_id=plan.id, version_no=1, parent_version_id=None,
            name=name, note=note, request_json=request,
            snapshot_json=snapshot, mc_samples=mc_samples, random_seed=seed)
        s.add(version)
        s.commit()
        return plan.id, version.id


def save_process_plan_version(plan_id: int, version_no: int,
                              parent_version_id: int | None, name: str,
                              note: str, request: dict, snapshot: dict,
                              mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        version = ProcessPlanVersionRow(
            plan_id=plan_id, version_no=version_no,
            parent_version_id=parent_version_id, name=name, note=note,
            request_json=request, snapshot_json=snapshot,
            mc_samples=mc_samples, random_seed=seed)
        s.add(version)
        s.commit()
        return version.id


def get_process_plan(plan_id: int) -> ProcessPlanRow | None:
    with session_factory() as s:
        row = s.get(ProcessPlanRow, plan_id)
        if row is not None:
            s.expunge(row)
        return row


def list_process_plans() -> list[dict]:
    with session_factory() as s:
        plans = s.scalars(
            select(ProcessPlanRow).order_by(ProcessPlanRow.id)).all()
        out = []
        for p in plans:
            versions = s.scalars(
                select(ProcessPlanVersionRow)
                .where(ProcessPlanVersionRow.plan_id == p.id)
                .order_by(ProcessPlanVersionRow.version_no)
            ).all()
            out.append({
                "plan_id": p.id,
                "name": p.name,
                "note": p.note,
                "source_chain_id": p.source_chain_id,
                "created_at": p.created_at.isoformat(),
                "versions": [
                    {
                        "version_id": v.id,
                        "version_no": v.version_no,
                        "parent_version_id": v.parent_version_id,
                        "note": v.note,
                        "operations": len(v.snapshot_json["edges"]),
                        "closures": len(v.snapshot_json["closures"]),
                        "mc_samples": v.mc_samples,
                        "random_seed": v.random_seed,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
            })
        return out


def get_process_plan_version(version_id: int) -> ProcessPlanVersionRow | None:
    with session_factory() as s:
        row = s.get(ProcessPlanVersionRow, version_id)
        if row is not None:
            s.expunge(row)
        return row


def list_process_plan_versions(plan_id: int) -> list[ProcessPlanVersionRow]:
    with session_factory() as s:
        rows = s.scalars(
            select(ProcessPlanVersionRow)
            .where(ProcessPlanVersionRow.plan_id == plan_id)
            .order_by(ProcessPlanVersionRow.version_no)
        ).all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def save_process_solution(plan_id: int, version_id: int, name: str, note: str,
                          request: dict, result: dict) -> int:
    with session_factory() as s:
        row = ProcessSolutionRow(
            plan_id=plan_id, version_id=version_id, name=name, note=note,
            request_json=request, result_json=result, frozen=0)
        s.add(row)
        s.commit()
        return row.id


def get_process_solution(solution_id: int) -> ProcessSolutionRow | None:
    with session_factory() as s:
        row = s.get(ProcessSolutionRow, solution_id)
        if row is not None:
            s.expunge(row)
        return row


def list_process_solutions(version_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(ProcessSolutionRow)
            .where(ProcessSolutionRow.version_id == version_id)
            .order_by(ProcessSolutionRow.id)
        ).all()
        return [
            {
                "solution_id": r.id,
                "plan_id": r.plan_id,
                "version_id": r.version_id,
                "name": r.name,
                "note": r.note,
                "frozen": bool(r.frozen),
                "selected_rank": r.selected_rank,
                "candidate_count": len(r.result_json.get("candidates", [])),
                "status": r.result_json.get("status"),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def freeze_process_solution(solution_id: int, rank: int, note: str) -> None:
    """幂等选定：写入 selected_rank 并置 frozen=1（调用方保证只选一次）。"""
    with session_factory() as s:
        row = s.get(ProcessSolutionRow, solution_id)
        row.selected_rank = rank
        row.selected_note = note
        row.frozen = 1
        s.commit()


# ------------------------------------------------------- 孔系装配 CRUD

def save_hole_first_version(name: str, note: str, request: dict,
                            snapshot: dict, result: dict,
                            mc_samples: int, seed: int) -> tuple[int, int]:
    """事务性创建孔系对象行与版本 1，返回 (pattern_id, version_id)。"""
    with session_factory() as s:
        pattern = HolePatternRow(name=name, note=note)
        s.add(pattern)
        s.flush()
        version = HoleVersionRow(
            pattern_id=pattern.id, version_no=1, parent_version_id=None,
            name=name, note=note, request_json=request,
            snapshot_json=snapshot, result_json=result,
            mc_samples=mc_samples, random_seed=seed)
        s.add(version)
        s.commit()
        return pattern.id, version.id


def save_hole_version(pattern_id: int, version_no: int,
                      parent_version_id: int | None, name: str, note: str,
                      request: dict, snapshot: dict, result: dict,
                      mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        version = HoleVersionRow(
            pattern_id=pattern_id, version_no=version_no,
            parent_version_id=parent_version_id, name=name, note=note,
            request_json=request, snapshot_json=snapshot,
            result_json=result, mc_samples=mc_samples, random_seed=seed)
        s.add(version)
        s.commit()
        return version.id


def get_hole_pattern(pattern_id: int) -> HolePatternRow | None:
    with session_factory() as s:
        row = s.get(HolePatternRow, pattern_id)
        if row is not None:
            s.expunge(row)
        return row


def list_hole_patterns() -> list[dict]:
    with session_factory() as s:
        patterns = s.scalars(
            select(HolePatternRow).order_by(HolePatternRow.id)).all()
        out = []
        for p in patterns:
            versions = s.scalars(
                select(HoleVersionRow)
                .where(HoleVersionRow.pattern_id == p.id)
                .order_by(HoleVersionRow.version_no)).all()
            out.append({
                "pattern_id": p.id,
                "name": p.name,
                "note": p.note,
                "created_at": p.created_at.isoformat(),
                "versions": [
                    {"version_id": v.id, "version_no": v.version_no,
                     "parent_version_id": v.parent_version_id,
                     "note": v.note, "mc_samples": v.mc_samples,
                     "random_seed": v.random_seed,
                     "mates": len(v.snapshot_json["mates"]),
                     "worst_case_feasible":
                         v.result_json["worst_case"]["feasible"],
                     "assembly_success_rate":
                         v.result_json["monte_carlo"]
                         ["assembly_success_rate"],
                     "created_at": v.created_at.isoformat()}
                    for v in versions],
            })
        return out


def get_hole_version(version_id: int) -> HoleVersionRow | None:
    with session_factory() as s:
        row = s.get(HoleVersionRow, version_id)
        if row is not None:
            s.expunge(row)
        return row


def list_hole_versions(pattern_id: int) -> list[HoleVersionRow]:
    with session_factory() as s:
        rows = s.scalars(
            select(HoleVersionRow)
            .where(HoleVersionRow.pattern_id == pattern_id)
            .order_by(HoleVersionRow.version_no)).all()
        for r in rows:
            s.expunge(r)
        return list(rows)


def save_hole_remedy(pattern_id: int, version_id: int, name: str, note: str,
                     request: dict, result: dict) -> int:
    with session_factory() as s:
        row = HoleRemedyRow(
            pattern_id=pattern_id, version_id=version_id, name=name,
            note=note, request_json=request, result_json=result)
        s.add(row)
        s.commit()
        return row.id


def get_hole_remedy(remedy_id: int) -> HoleRemedyRow | None:
    with session_factory() as s:
        row = s.get(HoleRemedyRow, remedy_id)
        if row is not None:
            s.expunge(row)
        return row


def list_hole_remedies(version_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(HoleRemedyRow)
            .where(HoleRemedyRow.version_id == version_id)
            .order_by(HoleRemedyRow.id)).all()
        return [
            {"remedy_id": r.id, "pattern_id": r.pattern_id,
             "version_id": r.version_id, "name": r.name, "note": r.note,
             "frozen": bool(r.frozen),
             "selected_rank": r.selected_rank,
             "selected_version_id": r.selected_version_id,
             "candidate_count": len(r.result_json.get("candidates", [])),
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]


def freeze_hole_remedy(remedy_id: int, rank: int, note: str,
                       new_version_id: int) -> None:
    """采纳：写入 rank、新版本 id 并置 frozen=1（调用方保证只采纳一次）。"""
    with session_factory() as s:
        row = s.get(HoleRemedyRow, remedy_id)
        row.selected_rank = rank
        row.selected_version_id = new_version_id
        row.selected_note = note
        row.frozen = 1
        s.commit()


# ------------------------------------------------------- 服役磨损研究 CRUD

def save_wear_study(chain_id: int, name: str, note: str,
                    request: dict, model_snapshot: dict, result: dict,
                    mc_samples: int, seed: int) -> int:
    with session_factory() as s:
        row = WearStudyRow(
            chain_id=chain_id, name=name, note=note,
            request_json=request, model_json=model_snapshot,
            result_json=result, mc_samples=mc_samples, random_seed=seed,
            frozen=1,
        )
        s.add(row)
        s.commit()
        return row.id


def get_wear_study(study_id: int) -> WearStudyRow | None:
    with session_factory() as s:
        row = s.get(WearStudyRow, study_id)
        if row is not None:
            s.expunge(row)
        return row


def list_wear_studies(chain_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(WearStudyRow)
            .where(WearStudyRow.chain_id == chain_id)
            .order_by(WearStudyRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "chain_id": r.chain_id,
                "name": r.name,
                "note": r.note,
                "mc_samples": r.mc_samples,
                "random_seed": r.random_seed,
                "frozen": bool(r.frozen),
                "node_count": r.result_json["summary"]["node_count"],
                "final_cycles": r.result_json["summary"]["final_cycles"],
                "worst_reject_probability":
                    r.result_json["summary"]["worst_reject_probability"],
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def save_wear_maintenance(study_id: int, chain_id: int, name: str,
                          note: str, request: dict, result: dict) -> int:
    with session_factory() as s:
        row = WearMaintenanceRow(
            study_id=study_id, chain_id=chain_id, name=name, note=note,
            request_json=request, result_json=result, frozen=0)
        s.add(row)
        s.commit()
        return row.id


def get_wear_maintenance(search_id: int) -> WearMaintenanceRow | None:
    with session_factory() as s:
        row = s.get(WearMaintenanceRow, search_id)
        if row is not None:
            s.expunge(row)
        return row


def list_wear_maintenances(study_id: int) -> list[dict]:
    with session_factory() as s:
        rows = s.scalars(
            select(WearMaintenanceRow)
            .where(WearMaintenanceRow.study_id == study_id)
            .order_by(WearMaintenanceRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "study_id": r.study_id,
                "chain_id": r.chain_id,
                "name": r.name,
                "note": r.note,
                "frozen": bool(r.frozen),
                "selected_rank": r.selected_rank,
                "candidate_count": len(r.result_json.get("candidates", [])),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def freeze_wear_maintenance(search_id: int, rank: int, note: str) -> None:
    """幂等选定：写入 selected_rank 并置 frozen=1（调用方保证只选一次）。"""
    with session_factory() as s:
        row = s.get(WearMaintenanceRow, search_id)
        row.selected_rank = rank
        row.selected_note = note
        row.frozen = 1
        s.commit()
