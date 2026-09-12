"""SQLite 持久层（SQLAlchemy Core + ORM）。

存储：
* chains             基线尺寸链（输入 + 规范化 + 结果 JSON）
* scenarios          方案分支（属于某条链，永不覆盖基线）
* measurement_plans  不可变测量方案（量具误差声明 + 合成结果）
* inspection_batches 来料检验批次（冻结；可含测量方案快照与判定报告）
* thermal_analyses   热分析版本（冻结基线之上的热参数 + 工况 + 结果）
* thermal_proposals  热分析方案搜索结果（候选材料/垫片/基准温度，冻结）
* gage_rr_studies    量具 R&R 研究（单尺寸 ANOVA 交叉表 + 结果，冻结）
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
