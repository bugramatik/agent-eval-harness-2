"""SQLAlchemy tables for runs, per-task results and the task suite."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pass_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    results: Mapped[list["TaskResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="TaskResult.id"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Run id={self.id} suite={self.suite!r} status={self.status!r}>"


class TaskResult(Base):
    __tablename__ = "task_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    judge_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    run: Mapped[Run] = relationship(back_populates="results")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TaskResult run={self.run_id} task={self.task_id!r} passed={self.passed}>"


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    suite: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    expected_tool: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_args: Mapped[str | None] = mapped_column(Text, nullable=True)
    must_contain: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Task id={self.id!r} suite={self.suite!r}>"
