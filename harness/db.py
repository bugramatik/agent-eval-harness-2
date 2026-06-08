"""Engine, session factory and table creation for the harness database."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from harness.config import Settings, get_settings
from harness.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the shared engine, creating it on first use."""

    global _engine, _session_factory
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return the session factory bound to the shared engine."""

    get_engine(settings)
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Open a session, commit on success and roll back on error."""

    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(settings: Settings | None = None) -> None:
    """Create the tables when they are missing; safe to call on every start."""

    Base.metadata.create_all(bind=get_engine(settings))


def reset_engine() -> None:
    """Drop the cached engine and sessions, used by tests that swap the database URL."""

    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
