from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.migrations import run_migrations


def create_db_engine(database_url: str) -> Engine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    run_migrations(engine)


def get_session(engine: Engine) -> Session:
    return Session(engine)
