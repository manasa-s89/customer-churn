"""
Database configuration and session management for Customer Churn backend.
Supports SQLite (default) and PostgreSQL via DATABASE_URL environment variable.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Default to local SQLite database, or use PostgreSQL if DATABASE_URL is set
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(db_dir, "churn.db")

    if os.getenv("VERCEL"):
        # Vercel serverless functions have a read-only root filesystem; only /tmp is writable
        tmp_db_path = "/tmp/churn.db"
        if not os.path.exists(tmp_db_path) and os.path.exists(default_db_path):
            import shutil
            try:
                shutil.copy2(default_db_path, tmp_db_path)
            except Exception as e:
                print(f"[!] Note: Could not copy seed db to /tmp: {e}")
        DATABASE_URL = f"sqlite:///{tmp_db_path}"
    else:
        DATABASE_URL = f"sqlite:///{default_db_path}"

# SQLite requires 'check_same_thread': False for multithreaded FastAPI access
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    # Import models to ensure they are registered with Base.metadata
    import models_db  # noqa: F401
    Base.metadata.create_all(bind=engine)
