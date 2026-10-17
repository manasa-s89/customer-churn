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
        import tempfile
        import shutil

        tmp_dir = tempfile.gettempdir()
        tmp_db_path = os.path.join(tmp_dir, "churn.db")

        # Locate seed churn.db across potential root/subfolder working directories
        candidate_seed_paths = [
            default_db_path,
            os.path.join(os.getcwd(), "customer churn", "churn.db"),
            os.path.join(os.getcwd(), "churn.db")
        ]

        if not os.path.exists(tmp_db_path):
            for seed_path in candidate_seed_paths:
                if os.path.exists(seed_path):
                    try:
                        shutil.copy2(seed_path, tmp_db_path)
                        break
                    except Exception as e:
                        print(f"[!] Note: Could not copy seed db from {seed_path} to {tmp_db_path}: {e}")

        # Format URL with forward slashes for cross-platform SQLite URL compatibility
        clean_tmp_path = tmp_db_path.replace("\\", "/")
        DATABASE_URL = f"sqlite:///{clean_tmp_path}"
    else:
        clean_default_path = default_db_path.replace("\\", "/")
        DATABASE_URL = f"sqlite:///{clean_default_path}"

from sqlalchemy.pool import NullPool

# SQLite requires 'check_same_thread': False for multithreaded FastAPI access
connect_args = {}
engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    engine_kwargs["connect_args"] = connect_args
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 30

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs
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
