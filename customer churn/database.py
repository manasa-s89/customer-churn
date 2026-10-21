"""
Database configuration and session management for Customer Churn backend.
Supports SQLite (default) and PostgreSQL via DATABASE_URL environment variable.
"""

import os
import shutil
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# Load environment variables from .env file if present next to main.py / database.py
_env_files = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.getcwd(), ".env"),
    os.path.join(os.getcwd(), "customer churn", ".env")
]
for _env_file in _env_files:
    if os.path.isfile(_env_file):
        try:
            with open(_env_file, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _k, _v = _k.strip(), _v.strip().strip("'\"")
                        if _k and _k not in os.environ and _v:
                            os.environ[_k] = _v
        except Exception:
            pass
        break

# Default to local SQLite database, or use PostgreSQL if DATABASE_URL is set
DATABASE_URL = os.getenv("DATABASE_URL")

db_dir = os.path.dirname(os.path.abspath(__file__))
default_db_path = os.path.join(db_dir, "churn.db")

candidate_seed_paths = [
    default_db_path,
    os.path.join(os.path.dirname(db_dir), "churn.db"),
    os.path.join(os.getcwd(), "customer churn", "churn.db"),
    os.path.join(os.getcwd(), "churn.db")
]

if DATABASE_URL:
    # Normalize postgres:// and postgresql:// to postgresql+psycopg2:// for psycopg2-binary compatibility
    clean_url = DATABASE_URL
    if clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+"):
        clean_url = clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    # Ensure sslmode=require for Neon / cloud Postgres
    if "sslmode=" not in clean_url and "sqlite" not in clean_url:
        sep = "&" if "?" in clean_url else "?"
        clean_url = f"{clean_url}{sep}sslmode=require"

    DATABASE_URL = clean_url

    engine_kwargs = {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10
    }
else:
    # Local SQLite or Vercel serverless SQLite
    if os.getenv("VERCEL"):
        # Vercel serverless functions have a read-only root filesystem; only /tmp is writable
        tmp_dir = tempfile.gettempdir()
        tmp_db_path = os.path.join(tmp_dir, "churn.db")

        if not os.path.exists(tmp_db_path) or os.path.getsize(tmp_db_path) == 0:
            for seed_path in candidate_seed_paths:
                if os.path.exists(seed_path) and os.path.getsize(seed_path) > 0:
                    try:
                        shutil.copy2(seed_path, tmp_db_path)
                        try:
                            os.chmod(tmp_db_path, 0o666)
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        print(f"[!] Note: Could not copy seed db from {seed_path} to {tmp_db_path}: {e}")

        if os.path.exists(tmp_db_path):
            try:
                os.chmod(tmp_db_path, 0o666)
            except Exception:
                pass

        clean_tmp_path = tmp_db_path.replace("\\", "/")
        DATABASE_URL = f"sqlite:///{clean_tmp_path}"
    else:
        clean_default_path = default_db_path.replace("\\", "/")
        DATABASE_URL = f"sqlite:///{clean_default_path}"

    engine_kwargs = {
        "echo": False,
        "connect_args": {"check_same_thread": False},
        "poolclass": NullPool
    }

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


def _seed_from_sqlite(db, sqlite_path):
    """Bulk copy rows from seed SQLite database into Postgres or new database."""
    import sqlite3
    from datetime import datetime, timezone
    import models_db

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # 1. Customers
    cur.execute("SELECT customer_id, subscription_plan, monthly_price, watch_hours_last_30_days, days_since_last_watch, number_of_devices, number_of_profiles, downloads_count, customer_support_tickets, payment_failures, free_trial_converted, tenure_months, created_at FROM customers")
    cust_rows = cur.fetchall()
    cust_objs = []
    for r in cust_rows:
        created_at = None
        if r[12]:
            try:
                created_at = datetime.fromisoformat(r[12])
            except Exception:
                created_at = datetime.now(timezone.utc)
        cust_objs.append(models_db.CustomerRecord(
            customer_id=r[0],
            subscription_plan=r[1],
            monthly_price=r[2],
            watch_hours_last_30_days=r[3],
            days_since_last_watch=r[4],
            number_of_devices=r[5],
            number_of_profiles=r[6],
            downloads_count=r[7],
            customer_support_tickets=r[8],
            payment_failures=r[9],
            free_trial_converted=r[10],
            tenure_months=r[11],
            created_at=created_at
        ))
    for i in range(0, len(cust_objs), 1000):
        db.bulk_save_objects(cust_objs[i:i + 1000])
        db.flush()

    # 2. Churn Scores
    cur.execute("SELECT customer_id, churn_probability, risk_tier, predicted_churn, top_risk_drivers_json, top_protective_factors_json, explanation_summary, recommended_action, created_at FROM churn_scores")
    score_rows = cur.fetchall()
    score_objs = []
    for r in score_rows:
        created_at = None
        if r[8]:
            try:
                created_at = datetime.fromisoformat(r[8])
            except Exception:
                created_at = datetime.now(timezone.utc)
        score_objs.append(models_db.ChurnScoreRecord(
            customer_id=r[0],
            churn_probability=r[1],
            risk_tier=r[2],
            predicted_churn=bool(r[3]),
            top_risk_drivers_json=r[4],
            top_protective_factors_json=r[5],
            explanation_summary=r[6],
            recommended_action=r[7],
            created_at=created_at
        ))
    for i in range(0, len(score_objs), 1000):
        db.bulk_save_objects(score_objs[i:i + 1000])
        db.flush()

    # 3. Interventions
    cur.execute("SELECT customer_id, action_type, title, description, urgency, recommended_channel, status, notes, performed_by, created_at, updated_at FROM interventions")
    int_rows = cur.fetchall()
    int_objs = []
    for r in int_rows:
        created_at = None
        if r[9]:
            try:
                created_at = datetime.fromisoformat(r[9])
            except Exception:
                created_at = datetime.now(timezone.utc)
        updated_at = None
        if r[10]:
            try:
                updated_at = datetime.fromisoformat(r[10])
            except Exception:
                updated_at = created_at
        int_objs.append(models_db.InterventionRecord(
            customer_id=r[0],
            action_type=r[1],
            title=r[2],
            description=r[3],
            urgency=r[4] or "High",
            recommended_channel=r[5] or "email",
            status=r[6] or "Sent",
            notes=r[7],
            performed_by=r[8] or "Retention Lead",
            created_at=created_at,
            updated_at=updated_at
        ))
    for i in range(0, len(int_objs), 500):
        db.bulk_save_objects(int_objs[i:i + 500])
        db.flush()

    db.commit()
    conn.close()


def _check_and_seed_database():
    """Seed customers, churn_scores, and interventions if DB is empty, preventing duplicate cold-start seeding."""
    import models_db
    from sqlalchemy import func

    with SessionLocal() as db:
        try:
            cust_count = db.query(func.count(models_db.CustomerRecord.customer_id)).scalar() or 0
        except Exception as e:
            print(f"[!] Note: Could not query customer count for seeding check: {e}")
            return

        if cust_count > 0:
            # Table already populated (previous deployment or cold start) - do not duplicate
            return

        print("[*] Empty database detected. Loading seed data (customers, interventions, scores)...", flush=True)

        seed_sqlite_path = None
        for p in candidate_seed_paths:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                seed_sqlite_path = p
                break

        if seed_sqlite_path:
            try:
                _seed_from_sqlite(db, seed_sqlite_path)
                print(f"[OK] Successfully seeded database from {seed_sqlite_path}.", flush=True)
                return
            except Exception as e:
                print(f"[!] Error seeding from SQLite: {e}. Trying CSV fallback...", flush=True)
                db.rollback()

        # Fallback to CSV seeder if SQLite seed was not available or failed
        _seed_from_csv(db)


def _seed_from_csv(db):
    """Fallback seeder using CSV dataset."""
    import csv
    from datetime import datetime, timezone, timedelta
    import models_db
    import uuid

    csv_candidates = [
        os.path.join(db_dir, "data", "ott_customer_churn.csv"),
        os.path.join(os.path.dirname(db_dir), "customer churn", "data", "ott_customer_churn.csv"),
        os.path.join(os.getcwd(), "customer churn", "data", "ott_customer_churn.csv"),
        os.path.join(os.getcwd(), "data", "ott_customer_churn.csv")
    ]
    csv_file = None
    for p in csv_candidates:
        if os.path.isfile(p):
            csv_file = p
            break

    if not csv_file:
        print("[!] Warning: No seed CSV found for database initialization.")
        return

    try:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for i, r in enumerate(reader) if i < 250]

        now = datetime.now(timezone.utc)
        cust_objs = []
        for r in rows:
            cid = str(r.get("customer_id", f"OTT-{uuid.uuid4().hex[:6].upper()}")).strip()
            plan = str(r.get("subscription_plan", "Basic")).strip().capitalize()
            price = float(r.get("monthly_price", 8.99) or 8.99)
            watch = float(r.get("watch_hours_last_30_days", 15.0) or 15.0)
            days = int(float(r.get("days_since_last_watch", 2) or 2))
            dev = int(float(r.get("number_of_devices", 1) or 1))
            prof = int(float(r.get("number_of_profiles", 1) or 1))
            down = int(float(r.get("downloads_count", 0) or 0))
            tix = int(float(r.get("customer_support_tickets", 0) or 0))
            fails = int(float(r.get("payment_failures", 0) or 0))
            trial = str(r.get("free_trial_converted", "Yes")).strip().capitalize()
            tenure = int(float(r.get("tenure_months", 3) or 3))

            cust_objs.append(models_db.CustomerRecord(
                customer_id=cid,
                subscription_plan=plan,
                monthly_price=price,
                watch_hours_last_30_days=watch,
                days_since_last_watch=days,
                number_of_devices=dev,
                number_of_profiles=prof,
                downloads_count=down,
                customer_support_tickets=tix,
                payment_failures=fails,
                free_trial_converted=trial,
                tenure_months=tenure,
                created_at=now
            ))
        db.bulk_save_objects(cust_objs)
        db.flush()

        # Seed sample interventions
        sample_actions = [
            ("proactive_billing_support", "Proactive Billing Support & Grace Period", "Sent 7-day streaming grace email while payment is updated.", "in-app", "Resolved"),
            ("pause_subscription_offer", "Pause Subscription Offer (1-3 Months Free Hold)", "Offered 60-day pause instead of cancellation.", "in-app", "Responded"),
            ("personalized_content_discovery", "Curated Sci-Fi Watchlist", "Dispatched personalized recommendations matching viewing history.", "email", "Sent"),
            ("annual_discount_retention", "Annual Plan Switch Loyalty Discount", "Offered 25% discount to switch to Annual Standard.", "email", "Pending")
        ]
        target_ids = [c.customer_id for c in cust_objs[:4]]
        for i, cid in enumerate(target_ids):
            act = sample_actions[i % len(sample_actions)]
            int_rec = models_db.InterventionRecord(
                customer_id=cid,
                action_type=act[0],
                title=act[1],
                description=act[2],
                urgency="High",
                recommended_channel=act[3],
                status=act[4],
                notes="Seeded on first database initialization.",
                performed_by="Retention Concierge",
                created_at=now - timedelta(days=i + 1),
                updated_at=now - timedelta(days=i)
            )
            db.add(int_rec)

        db.commit()
        print(f"[OK] Successfully seeded database from CSV ({len(cust_objs)} subscribers).", flush=True)
    except Exception as e:
        print(f"[!] Error seeding from CSV: {e}")
        db.rollback()


def init_db():
    """Initialize database tables and seed Postgres/SQLite on first start."""
    try:
        import models_db  # noqa: F401
        Base.metadata.create_all(bind=engine)
        _check_and_seed_database()
    except Exception as e:
        print(f"[!] Warning: Database init_db failed: {e}")
