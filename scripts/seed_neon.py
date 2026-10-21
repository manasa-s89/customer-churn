"""
scripts/seed_neon.py
One-time database seeder for Neon PostgreSQL.
Copies all subscriber records, model prediction scores, and retention interventions
from the local SQLite churn.db into the target PostgreSQL database.

Usage:
    python scripts/seed_neon.py
    # or specify custom DB:
    DATABASE_URL="postgresql://..." python scripts/seed_neon.py
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Add application directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "customer churn"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 1. Load DATABASE_URL from .env file if not already set in environment
env_candidates = [
    app_dir / ".env",
    root_dir / ".env",
    Path.cwd() / ".env",
    Path.cwd() / "customer churn" / ".env"
]

try:
    from dotenv import load_dotenv  # type: ignore
    for env_path in env_candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    pass

# Fallback parser if python-dotenv is not installed
if not os.environ.get("DATABASE_URL"):
    for env_path in env_candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k == "DATABASE_URL" and v:
                            os.environ[k] = v
                            break
            except Exception as e:
                print(f"[!] Warning reading {env_path}: {e}")
            if os.environ.get("DATABASE_URL"):
                break

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] DATABASE_URL is not set. Please set it in .env or your environment.")
    sys.exit(1)

# Normalize URL to postgresql+psycopg2 for SQLAlchemy
clean_url = DATABASE_URL
if clean_url.startswith("postgres://"):
    clean_url = clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+"):
    clean_url = clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)

if "sslmode=" not in clean_url and "sqlite" not in clean_url:
    sep = "&" if "?" in clean_url else "?"
    clean_url = f"{clean_url}{sep}sslmode=require"

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import models_db
from database import Base

print("[*] Connecting to PostgreSQL database...")
engine = create_engine(
    clean_url,
    echo=False,
    pool_pre_ping=True
)

Session = sessionmaker(bind=engine)

# 2. Ensure all tables are created
print("[*] Creating database schema/tables if not present...")
Base.metadata.create_all(bind=engine)

# 3. Check if tables are already populated
with Session() as session:
    existing_count = session.query(func.count(models_db.CustomerRecord.customer_id)).scalar() or 0
    if existing_count > 0:
        score_count = session.query(func.count(models_db.ChurnScoreRecord.id)).scalar() or 0
        int_count = session.query(func.count(models_db.InterventionRecord.id)).scalar() or 0
        print(f"[OK] Database already contains {existing_count} customers, {score_count} scores, and {int_count} interventions.")
        print("[!] Seeding skipped to prevent duplicates. Database is ready!")
        sys.exit(0)

# 4. Locate local SQLite seed database
sqlite_candidates = [
    app_dir / "churn.db",
    root_dir / "churn.db",
    Path.cwd() / "customer churn" / "churn.db",
    Path.cwd() / "churn.db"
]
sqlite_path = None
for p in sqlite_candidates:
    if p.is_file() and p.stat().st_size > 0:
        sqlite_path = p
        break

if not sqlite_path:
    print("[ERROR] Could not find local churn.db SQLite seed database.")
    sys.exit(1)

print(f"[*] Reading seed data from SQLite: {sqlite_path}")
conn = sqlite3.connect(str(sqlite_path))
cur = conn.cursor()

with Session() as session:
    # A. Seed Customers
    cur.execute("""
        SELECT customer_id, subscription_plan, monthly_price, watch_hours_last_30_days,
               days_since_last_watch, number_of_devices, number_of_profiles, downloads_count,
               customer_support_tickets, payment_failures, free_trial_converted, tenure_months, created_at
        FROM customers
    """)
    cust_rows = cur.fetchall()
    print(f"[*] Found {len(cust_rows)} customer records to seed...")
    cust_objs = []
    for r in cust_rows:
        dt = None
        if r[12]:
            try:
                dt = datetime.fromisoformat(r[12])
            except Exception:
                dt = datetime.now(timezone.utc)
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
            created_at=dt
        ))
    for i in range(0, len(cust_objs), 1000):
        session.bulk_save_objects(cust_objs[i:i + 1000])
        session.flush()
    print(f"[OK] Seeded {len(cust_objs)} customers.")

    # B. Seed Churn Scores
    cur.execute("""
        SELECT customer_id, churn_probability, risk_tier, predicted_churn,
               top_risk_drivers_json, top_protective_factors_json, explanation_summary,
               recommended_action, created_at
        FROM churn_scores
    """)
    score_rows = cur.fetchall()
    print(f"[*] Found {len(score_rows)} churn score records to seed...")
    score_objs = []
    for r in score_rows:
        dt = None
        if r[8]:
            try:
                dt = datetime.fromisoformat(r[8])
            except Exception:
                dt = datetime.now(timezone.utc)
        score_objs.append(models_db.ChurnScoreRecord(
            customer_id=r[0],
            churn_probability=r[1],
            risk_tier=r[2],
            predicted_churn=bool(r[3]),
            top_risk_drivers_json=r[4],
            top_protective_factors_json=r[5],
            explanation_summary=r[6],
            recommended_action=r[7],
            created_at=dt
        ))
    for i in range(0, len(score_objs), 1000):
        session.bulk_save_objects(score_objs[i:i + 1000])
        session.flush()
    print(f"[OK] Seeded {len(score_objs)} churn scores.")

    # C. Seed Interventions
    cur.execute("""
        SELECT customer_id, action_type, title, description, urgency,
               recommended_channel, status, notes, performed_by, created_at, updated_at
        FROM interventions
    """)
    int_rows = cur.fetchall()
    print(f"[*] Found {len(int_rows)} intervention records to seed...")
    int_objs = []
    for r in int_rows:
        dt = None
        if r[9]:
            try:
                dt = datetime.fromisoformat(r[9])
            except Exception:
                dt = datetime.now(timezone.utc)
        up_dt = None
        if r[10]:
            try:
                up_dt = datetime.fromisoformat(r[10])
            except Exception:
                up_dt = dt
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
            created_at=dt,
            updated_at=up_dt
        ))
    for i in range(0, len(int_objs), 500):
        session.bulk_save_objects(int_objs[i:i + 500])
        session.flush()
    print(f"[OK] Seeded {len(int_objs)} interventions.")

    session.commit()

conn.close()

# 5. Final Verification
with Session() as session:
    c_count = session.query(func.count(models_db.CustomerRecord.customer_id)).scalar()
    s_count = session.query(func.count(models_db.ChurnScoreRecord.id)).scalar()
    i_count = session.query(func.count(models_db.InterventionRecord.id)).scalar()

print("=" * 60)
print("NEON POSTGRES SEEDING COMPLETED SUCCESSFULLY!")
print(f"  Total Customers:     {c_count}")
print(f"  Total Churn Scores:  {s_count}")
print(f"  Total Interventions: {i_count}")
print("=" * 60)
