"""
Lightweight database seeder for OTT streaming subscriber records and interventions.
Seeds subscribers, runs inference with the OTT model, assigns 4 risk tiers, and populates sample playbooks.
"""

import os
import csv
from datetime import datetime, timedelta, timezone

from database import SessionLocal, init_db, engine, Base
import models_db
from schemas import CustomerInput
from main import _process_single_prediction
from recommender import (
    ACTION_BILLING_SUPPORT,
    ACTION_PAUSE_SUBSCRIPTION,
    ACTION_CONTENT_RECOMMENDATION,
    ACTION_ANNUAL_DISCOUNT,
    ACTION_STREAMING_CONCIERGE,
    ACTION_VIP_LOYALTY
)
from generate_data import generate_ott_churn_dataset


def seed_database(limit: int = 250):
    # Re-create tables cleanly to align with OTT schema
    Base.metadata.drop_all(bind=engine)
    init_db()
    db = SessionLocal()

    csv_path = "data/ott_customer_churn.csv"
    if not os.path.exists(csv_path):
        print(f"[*] Generating synthetic OTT data for seeding...", flush=True)
        df = generate_ott_churn_dataset(n_samples=max(limit, 1000), random_state=42)
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)

    print(f"[*] Reading top {limit} OTT subscriber records from {csv_path}...", flush=True)
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for i, r in enumerate(reader) if i < limit]

    print(f"[*] Processing and scoring {len(rows)} OTT subscribers with 4-tier risk classification...", flush=True)
    tier_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    critical_ids = []
    high_risk_ids = []

    for r in rows:
        cust_id = str(r["customer_id"]).strip()
        plan = str(r.get("subscription_plan", "Basic")).strip().capitalize()
        price = float(r.get("monthly_price", 8.99))
        watch_hours = float(r.get("watch_hours_last_30_days", 15.0))
        days_inactive = int(float(r.get("days_since_last_watch", 2)))
        devices = int(float(r.get("number_of_devices", 1)))
        profiles = int(float(r.get("number_of_profiles", 1)))
        downloads = int(float(r.get("downloads_count", 0)))
        tickets = int(float(r.get("customer_support_tickets", 0)))
        failures = int(float(r.get("payment_failures", 0)))
        trial = str(r.get("free_trial_converted", "Yes")).strip().capitalize()
        tenure = int(float(r.get("tenure_months", 3)))

        cust_input = CustomerInput(
            customer_id=cust_id,
            subscription_plan=plan,
            monthly_price=price,
            watch_hours_last_30_days=watch_hours,
            days_since_last_watch=days_inactive,
            number_of_devices=devices,
            number_of_profiles=profiles,
            downloads_count=downloads,
            customer_support_tickets=tickets,
            payment_failures=failures,
            free_trial_converted=trial,
            tenure_months=tenure
        )

        res = _process_single_prediction(cust_input, db)
        tier_counts[res.risk_tier] = tier_counts.get(res.risk_tier, 0) + 1
        if res.risk_tier == "Critical":
            critical_ids.append(cust_id)
        elif res.risk_tier == "High":
            high_risk_ids.append(cust_id)

    print(f"[OK] Persisted {len(rows)} subscribers. Risk Tier Distribution: {tier_counts}", flush=True)

    # Seed varied OTT interventions
    sample_interventions = [
        {
            "action_type": ACTION_BILLING_SUPPORT,
            "title": "Proactive Billing Support & 1-Click Payment Recovery",
            "details": "Sent automated SMS & in-app modal prompt offering 7-day streaming grace period while card is updated.",
            "channel": "in-app modal",
            "status": "Resolved",
            "notes": "Subscriber updated payment method to Apple Pay; grace period ended smoothly.",
            "performed_by": "Billing Concierge Team",
            "days_ago": 3
        },
        {
            "action_type": ACTION_PAUSE_SUBSCRIPTION,
            "title": "Pause Subscription Offer (1–3 Months Free Hold)",
            "details": "Offered subscriber 2 months pause instead of cancellation with saved watchlist intact.",
            "channel": "in-app modal",
            "status": "Responded",
            "notes": "Subscriber accepted 60-day pause; auto-resume scheduled for next month.",
            "performed_by": "AI Retention Engine",
            "days_ago": 5
        },
        {
            "action_type": ACTION_CONTENT_RECOMMENDATION,
            "title": "Personalized Content Discovery & Trending Watchlist Email",
            "details": "Triggered curated recommendations matching Sci-Fi & Crime Thriller viewing history.",
            "channel": "email",
            "status": "Sent",
            "notes": "Email opened; subscriber watched 2 episodes of featured series within 24h.",
            "performed_by": "Content Recommender Bot",
            "days_ago": 8
        },
        {
            "action_type": ACTION_ANNUAL_DISCOUNT,
            "title": "Annual Plan Switch Loyalty Discount (25% Savings)",
            "details": "Offered Basic subscriber 25% discount to upgrade to Annual Standard commitment.",
            "channel": "email",
            "status": "Pending",
            "notes": "Promotion banner active in account portal until billing cycle end.",
            "performed_by": "Growth Desk",
            "days_ago": 1
        },
        {
            "action_type": ACTION_STREAMING_CONCIERGE,
            "title": "VIP Streaming Quality & Device Optimization Outreach",
            "details": "Assisted with Smart TV 4K buffering optimization and network cache clearing.",
            "channel": "email",
            "status": "Resolved",
            "notes": "Subscriber confirmed zero buffering on FireTV 4K.",
            "performed_by": "Senior Tech Support Alex",
            "days_ago": 12
        }
    ]

    target_ids = (critical_ids[:3] + high_risk_ids[:4]) if (critical_ids or high_risk_ids) else [rows[0]["customer_id"]]
    now = datetime.now(timezone.utc)

    for i, cid in enumerate(target_ids):
        action_data = sample_interventions[i % len(sample_interventions)]
        created_time = now - timedelta(days=action_data["days_ago"])
        int_rec = models_db.InterventionRecord(
            customer_id=cid,
            action_type=action_data["action_type"],
            title=action_data["title"],
            description=action_data["details"],
            urgency="High",
            recommended_channel=action_data["channel"],
            status=action_data["status"],
            notes=action_data["notes"],
            performed_by=action_data["performed_by"],
            created_at=created_time,
            updated_at=created_time + timedelta(hours=12)
        )
        db.add(int_rec)

    db.commit()
    db.close()
    print("[OK] Seeding complete! Database is primed for OTT subscriber churn analytics.", flush=True)


if __name__ == "__main__":
    seed_database(250)
