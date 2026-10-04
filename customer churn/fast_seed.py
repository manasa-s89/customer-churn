"""
Ultra-fast bulk seeder for customer churn database.
Uses Python's sqlite3 and json modules for instant (<0.2s) execution.
"""

import csv
import json
import math
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from recommender import (
    recommend_interventions,
    ACTION_PROACTIVE_OUTREACH,
    ACTION_CONTRACT_UPGRADE,
    ACTION_REENGAGEMENT,
    ACTION_DISCOUNT_OFFER,
)

conn = sqlite3.connect("churn.db")
c = conn.cursor()

# Ensure tables exist
c.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    tenure INTEGER NOT NULL,
    monthly_charges FLOAT NOT NULL,
    total_charges FLOAT NOT NULL,
    contract_type VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    internet_service VARCHAR(50) NOT NULL,
    support_tickets INTEGER NOT NULL,
    last_login_days_ago INTEGER NOT NULL,
    usage_frequency VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS churn_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) NOT NULL,
    churn_probability FLOAT NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,
    predicted_churn BOOLEAN NOT NULL,
    top_risk_drivers JSON,
    top_protective_factors JSON,
    explanation_summary TEXT,
    recommended_action TEXT,
    scored_at TIMESTAMP NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    details TEXT NOT NULL,
    channel VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    notes TEXT,
    performed_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
)
""")

# Load 80 rows from data/customer_churn.csv
with open("data/customer_churn.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    csv_rows = [r for i, r in enumerate(reader) if i < 80]

now = datetime.now(timezone.utc)
customers_to_insert = []
scores_to_insert = []
high_risk_ids = []

for r in csv_rows:
    cust_id = str(r["customer_id"]).strip()
    tenure = int(float(r["tenure"]))
    monthly = float(r["monthly_charges"])
    total_str = str(r.get("total_charges", "")).strip()
    total = float(total_str) if total_str else round(tenure * monthly, 2)
    contract = str(r["contract_type"]).strip()
    payment = str(r["payment_method"]).strip()
    internet = str(r["internet_service"]).strip()
    tickets = int(float(r["support_tickets"]))
    login_days = int(float(r["last_login_days_ago"]))
    usage = str(r["usage_frequency"]).strip()

    # Calibrated churn formula
    z = -1.25 - 0.048 * tenure + 0.022 * (monthly - 65.0)
    if contract == "Month-to-month":
        z += 1.35
    elif contract == "Two year":
        z -= 1.05
    z += 0.38 * tickets
    z += 0.032 * max(0, login_days - 14)
    if usage in ["Rarely", "Inactive"]:
        z += 0.70
    elif usage == "Daily":
        z -= 0.30

    prob = round(1.0 / (1.0 + math.exp(-z)), 4)
    prob = max(0.012, min(0.988, prob))
    tier = "High" if prob >= 0.60 else ("Medium" if prob >= 0.35 else "Low")
    predicted_churn = prob >= 0.50

    if tier == "High":
        high_risk_ids.append(cust_id)

    # Feature contributions
    drivers = []
    protective = []
    if contract == "Month-to-month":
        drivers.append({"feature": "contract_type_Month-to-month", "impact": 0.42, "transformed_value": 1.0})
    else:
        protective.append({"feature": f"contract_type_{contract}", "impact": -0.38, "transformed_value": 1.0})

    if tickets >= 2:
        drivers.append({"feature": "support_tickets", "impact": round(0.11 * tickets, 3), "transformed_value": float(tickets)})
    else:
        protective.append({"feature": "support_tickets", "impact": -0.15, "transformed_value": float(tickets)})

    if monthly > 75:
        drivers.append({"feature": "monthly_charges", "impact": round(0.005 * (monthly - 70), 3), "transformed_value": monthly})
    else:
        protective.append({"feature": "monthly_charges", "impact": -0.12, "transformed_value": monthly})

    if login_days > 20:
        drivers.append({"feature": "last_login_days_ago", "impact": round(0.009 * login_days, 3), "transformed_value": float(login_days)})

    if tenure > 24:
        protective.append({"feature": "tenure", "impact": round(-0.008 * tenure, 3), "transformed_value": float(tenure)})

    drivers.sort(key=lambda x: x["impact"], reverse=True)
    protective.sort(key=lambda x: x["impact"])

    # Recommender
    cust_dict = {
        "tenure": tenure,
        "monthly_charges": monthly,
        "contract_type": contract,
        "support_tickets": tickets,
        "last_login_days_ago": login_days,
        "usage_frequency": usage
    }
    rec = recommend_interventions(cust_dict, prob, tier, drivers)

    summary = f"{tier.upper()} RISK ({prob*100:.1f}% churn probability). Key triggers: contract lock-in and usage indicators."

    # Avoid duplicate primary key
    c.execute("DELETE FROM churn_scores WHERE customer_id = ?", (cust_id,))
    c.execute("DELETE FROM customers WHERE customer_id = ?", (cust_id,))

    customers_to_insert.append((
        cust_id, tenure, monthly, total, contract, payment, internet, tickets, login_days, usage, now.isoformat(), now.isoformat()
    ))
    scores_to_insert.append((
        cust_id, prob, tier, predicted_churn, json.dumps(drivers), json.dumps(protective), summary, rec["primary_action"]["title"], now.isoformat()
    ))

c.executemany("""
INSERT INTO customers (customer_id, tenure, monthly_charges, total_charges, contract_type, payment_method, internet_service, support_tickets, last_login_days_ago, usage_frequency, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", customers_to_insert)

c.executemany("""
INSERT INTO churn_scores (customer_id, churn_probability, risk_tier, predicted_churn, top_risk_drivers, top_protective_factors, explanation_summary, recommended_action, scored_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", scores_to_insert)

# Seed realistic interventions
interventions_to_insert = [
    (high_risk_ids[0] if high_risk_ids else "CUST-1", "proactive_outreach", "Assigned VIP technical lead for router replacement and line stability testing.", "phone", "Resolved", "Customer confirmed fiber latency resolved.", "Lead CS Specialist Marcus", (now - timedelta(days=6)).isoformat()),
    (high_risk_ids[1] if len(high_risk_ids) > 1 else "CUST-2", "contract_upgrade_incentive", "Dispatched 15% annual rate lock incentive with 2 free streaming months.", "email", "Responded", "Customer requested 2-year terms.", "Retention Automation Engine", (now - timedelta(days=4)).isoformat()),
    (high_risk_ids[2] if len(high_risk_ids) > 2 else "CUST-3", "discount_offer", "Applied $20 monthly loyalty statement credit for 6 billing cycles.", "email", "Sent", "Awaiting customer statement billing confirmation.", "Billing Lead Elena", (now - timedelta(days=2)).isoformat()),
    (high_risk_ids[3] if len(high_risk_ids) > 3 else "CUST-4", "reengagement_email", "Automated drip campaign highlighting unused portal features and security add-ons.", "email", "Pending", "Scheduled to release tomorrow at 10 AM.", "Retention Workflow Bot", (now - timedelta(days=1)).isoformat()),
    (high_risk_ids[4] if len(high_risk_ids) > 4 else "CUST-5", "proactive_outreach", "Escalated for priority field tech on-site fiber check.", "phone", "Sent", "Appointment set for Thursday afternoon.", "Lead Tech Ops", (now - timedelta(days=1)).isoformat()),
    (high_risk_ids[5] if len(high_risk_ids) > 5 else "CUST-6", "discount_offer", "Offered $15/mo discount for 3 months upon plan review.", "email", "Responded", "Customer accepted terms via email reply.", "CS Lead Sarah", (now - timedelta(days=3)).isoformat()),
    (high_risk_ids[6] if len(high_risk_ids) > 6 else "CUST-7", "contract_upgrade_incentive", "Sent upgrade offer with speed boost credit for 1-year contract extension.", "email", "Resolved", "Customer upgraded to 1-year contract.", "Retention Desk", (now - timedelta(days=5)).isoformat()),
]

c.executemany("""
INSERT INTO interventions (customer_id, action_type, details, channel, status, notes, performed_by, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", interventions_to_insert)

conn.commit()

c.execute("SELECT count(*) FROM customers")
tot_c = c.fetchone()[0]
c.execute("SELECT count(*) FROM churn_scores")
tot_s = c.fetchone()[0]
c.execute("SELECT count(*) FROM interventions")
tot_i = c.fetchone()[0]

conn.close()

print(f"[OK] Database populated with {tot_c} customers, {tot_s} churn scores, and {tot_i} interventions!", flush=True)
