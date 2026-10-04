import sys
import os
import io

print("[1/7] Initializing TestClient with FastAPI app...", flush=True)
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("[2/7] Testing GET / and GET /dashboard...", flush=True)
res_dash = client.get("/dashboard")
assert res_dash.status_code == 200, f"Dashboard failed: {res_dash.status_code}"
assert "<!DOCTYPE html>" in res_dash.text or "RetainPulse" in res_dash.text, "Dashboard HTML missing"
print("  Dashboard HTML served successfully (Length: " + str(len(res_dash.text)) + " bytes).", flush=True)

res_root = client.get("/")
assert res_root.status_code == 200, f"Root failed: {res_root.status_code}"
print("  Root route verified.", flush=True)

print("[3/7] Testing GET /health...", flush=True)
res_health = client.get("/health")
assert res_health.status_code == 200
h_data = res_health.json()
print("  Health status:", h_data, flush=True)

print("[4/7] Testing GET /analytics/overview...", flush=True)
res_ov = client.get("/analytics/overview")
assert res_ov.status_code == 200, f"Analytics failed: {res_ov.text}"
ov_data = res_ov.json()
print("  Analytics KPIs:", {
    "total_customers": ov_data["total_customers"],
    "high_risk_count": ov_data["high_risk_count"],
    "high_risk_pct": f"{ov_data['high_risk_pct']}%",
    "revenue_at_risk": f"${ov_data['revenue_at_risk']}/mo",
    "avg_churn_probability": f"{ov_data['avg_churn_probability']*100:.1f}%",
    "risk_distribution": ov_data["risk_distribution"]
}, flush=True)

print("[5/7] Testing GET /customers with pagination and filtering...", flush=True)
res_cust = client.get("/customers?limit=10")
assert res_cust.status_code == 200
c_data = res_cust.json()
print(f"  Total customers in directory: {c_data['total_customers']} (retrieved {len(c_data['customers'])} in page 1)", flush=True)
top_cust = c_data["customers"][0]
print(f"  Top risk customer: {top_cust['customer_id']} | Risk: {top_cust['risk_tier']} | Prob: {top_cust['churn_probability_pct']}", flush=True)

# Test Customer 360 View
res_detail = client.get(f"/customers/{top_cust['customer_id']}")
assert res_detail.status_code == 200
detail = res_detail.json()
print(f"  Customer 360 retrieved: {detail['customer']['customer_id']} with {len(detail['current_recommendations']['suggested_actions'])} suggested retention actions.", flush=True)

print("[6/7] Testing GET /interventions & PUT /interventions/{id}...", flush=True)
res_ints = client.get("/interventions")
assert res_ints.status_code == 200
ints = res_ints.json()
print(f"  Retrieved {len(ints)} total intervention records.", flush=True)
first_int = ints[0]
res_upd = client.put(f"/interventions/{first_int['id']}", json={"status": "Resolved", "notes": "Verified by test suite"})
assert res_upd.status_code == 200
assert res_upd.json()["status"] == "Resolved"
print(f"  Updated intervention #{first_int['id']} status to 'Resolved'.", flush=True)

print("[7/7] Testing POST /upload_csv (Batch CSV Scoring)...", flush=True)
sample_csv = """customer_id,tenure,monthly_charges,total_charges,contract_type,payment_method,internet_service,support_tickets,last_login_days_ago,usage_frequency
BATCH-01,2,105.50,211.00,Month-to-month,Electronic check,Fiber optic,5,42,Rarely
BATCH-02,55,39.00,2145.00,Two year,Bank transfer (automatic),DSL,0,3,Daily
BATCH-03,12,85.00,1020.00,Month-to-month,Credit card (automatic),Fiber optic,3,15,Weekly
"""
csv_file = io.BytesIO(sample_csv.encode("utf-8"))
res_upload = client.post("/upload_csv", files={"file": ("test_batch.csv", csv_file, "text/csv")})
assert res_upload.status_code == 200, f"Upload CSV failed: {res_upload.text}"
up_data = res_upload.json()
print("  Batch upload result:", {
    "rows_processed": up_data["rows_processed"],
    "high_risk_detected": up_data["high_risk_detected"],
    "medium_risk_detected": up_data["medium_risk_detected"],
    "low_risk_detected": up_data["low_risk_detected"],
    "revenue_at_risk": f"${up_data['total_revenue_at_risk']}/mo"
}, flush=True)

print("\n========================================================")
print("[ALL 7 FRONTEND & BACKEND MODULES VERIFIED SUCCESSFULLY!]")
print("========================================================")
