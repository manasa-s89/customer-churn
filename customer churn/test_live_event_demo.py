import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers={"User-Agent": "DemoClient/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def post(path, data=None):
    body = json.dumps(data).encode("utf-8") if data is not None else b"{}"
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "DemoClient/1.0"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

print("1. Checking Server Health...")
try:
    status, health_data = get("/health")
    print(f"Health Status: HTTP {status}")
    print(json.dumps(health_data, indent=2))
except Exception as e:
    print("Error contacting server:", e)
    exit(1)

print("\n2. Finding a High-Risk Subscriber from Database...")
status, cust_data = get("/customers?limit=10&risk_tier=High")
customers = cust_data.get("customers", [])
if not customers:
    status, cust_data = get("/customers?limit=10")
    customers = cust_data.get("customers", [])

target_cust = customers[0]
cust_id = target_cust["customer_id"]
initial_prob = target_cust["churn_probability"]
initial_tier = target_cust["risk_tier"]

print(f"Target Subscriber: {cust_id}")
print(f"Initial State: Tier={initial_tier}, Probability={initial_prob*100:.1f}%, Plan={target_cust['subscription_plan']}, Payment Failures={target_cust['payment_failures']}")

print("\n3. Ingesting Real-Time Event: 'payment_failure' (Card Declined on Renewal)...")
event_payload = {
    "customer_id": cust_id,
    "event_type": "payment_failure",
    "metadata": {
        "reason": "Card expired on auto-renewal",
        "processor_code": "DECLINE_EXPIRED_01"
    }
}

status, event_data = post("/events/ingest", event_payload)
print(f"Event Ingest Response (HTTP {status}):")
print(json.dumps(event_data, indent=2))

print("\n4. Checking Simulator Status & Running Single Step Simulation...")
status, step_data = post("/simulate/live-events/step")
print(f"Simulator Step Response (HTTP {status}):")
print(json.dumps(step_data, indent=2))

status, sim_status = get("/simulate/live-events/status")
print(f"\nSimulator Status (HTTP {status}):")
print(json.dumps(sim_status, indent=2))
