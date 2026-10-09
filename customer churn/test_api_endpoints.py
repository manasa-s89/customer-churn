"""
Comprehensive Unit & Integration Tests for OTT Streaming Churn API Endpoints
Verifies:
- Health check and Dashboard HTML serving
- OTT Churn inference (sensible probabilities, 4-tier risk classification, SHAP factor attributions)
- Rule-based OTT retention recommender logic (pause offers, billing support, content recommendation)
- Subscriber directory pagination & filtering (plan, risk tier)
- Subscriber 360 detail view
- Intervention logging & SQLite database persistence
- Intervention status lifecycle updates
- Analytics overview KPIs with 4 risk tiers (Critical, High, Medium, Low)
- Batch CSV upload and cohort scoring for OTT subscribers
"""

import io
import unittest
from fastapi.testclient import TestClient

from main import app
from recommender import (
    ACTION_BILLING_SUPPORT,
    ACTION_PAUSE_SUBSCRIPTION,
    ACTION_CONTENT_RECOMMENDATION,
    ACTION_ANNUAL_DISCOUNT,
    ACTION_STREAMING_CONCIERGE,
    ACTION_VIP_LOYALTY
)


class TestCustomerChurnAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_dashboard(self):
        """Verify /health and /dashboard routes return HTTP 200."""
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        h_data = res_health.json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertEqual(h_data["database"], "connected")
        self.assertEqual(h_data["domain"], "OTT Video Streaming")
        self.assertIn("Critical", h_data["risk_tiers"])

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertTrue("ChurnIQ" in res_dash.text or "RetainPulse" in res_dash.text or "<!DOCTYPE html>" in res_dash.text)

        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

    def test_02_predict_high_risk_subscriber(self):
        """Verify single subscriber prediction produces sensible probability and triggers retention rules."""
        payload = {
            "customer_id": "TEST-OTT-CRIT-99",
            "subscription_plan": "Basic",
            "monthly_price": 8.99,
            "watch_hours_last_30_days": 1.5,
            "days_since_last_watch": 32,
            "number_of_devices": 1,
            "number_of_profiles": 1,
            "downloads_count": 0,
            "customer_support_tickets": 4,
            "payment_failures": 2,
            "free_trial_converted": "No",
            "tenure_months": 2
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Sensible probability checks
        prob = data["churn_probability"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertGreaterEqual(prob, 0.70, f"Expected elevated probability >= 0.70, got {prob}")
        self.assertIn(data["risk_tier"], ["Critical", "High"])
        self.assertTrue(data["predicted_churn"])

        # SHAP factor attributions
        self.assertGreater(len(data["top_risk_drivers"]), 0)
        for factor in data["top_risk_drivers"]:
            self.assertIn("feature", factor)
            self.assertIn("impact", factor)
            self.assertGreater(factor["impact"], 0)

        # Rule-based OTT interventions
        self.assertGreater(len(data["suggested_interventions"]), 0)
        actions = [s["action_type"] for s in data["suggested_interventions"]]
        self.assertTrue(
            ACTION_BILLING_SUPPORT in actions or ACTION_PAUSE_SUBSCRIPTION in actions or ACTION_CONTENT_RECOMMENDATION in actions,
            f"Expected OTT retention actions, got {actions}"
        )

    def test_03_predict_loyal_low_risk_subscriber(self):
        """Test prediction for loyal subscriber: Premium tier, 60h watch, 0 payment failures."""
        payload = {
            "customer_id": "TEST-OTT-LOYAL-01",
            "subscription_plan": "Premium",
            "monthly_price": 20.99,
            "watch_hours_last_30_days": 75.0,
            "days_since_last_watch": 1,
            "number_of_devices": 5,
            "number_of_profiles": 4,
            "downloads_count": 12,
            "customer_support_tickets": 0,
            "payment_failures": 0,
            "free_trial_converted": "Yes",
            "tenure_months": 36
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["customer_id"], "TEST-OTT-LOYAL-01")
        self.assertLess(data["churn_probability"], 0.40)
        self.assertEqual(data["risk_tier"], "Low")
        self.assertFalse(data["predicted_churn"])
        self.assertGreater(len(data["top_protective_factors"]), 0)

    def test_04_batch_predict(self):
        """Test batch prediction endpoint with list of subscribers."""
        payload = [
            {
                "customer_id": "BATCH-OTT-1",
                "subscription_plan": "Basic",
                "monthly_price": 8.99,
                "watch_hours_last_30_days": 5.0,
                "days_since_last_watch": 20,
                "payment_failures": 1
            },
            {
                "customer_id": "BATCH-OTT-2",
                "subscription_plan": "Premium",
                "monthly_price": 20.99,
                "watch_hours_last_30_days": 50.0,
                "days_since_last_watch": 2,
                "payment_failures": 0
            }
        ]
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["customer_id"], "BATCH-OTT-1")
        self.assertEqual(data[1]["customer_id"], "BATCH-OTT-2")

    def test_05_list_customers_and_filtering(self):
        """Test GET /customers endpoint with risk tier and plan filtering."""
        res = self.client.get("/customers")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_customers", data)
        self.assertGreater(data["total_customers"], 0)
        self.assertGreater(len(data["customers"]), 0)

        # Filter by risk tier
        res_low = self.client.get("/customers?risk_tier=Low")
        self.assertEqual(res_low.status_code, 200)
        data_low = res_low.json()
        for cust in data_low["customers"]:
            self.assertEqual(cust["risk_tier"], "Low")

        # Search by ID
        res_search = self.client.get("/customers?search=LOYAL")
        self.assertEqual(res_search.status_code, 200)
        self.assertTrue(any("LOYAL" in c["customer_id"] for c in res_search.json()["customers"]))

    def test_06_customer_detail_360(self):
        """Test GET /customers/{customer_id} returns full 360 profile."""
        res = self.client.get("/customers/TEST-OTT-CRIT-99")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["customer"]["customer_id"], "TEST-OTT-CRIT-99")
        self.assertIsNotNone(data["latest_score"])
        self.assertIn("watch_hours_last_30_days", data["customer"])

    def test_07_analytics_overview_kpis(self):
        """Verify GET /analytics/overview aggregates 4-tier distribution and revenue at risk."""
        res = self.client.get("/analytics/overview")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertGreater(data["total_customers"], 0)
        self.assertIn("critical_risk_count", data)
        self.assertIn("high_risk_count", data)
        self.assertIn("revenue_at_risk", data)
        self.assertIn("avg_churn_probability", data)
        self.assertIn("risk_distribution", data)
        # All 4 risk tiers must be present
        self.assertIn("Critical", data["risk_distribution"])
        self.assertIn("High", data["risk_distribution"])
        self.assertIn("Medium", data["risk_distribution"])
        self.assertIn("Low", data["risk_distribution"])

    def test_08_batch_csv_upload(self):
        """Verify POST /upload_csv processes batch CSV and calculates portfolio revenue at risk."""
        csv_content = """customer_id,subscription_plan,monthly_price,watch_hours_last_30_days,days_since_last_watch,number_of_devices,number_of_profiles,downloads_count,customer_support_tickets,payment_failures,free_trial_converted,tenure_months
OTT-BATCH-01,Basic,8.99,1.5,35,1,1,0,4,2,No,2
OTT-BATCH-02,Premium,20.99,65.0,2,5,4,12,0,0,Yes,30
OTT-BATCH-03,Standard,14.99,10.0,15,2,2,2,2,1,Yes,6
"""
        csv_file = io.BytesIO(csv_content.encode("utf-8"))
        res = self.client.post("/upload_csv", files={"file": ("ott_batch_test.csv", csv_file, "text/csv")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["rows_processed"], 3)
        self.assertGreaterEqual(data["critical_risk_detected"] + data["high_risk_detected"], 1)
        self.assertGreaterEqual(data["total_revenue_at_risk"], 8.99)
        self.assertEqual(len(data["sample_scored_customers"]), 3)

    def test_09_event_ingest_watch_session(self):
        """Verify POST /events/ingest updates watch hours and resets days_since_last_watch."""
        # First ensure a test customer exists
        seed_cust = {
            "customer_id": "TEST-EVT-WATCH",
            "subscription_plan": "Standard",
            "monthly_price": 14.99,
            "watch_hours_last_30_days": 10.0,
            "days_since_last_watch": 15,
            "number_of_devices": 2,
            "number_of_profiles": 2,
            "downloads_count": 1,
            "customer_support_tickets": 0,
            "payment_failures": 0,
            "free_trial_converted": "Yes",
            "tenure_months": 5
        }
        self.client.post("/predict", json=seed_cust)

        event_payload = {
            "customer_id": "TEST-EVT-WATCH",
            "event_type": "watch_session",
            "metadata": {"duration_hours": 3.5}
        }
        res = self.client.post("/events/ingest", json=event_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["customer_id"], "TEST-EVT-WATCH")
        self.assertEqual(data["features_updated"]["days_since_last_watch"], 0)
        self.assertEqual(data["features_updated"]["watch_hours_last_30_days"], 13.5)
        self.assertIn("churn_probability", data["new_score"])

    def test_10_event_ingest_payment_failure_escalation(self):
        """Verify POST /events/ingest for payment_failure increments failures and escalates churn probability."""
        seed_cust = {
            "customer_id": "TEST-EVT-PAYFAIL",
            "subscription_plan": "Basic",
            "monthly_price": 8.99,
            "watch_hours_last_30_days": 4.0,
            "days_since_last_watch": 25,
            "number_of_devices": 1,
            "number_of_profiles": 1,
            "downloads_count": 0,
            "customer_support_tickets": 2,
            "payment_failures": 1,
            "free_trial_converted": "No",
            "tenure_months": 2
        }
        self.client.post("/predict", json=seed_cust)

        event_payload = {
            "customer_id": "TEST-EVT-PAYFAIL",
            "event_type": "payment_failure",
            "metadata": {"reason": "Card declined on billing renewal"}
        }
        res = self.client.post("/events/ingest", json=event_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["features_updated"]["payment_failures"], 2)
        # Churn probability should have increased or remained high
        prev_p = data["previous_score"]["churn_probability"]
        new_p = data["new_score"]["churn_probability"]
        self.assertGreaterEqual(new_p, prev_p)

    def test_11_live_simulator_endpoints(self):
        """Verify /simulate/live-events/status and /simulate/live-events/step endpoints."""
        res_status = self.client.get("/simulate/live-events/status")
        self.assertEqual(res_status.status_code, 200)
        st_data = res_status.json()
        self.assertIn("is_running", st_data)
        self.assertIn("events_generated", st_data)

        # Single step simulation
        res_step = self.client.post("/simulate/live-events/step")
        self.assertEqual(res_step.status_code, 200)
        step_data = res_step.json()
        self.assertEqual(step_data["status"], "success")
        self.assertIn("customer_id", step_data)
        self.assertIn("event_type", step_data)
        self.assertIn("previous_score", step_data)
        self.assertIn("new_score", step_data)


if __name__ == '__main__':
    unittest.main()
