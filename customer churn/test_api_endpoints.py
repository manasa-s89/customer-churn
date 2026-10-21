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

        res_dash_slash = self.client.get("/dashboard/")
        self.assertEqual(res_dash_slash.status_code, 200)

        res_api_dash = self.client.get("/api/dashboard")
        self.assertEqual(res_api_dash.status_code, 200)

        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers.get("content-type", ""))

        # Test static CSS returns real CSS with text/css (not HTML dashboard)
        res_css = self.client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("text/css", res_css.headers.get("content-type", ""))
        self.assertNotIn("<!DOCTYPE html>", res_css.text)

        # Test static JS returns real JS with javascript media type (not HTML dashboard)
        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertTrue(
            "text/javascript" in res_js.headers.get("content-type", "")
            or "application/javascript" in res_js.headers.get("content-type", "")
        )
        self.assertNotIn("<!DOCTYPE html>", res_js.text)

        # Test simulated Vercel proxy rewrite preserving original path
        res_v_css = self.client.get("/api/index.py", headers={"x-matched-path": "/static/styles.css"})
        self.assertEqual(res_v_css.status_code, 200)
        self.assertIn("text/css", res_v_css.headers.get("content-type", ""))
        self.assertNotIn("<!DOCTYPE html>", res_v_css.text)

        # Test missing static file returns 404, never fallback to index.html
        res_404 = self.client.get("/static/non_existent_asset.css")
        self.assertEqual(res_404.status_code, 404)
        self.assertNotIn("<!DOCTYPE html>", res_404.text)

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

    def test_12_predict_under_vercel_simulation(self):
        """Simulate Vercel environment (VERCEL=1) and verify POST /predict with exact dashboard payload."""
        import os
        import traceback

        old_vercel = os.environ.get("VERCEL")
        os.environ["VERCEL"] = "1"
        try:
            payload = {
                "customer_id": "OTT-SCENARIO-01",
                "subscription_plan": "Standard",
                "monthly_price": 14.99,
                "tenure_months": 10,
                "free_trial_converted": "Yes",
                "watch_hours_last_30_days": 26.0,
                "days_since_last_watch": 4,
                "customer_support_tickets": 0,
                "payment_failures": 0,
                "number_of_devices": 2,
                "number_of_profiles": 2,
                "downloads_count": 4
            }
            try:
                res = self.client.post("/predict", json=payload)
                if res.status_code != 200:
                    print(f"\n[!] Vercel /predict failed with status {res.status_code}:\n{res.text}")
            except Exception:
                traceback.print_exc()
                raise

            self.assertEqual(res.status_code, 200, f"Expected 200, got {res.status_code}: {res.text}")
            data = res.json()
            self.assertEqual(data["customer_id"], "OTT-SCENARIO-01")
            self.assertIn("churn_probability", data)
            self.assertIn("risk_tier", data)
            self.assertIn("top_risk_drivers", data)
            self.assertIn("top_protective_factors", data)
            self.assertIn("suggested_interventions", data)
        finally:
            if old_vercel is None:
                os.environ.pop("VERCEL", None)
            else:
                os.environ["VERCEL"] = old_vercel

    def test_13_predict_db_failure_resilience(self):
        """Verify /predict does not crash even if database persistence fails."""
        from unittest.mock import patch

        payload = {
            "customer_id": "OTT-SCENARIO-01",
            "subscription_plan": "Standard",
            "monthly_price": 14.99,
            "tenure_months": 10,
            "free_trial_converted": "Yes",
            "watch_hours_last_30_days": 26.0,
            "days_since_last_watch": 4,
            "customer_support_tickets": 0,
            "payment_failures": 0,
            "number_of_devices": 2,
            "number_of_profiles": 2,
            "downloads_count": 4
        }
        with patch("sqlalchemy.orm.Session.commit", side_effect=Exception("Read-only filesystem: commit failed")):
            res = self.client.post("/predict", json=payload)
            self.assertEqual(res.status_code, 200, f"Expected 200 on DB failure, got {res.status_code}: {res.text}")
            data = res.json()
            self.assertEqual(data["customer_id"], "OTT-SCENARIO-01")
            self.assertIn("churn_probability", data)


    def test_14_all_frontend_endpoints_under_vercel_simulation(self):
        """
        Verify all API endpoints called by static/app.js succeed under VERCEL=1 simulation:
        health, analytics/overview, customers, predict, upload_csv, interventions,
        simulate/live-events, and events/ingest.
        """
        import os
        old_vercel = os.environ.get("VERCEL")
        os.environ["VERCEL"] = "1"
        try:
            # 1. GET /health
            res = self.client.get("/health")
            self.assertEqual(res.status_code, 200)

            # 2. GET /analytics/overview
            res = self.client.get("/analytics/overview")
            self.assertEqual(res.status_code, 200)

            # 3. GET /customers
            res = self.client.get("/customers?limit=5")
            self.assertEqual(res.status_code, 200)

            # 4. GET /customers/{customer_id}
            res = self.client.get("/customers/TEST-OTT-CRIT-99")
            self.assertEqual(res.status_code, 200)

            # 5. POST /predict
            payload = {
                "customer_id": "VERCEL-SUITE-01",
                "subscription_plan": "Standard",
                "monthly_price": 14.99,
                "watch_hours_last_30_days": 18.0,
                "days_since_last_watch": 6,
                "number_of_devices": 2,
                "number_of_profiles": 2,
                "downloads_count": 2,
                "customer_support_tickets": 1,
                "payment_failures": 0,
                "free_trial_converted": "Yes",
                "tenure_months": 8
            }
            res = self.client.post("/predict", json=payload)
            self.assertEqual(res.status_code, 200)

            # 6. POST /upload_csv
            csv_content = "customer_id,subscription_plan,monthly_price,watch_hours_last_30_days,days_since_last_watch,number_of_devices,number_of_profiles,downloads_count,customer_support_tickets,payment_failures,free_trial_converted,tenure_months\nUP-V-01,Basic,8.99,10,2,1,1,0,0,0,Yes,5\n"
            csv_file = io.BytesIO(csv_content.encode("utf-8"))
            res = self.client.post("/upload_csv", files={"file": ("test_vercel.csv", csv_file, "text/csv")})
            self.assertEqual(res.status_code, 200)

            # 7. GET /interventions
            res = self.client.get("/interventions")
            self.assertEqual(res.status_code, 200)

            # 8. PUT /interventions/{id}
            ints = res.json()
            if ints:
                int_id = ints[0]["id"]
                res_upd = self.client.put(f"/interventions/{int_id}", json={"status": "Resolved", "notes": "Vercel test verified"})
                self.assertEqual(res_upd.status_code, 200)

            # 9. POST /intervene
            int_payload = {
                "customer_id": "TEST-OTT-CRIT-99",
                "action_type": "pause_subscription_offer",
                "title": "Pause Subscription Offer",
                "description": "Offering temporary pause under Vercel test suite.",
                "urgency": "High",
                "recommended_channel": "email",
                "status": "Sent"
            }
            res = self.client.post("/intervene", json=int_payload)
            self.assertIn(res.status_code, [200, 201])

            # 10. GET /simulate/live-events/status
            res = self.client.get("/simulate/live-events/status")
            self.assertEqual(res.status_code, 200)

            # 11. POST /simulate/live-events/start
            res = self.client.post("/simulate/live-events/start")
            self.assertEqual(res.status_code, 200)

            # 12. POST /simulate/live-events/stop
            res = self.client.post("/simulate/live-events/stop")
            self.assertEqual(res.status_code, 200)

            # 13. POST /simulate/live-events/step
            res = self.client.post("/simulate/live-events/step")
            self.assertEqual(res.status_code, 200)

            # 14. POST /events/ingest
            evt_payload = {
                "customer_id": "TEST-OTT-CRIT-99",
                "event_type": "watch_session",
                "metadata": {"duration_hours": 1.5}
            }
            res = self.client.post("/events/ingest", json=evt_payload)
            self.assertEqual(res.status_code, 200)

            # 15. GET / and GET /dashboard
            res_root = self.client.get("/")
            self.assertEqual(res_root.status_code, 200)
            res_dash = self.client.get("/dashboard")
            self.assertEqual(res_dash.status_code, 200)

        finally:
            if old_vercel is None:
                os.environ.pop("VERCEL", None)
            else:
                os.environ["VERCEL"] = old_vercel

    def test_15_docs_redoc_openapi_not_caught_by_dashboard_fallback(self):
        """
        Verify /docs, /redoc, and /openapi.json return proper API documentation and schemas
        and are never caught by the dashboard HTML fallback.
        """
        # Direct routes
        res_docs = self.client.get("/docs")
        self.assertEqual(res_docs.status_code, 200)
        self.assertIn("swagger-ui", res_docs.text.lower())
        self.assertNotIn("sidebar-brand-text", res_docs.text)

        res_redoc = self.client.get("/redoc")
        self.assertEqual(res_redoc.status_code, 200)
        self.assertIn("redoc", res_redoc.text.lower())
        self.assertNotIn("sidebar-brand-text", res_redoc.text)

        res_openapi = self.client.get("/openapi.json")
        self.assertEqual(res_openapi.status_code, 200)
        self.assertIn("application/json", res_openapi.headers.get("content-type", ""))
        openapi_data = res_openapi.json()
        self.assertIn("openapi", openapi_data)
        self.assertIn("paths", openapi_data)
        self.assertIn("/predict", openapi_data["paths"])

        # Simulated Vercel rewrites to api/index.py with x-matched-path
        res_v_docs = self.client.get("/api/index.py", headers={"x-matched-path": "/docs"})
        self.assertEqual(res_v_docs.status_code, 200)
        self.assertIn("swagger-ui", res_v_docs.text.lower())
        self.assertNotIn("sidebar-brand-text", res_v_docs.text)

        res_v_redoc = self.client.get("/api/index.py", headers={"x-matched-path": "/redoc"})
        self.assertEqual(res_v_redoc.status_code, 200)
        self.assertIn("redoc", res_v_redoc.text.lower())
        self.assertNotIn("sidebar-brand-text", res_v_redoc.text)

        res_v_openapi = self.client.get("/api/index.py", headers={"x-matched-path": "/openapi.json"})
        self.assertEqual(res_v_openapi.status_code, 200)
        self.assertIn("application/json", res_v_openapi.headers.get("content-type", ""))
        self.assertIn("openapi", res_v_openapi.json())

        # /api/docs and /api/redoc aliases
        res_api_docs = self.client.get("/api/docs")
        self.assertEqual(res_api_docs.status_code, 200)
        self.assertIn("swagger-ui", res_api_docs.text.lower())

        res_api_redoc = self.client.get("/api/redoc")
        self.assertEqual(res_api_redoc.status_code, 200)
        self.assertIn("redoc", res_api_redoc.text.lower())

        res_api_openapi = self.client.get("/api/openapi.json")
        self.assertEqual(res_api_openapi.status_code, 200)
        self.assertIn("openapi", res_api_openapi.json())

    def test_16_clear_json_error_responses(self):
        """Verify API errors return clear, structured JSON containing error, detail, and message."""
        # 1. 404 Not Found error
        res_404 = self.client.get("/customers/NON-EXISTENT-OTT-USER-999")
        self.assertEqual(res_404.status_code, 404)
        self.assertIn("application/json", res_404.headers.get("content-type", ""))
        err_404 = res_404.json()
        self.assertIn("detail", err_404)
        self.assertIn("message", err_404)
        self.assertTrue("not found" in err_404["message"].lower() or "not found" in str(err_404["detail"]).lower())

        # 2. 400 Bad Request error (invalid file upload)
        bad_file = io.BytesIO(b"random plain text")
        res_bad = self.client.post("/upload_csv", files={"file": ("malicious.exe", bad_file, "application/octet-stream")})
        self.assertEqual(res_bad.status_code, 400)
        self.assertIn("application/json", res_bad.headers.get("content-type", ""))
        err_bad = res_bad.json()
        self.assertIn("detail", err_bad)
        self.assertIn("message", err_bad)
        self.assertIn(".csv file", err_bad["message"])

        # 3. 422 Validation Error (malformed body)
        res_422 = self.client.post("/events/ingest", json={"invalid_field": 123})
        self.assertEqual(res_422.status_code, 422)
        self.assertIn("application/json", res_422.headers.get("content-type", ""))
        err_422 = res_422.json()
        self.assertIn("detail", err_422)
        self.assertIn("message", err_422)
        self.assertEqual(err_422["error"], "Validation Error")

    def test_17_database_postgres_and_seed_deduplication(self):
        """Verify database initialization loads seed data when DB is empty and avoids duplicate rows on re-runs."""
        from database import _check_and_seed_database, SessionLocal
        import models_db
        from sqlalchemy import func

        with SessionLocal() as db:
            count_before = db.query(func.count(models_db.CustomerRecord.customer_id)).scalar() or 0
            self.assertGreater(count_before, 0, "Database should have subscribers loaded.")

            # Simulate cold start 2: run _check_and_seed_database again
            _check_and_seed_database()

            count_after = db.query(func.count(models_db.CustomerRecord.customer_id)).scalar() or 0
            self.assertEqual(count_after, count_before, "Subsequent cold starts must not duplicate rows in database.")


if __name__ == '__main__':
    unittest.main()

