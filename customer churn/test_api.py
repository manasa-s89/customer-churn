"""
Comprehensive test suite for Customer Churn FastAPI Backend:
1. Health check (/health and /)
2. Single Customer Prediction (/predict) with probability, risk tier, contributing factors, rule-based interventions
3. Batch Customer Prediction (/predict)
4. List Customers with scores (/customers) and filtering by risk tier
5. Customer 360 Detail View (/customers/{id})
6. Log Intervention (/intervene) with discount offer, proactive outreach, contract upgrade, etc.
7. List Historical Interventions (/interventions/{id})
8. Dynamic Rule-Based Recommendations (/recommendations/{id})
"""

import sys
import unittest
from datetime import datetime
from fastapi.testclient import TestClient

# Ensure local imports work
import database
import models_db
from main import app
from recommender import (
    recommend_interventions,
    ACTION_PROACTIVE_OUTREACH,
    ACTION_CONTRACT_UPGRADE,
    ACTION_REENGAGEMENT,
    ACTION_DISCOUNT_OFFER,
)


class TestChurnBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_and_health(self):
        """Test root overview and health endpoints."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("predict", data["endpoints"])

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")
        self.assertEqual(res_health.json()["database"], "connected")

    def test_02_predict_high_risk_customer(self):
        """Test prediction for high risk customer: Month-to-month, high tickets, high charges."""
        payload = {
            "customer_id": "TEST-RISK-HIGH-01",
            "tenure": 2,
            "monthly_charges": 95.50,
            "total_charges": 191.00,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "support_tickets": 4,
            "last_login_days_ago": 35,
            "usage_frequency": "Rarely"
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Check response structure
        self.assertEqual(data["customer_id"], "TEST-RISK-HIGH-01")
        self.assertGreaterEqual(data["churn_probability"], 0.5)
        self.assertEqual(data["risk_tier"], "High")
        self.assertTrue(data["predicted_churn"])

        # Check contributing factors
        self.assertGreater(len(data["top_risk_drivers"]), 0)
        driver_features = [d["feature"].lower() for d in data["top_risk_drivers"]]
        self.assertTrue(
            any("contract" in f or "ticket" in f or "charge" in f or "login" in f for f in driver_features),
            f"Expected key drivers in {driver_features}"
        )

        # Check rule-based interventions
        self.assertGreater(len(data["suggested_interventions"]), 0)
        action_types = [s["action_type"] for s in data["suggested_interventions"]]
        # High tickets should trigger proactive outreach
        self.assertIn(ACTION_PROACTIVE_OUTREACH, action_types)
        # Month-to-month should trigger contract upgrade
        self.assertIn(ACTION_CONTRACT_UPGRADE, action_types)

    def test_03_predict_loyal_low_risk_customer(self):
        """Test prediction for loyal customer: 60 month tenure, Two year contract, 0 tickets."""
        payload = {
            "customer_id": "TEST-LOYAL-LOW-01",
            "tenure": 60,
            "monthly_charges": 42.00,
            "total_charges": 2520.00,
            "contract_type": "Two year",
            "payment_method": "Bank transfer (automatic)",
            "internet_service": "DSL",
            "support_tickets": 0,
            "last_login_days_ago": 2,
            "usage_frequency": "Daily"
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["customer_id"], "TEST-LOYAL-LOW-01")
        self.assertLess(data["churn_probability"], 0.40)
        self.assertEqual(data["risk_tier"], "Low")
        self.assertFalse(data["predicted_churn"])
        self.assertGreater(len(data["top_protective_factors"]), 0)

    def test_04_batch_predict(self):
        """Test batch prediction endpoint with list of customers."""
        payload = [
            {
                "customer_id": "BATCH-CUST-1",
                "tenure": 5,
                "monthly_charges": 80.0,
                "contract_type": "Month-to-month",
                "support_tickets": 3
            },
            {
                "customer_id": "BATCH-CUST-2",
                "tenure": 48,
                "monthly_charges": 35.0,
                "contract_type": "Two year",
                "support_tickets": 0
            }
        ]
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["customer_id"], "BATCH-CUST-1")
        self.assertEqual(data[1]["customer_id"], "BATCH-CUST-2")

    def test_05_list_customers_and_filtering(self):
        """Test GET /customers endpoint with risk tier filtering."""
        res = self.client.get("/customers")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_customers", data)
        self.assertGreater(data["total_customers"], 0)
        self.assertGreater(len(data["customers"]), 0)

        # Filter by High risk
        res_high = self.client.get("/customers?risk_tier=High")
        self.assertEqual(res_high.status_code, 200)
        data_high = res_high.json()
        for cust in data_high["customers"]:
            self.assertEqual(cust["risk_tier"], "High")

        # Search by ID
        res_search = self.client.get("/customers?search=LOYAL")
        self.assertEqual(res_search.status_code, 200)
        self.assertTrue(any("LOYAL" in c["customer_id"] for c in res_search.json()["customers"]))

    def test_06_customer_detail_360(self):
        """Test GET /customers/{customer_id}."""
        res = self.client.get("/customers/TEST-RISK-HIGH-01")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["customer"]["customer_id"], "TEST-RISK-HIGH-01")
        self.assertIsNotNone(data["latest_score"])
        self.assertIn("current_recommendations", data)

    def test_07_log_intervention(self):
        """Test POST /intervene endpoint to log an action."""
        payload = {
            "customer_id": "TEST-RISK-HIGH-01",
            "action_type": "proactive_outreach",
            "details": "Conducted VIP technical call with customer. Resolved fiber modem latency and granted $15 bill credit.",
            "channel": "phone",
            "status": "executed",
            "notes": "Customer accepted credit and agreed to test connection over 48 hours.",
            "performed_by": "Retention Lead Sarah"
        }
        res = self.client.post("/intervene", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["customer_id"], "TEST-RISK-HIGH-01")
        self.assertEqual(data["action_type"], "proactive_outreach")
        self.assertEqual(data["channel"], "phone")
        self.assertIsNotNone(data["id"])

        # Log a second intervention: contract upgrade incentive
        payload2 = {
            "customer_id": "TEST-RISK-HIGH-01",
            "action_type": "contract_upgrade_incentive",
            "details": "Sent promotional offer for 1-year contract extension with 15% discount.",
            "channel": "email",
            "status": "executed",
            "performed_by": "Automated Retention Workflow"
        }
        res2 = self.client.post("/intervene", json=payload2)
        self.assertEqual(res2.status_code, 201)

    def test_08_list_customer_interventions(self):
        """Test GET /interventions/{customer_id}."""
        res = self.client.get("/interventions/TEST-RISK-HIGH-01")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data), 2)
        actions = [i["action_type"] for i in data]
        self.assertIn("proactive_outreach", actions)
        self.assertIn("contract_upgrade_incentive", actions)

    def test_09_get_recommendations(self):
        """Test GET /recommendations/{customer_id}."""
        res = self.client.get("/recommendations/TEST-RISK-HIGH-01")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["customer_id"], "TEST-RISK-HIGH-01")
        self.assertIn("recommendations", data)
        self.assertIn("primary_action", data["recommendations"])
        self.assertIn("suggested_actions", data["recommendations"])

    def test_10_rule_engine_mappings(self):
        """Directly verify rule engine mappings for all key risk drivers."""
        # 1. High tickets -> Proactive Outreach
        rec1 = recommend_interventions(
            customer_features={"support_tickets": 4, "contract_type": "One year"},
            churn_probability=0.70,
            risk_tier="High"
        )
        actions1 = [a["action_type"] for a in rec1["suggested_actions"]]
        self.assertIn(ACTION_PROACTIVE_OUTREACH, actions1)

        # 2. Month-to-month -> Contract Upgrade Incentive
        rec2 = recommend_interventions(
            customer_features={"contract_type": "Month-to-month", "support_tickets": 0},
            churn_probability=0.60,
            risk_tier="High"
        )
        actions2 = [a["action_type"] for a in rec2["suggested_actions"]]
        self.assertIn(ACTION_CONTRACT_UPGRADE, actions2)

        # 3. Dormancy / Low Usage -> Re-engagement Email
        rec3 = recommend_interventions(
            customer_features={"usage_frequency": "Rarely", "last_login_days_ago": 35},
            churn_probability=0.55,
            risk_tier="Medium"
        )
        actions3 = [a["action_type"] for a in rec3["suggested_actions"]]
        self.assertIn(ACTION_REENGAGEMENT, actions3)

        # 4. High Charges -> Discount Offer
        rec4 = recommend_interventions(
            customer_features={"monthly_charges": 98.50},
            churn_probability=0.60,
            risk_tier="High"
        )
        actions4 = [a["action_type"] for a in rec4["suggested_actions"]]
        self.assertIn(ACTION_DISCOUNT_OFFER, actions4)


if __name__ == "__main__":
    unittest.main()
