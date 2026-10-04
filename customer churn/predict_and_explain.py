"""
OTT Subscriber Churn Prediction and Explainability (SHAP / Feature Contribution)
Takes OTT streaming subscriber data, predicts churn probability, breaks down risk drivers,
assigns the 4-tier risk classification, and links to retention recommendations.
"""

import json
import os
import sys
from typing import Dict, List, Any, Union, Tuple

import joblib
import numpy as np
import pandas as pd

from config import calculate_risk_tier, RISK_TIER_CONFIG
from data_pipeline import clean_raw_data, get_feature_names

HAS_SHAP = None


class ChurnExplainer:
    """Explainer for individual OTT subscriber churn predictions."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model or Preprocessor not found in '{models_dir}'. Please run train.py first."
            )

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.feature_names = get_feature_names(self.preprocessor)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP explainer if available and model is tree-based."""
        global HAS_SHAP
        if hasattr(self.model, "feature_importances_"):
            try:
                import shap
                HAS_SHAP = True
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                HAS_SHAP = False
                print(f"[!] Note on SHAP init: {e}. Falling back to feature contributions.")
        else:
            HAS_SHAP = False

    def predict_and_explain(
        self,
        customer_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        top_k_reasons: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Predict churn probability and return human-interpretable OTT risk factors.
        """
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        elif isinstance(customer_data, list):
            df = pd.DataFrame(customer_data)
        else:
            df = customer_data.copy()

        # Clean
        df = clean_raw_data(df)
        customer_ids = df["customer_id"].tolist() if "customer_id" in df.columns else [f"OTT-{i+1}" for i in range(len(df))]

        # Transform features
        X_proc = self.preprocessor.transform(df)

        # Probabilities
        probs = self.model.predict_proba(X_proc)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = []

        # Calculate contributions
        shap_values_matrix = None
        if self.explainer is not None:
            try:
                raw_shap = self.explainer.shap_values(X_proc)
                if isinstance(raw_shap, list):
                    shap_values_matrix = raw_shap[1]
                elif len(raw_shap.shape) == 3:
                    shap_values_matrix = raw_shap[:, :, 1]
                else:
                    shap_values_matrix = raw_shap
            except Exception:
                shap_values_matrix = None

        for idx in range(len(df)):
            cust_id = customer_ids[idx]
            churn_prob = float(probs[idx])
            # 4-tier risk classification
            churn_risk = calculate_risk_tier(churn_prob)
            will_churn = bool(preds[idx])

            # Decompose feature contributions
            if shap_values_matrix is not None:
                contribs = shap_values_matrix[idx]
            elif hasattr(self.model, "coef_"):
                # Logistic regression linear contribution: X_proc[idx] * coef
                contribs = X_proc[idx] * self.model.coef_[0]
            elif hasattr(self.model, "feature_importances_"):
                # Approximate direction using deviation from mean feature value
                contribs = (X_proc[idx] > 0).astype(float) * self.model.feature_importances_
            else:
                contribs = np.zeros(len(self.feature_names))

            # Pair with human readable feature names
            feature_impacts = []
            for feat_name, imp, val in zip(self.feature_names, contribs, X_proc[idx]):
                feature_impacts.append({
                    "feature": feat_name,
                    "impact": float(imp),
                    "transformed_value": float(val)
                })

            # Sort by impact
            risk_drivers = sorted([f for f in feature_impacts if f["impact"] > 0], key=lambda x: x["impact"], reverse=True)[:top_k_reasons]
            protective_factors = sorted([f for f in feature_impacts if f["impact"] < 0], key=lambda x: x["impact"])[:top_k_reasons]

            # Generate natural language summary and retention action
            explanation_summary, recommended_action = self._generate_retention_insight(
                churn_prob, churn_risk, df.iloc[idx].to_dict(), risk_drivers
            )

            results.append({
                "customer_id": cust_id,
                "churn_probability": round(churn_prob, 4),
                "churn_probability_pct": f"{churn_prob*100:.1f}%",
                "risk_tier": churn_risk,
                "predicted_churn": will_churn,
                "top_risk_drivers": risk_drivers,
                "top_protective_factors": protective_factors,
                "explanation_summary": explanation_summary,
                "recommended_action": recommended_action,
                "raw_features": df.iloc[idx].to_dict()
            })

        return results

    def _generate_retention_insight(
        self,
        prob: float,
        risk_tier: str,
        cust_row: Dict[str, Any],
        risk_drivers: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generate human-readable summary and tailored OTT retention recommendations."""
        tenure = cust_row.get('tenure_months', 0)
        plan = cust_row.get('subscription_plan', 'Basic')
        watch_hours = cust_row.get('watch_hours_last_30_days', 0.0)
        days_inactive = cust_row.get('days_since_last_watch', 0)
        payment_failures = cust_row.get('payment_failures', 0)
        tickets = cust_row.get('customer_support_tickets', 0)

        # Contextual summary
        summary = (
            f"Subscriber has an estimated {prob*100:.1f}% churn risk ({risk_tier} Risk). "
            f"Enrolled in {plan} tier for {tenure} months with {watch_hours:.1f} viewing hours "
            f"and {days_inactive} days since last stream."
        )

        # Contextual recommendation
        if payment_failures >= 1:
            rec = "CRITICAL ACTION: Send automated payment grace recovery email and prompt in-app billing update."
        elif risk_tier == "Critical":
            rec = "IMMEDIATE RETENTION: Present in-app Pause Subscription modal (1–3 months free hold) before cancellation."
        elif days_inactive >= 14 or watch_hours < 8:
            rec = "ENGAGEMENT OUTREACH: Trigger personalized trending watchlist email highlighting recommended new series."
        elif plan == "Basic" and prob >= 0.50:
            rec = "UPGRADE PLAYBOOK: Offer 25% discount to switch to annual Standard commitment."
        elif tickets >= 2:
            rec = "SUPPORT FOLLOW-UP: Dispatch streaming playback optimization guide to resolve buffering friction."
        else:
            rec = "ADVOCACY: Subscriber is engaged; offer VIP early premiere screening access and referral bonus."

        return summary, rec
