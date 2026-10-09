"""
Configuration for OTT Streaming Churn Intelligence Platform.
Centralizes risk tier thresholds, color palettes, and domain constants.
"""

from typing import Dict, Any, List

# 4 Risk Tiers: Critical (≥80%), High (60–79%), Moderate (30–59%), Low (<30%)
RISK_TIER_CONFIG = {
    "Critical": {
        "min_prob": 0.80,
        "max_prob": 1.00,
        "color": "#ef4444",       # Crimson Red
        "badge_class": "badge-critical",
        "label": "Critical Risk",
        "description": "Immediate churn imminent (≥80% probability)"
    },
    "High": {
        "min_prob": 0.60,
        "max_prob": 0.799999,
        "color": "#f97316",       # Vibrant Orange
        "badge_class": "badge-high",
        "label": "High Risk",
        "description": "High churn risk (60–79% probability)"
    },
    "Moderate": {
        "min_prob": 0.30,
        "max_prob": 0.599999,
        "color": "#38bdf8",       # Electric Blue
        "badge_class": "badge-moderate",
        "label": "Moderate Risk",
        "description": "Moderate churn risk (30–59% probability)"
    },
    # Backward-compatible alias
    "Medium": {
        "min_prob": 0.30,
        "max_prob": 0.599999,
        "color": "#38bdf8",
        "badge_class": "badge-moderate",
        "label": "Moderate Risk",
        "description": "Moderate churn risk (30–59% probability)"
    },
    "Low": {
        "min_prob": 0.00,
        "max_prob": 0.299999,
        "color": "#10b981",       # Emerald Green
        "badge_class": "badge-low",
        "label": "Low Risk",
        "description": "Healthy subscriber (<30% probability)"
    }
}

RISK_TIER_ORDER = ["Critical", "High", "Moderate", "Low"]

TIER_WEIGHTS = {
    "Critical": 4,
    "High": 3,
    "Moderate": 2,
    "Medium": 2,
    "Low": 1
}


def calculate_risk_tier(probability: float) -> str:
    """
    Standardized classification into 4 risk tiers:
    - Critical: >= 0.80 (>= 80%)
    - High: 0.60 <= prob < 0.80 (60% - 79%)
    - Moderate: 0.30 <= prob < 0.60 (30% - 59%)
    - Low: < 0.30 (< 30%)
    """
    prob = float(probability)
    if prob >= 0.80:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Moderate"
    else:
        return "Low"


# OTT Subscription Plans
SUBSCRIPTION_PLANS = ["Basic", "Standard", "Premium"]

# OTT Default Plan Pricing
PLAN_PRICING = {
    "Basic": 8.99,
    "Standard": 14.99,
    "Premium": 20.99
}
