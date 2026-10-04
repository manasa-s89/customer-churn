"""
Configuration for OTT Streaming Churn Intelligence Platform.
Centralizes risk tier thresholds, color palettes, and domain constants.
"""

from typing import Dict, Any, List

# 4 Risk Tiers: Critical (≥90%), High (70–89%), Medium (40–69%), Low (<40%)
RISK_TIER_CONFIG = {
    "Critical": {
        "min_prob": 0.90,
        "max_prob": 1.00,
        "color": "#ef4444",       # Crimson Red
        "badge_class": "badge-critical",
        "label": "Critical Risk",
        "description": "Immediate churn imminent (≥90% probability)"
    },
    "High": {
        "min_prob": 0.70,
        "max_prob": 0.899999,
        "color": "#f97316",       # Vibrant Orange
        "badge_class": "badge-high",
        "label": "High Risk",
        "description": "High churn risk (70–89% probability)"
    },
    "Medium": {
        "min_prob": 0.40,
        "max_prob": 0.699999,
        "color": "#eab308",       # Amber / Yellow
        "badge_class": "badge-medium",
        "label": "Medium Risk",
        "description": "Moderate churn risk (40–69% probability)"
    },
    "Low": {
        "min_prob": 0.00,
        "max_prob": 0.399999,
        "color": "#10b981",       # Emerald Green
        "badge_class": "badge-low",
        "label": "Low Risk",
        "description": "Healthy subscriber (<40% probability)"
    }
}

RISK_TIER_ORDER = ["Critical", "High", "Medium", "Low"]

TIER_WEIGHTS = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}


def calculate_risk_tier(probability: float) -> str:
    """
    Standardized classification into 4 risk tiers:
    - Critical: >= 0.90 (>= 90%)
    - High: 0.70 <= prob < 0.90 (70% - 89%)
    - Medium: 0.40 <= prob < 0.70 (40% - 69%)
    - Low: < 0.40 (< 40%)
    """
    prob = float(probability)
    if prob >= 0.90:
        return "Critical"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
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
