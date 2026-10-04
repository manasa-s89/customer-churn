"""
Rule-Based Retention Recommender for OTT Streaming Subscribers
Maps streaming telemetry and ML risk drivers to targeted, high-impact subscriber retention interventions:
- Low watch hours / dormancy   -> Personalized Content Recommendation Email / Push
- Payment failures / declines  -> Proactive Billing Support & Payment Method Grace Recovery
- Critical risk / cancel signal -> Pause Subscription Offer (1–3 months pause vs cancellation)
- Price-sensitive Basic plan   -> Annual Plan Switch Loyalty Discount (25% off annual lock-in)
- High support tickets         -> Streaming Quality & Playback Concierge Outreach
- Loyal / Low-risk users       -> VIP Early Access & Screening Pass
"""

from typing import Dict, List, Any, Optional

# Action type constants
ACTION_CONTENT_RECOMMENDATION = "content_recommendation_email"
ACTION_BILLING_SUPPORT = "proactive_billing_support"
ACTION_PAUSE_SUBSCRIPTION = "pause_subscription_offer"
ACTION_ANNUAL_DISCOUNT = "annual_plan_switch_discount"
ACTION_STREAMING_CONCIERGE = "streaming_quality_concierge"
ACTION_VIP_LOYALTY = "vip_early_access_screening"

# Backward-compatibility aliases for tests/older references
ACTION_PROACTIVE_OUTREACH = ACTION_BILLING_SUPPORT
ACTION_CONTRACT_UPGRADE = ACTION_ANNUAL_DISCOUNT
ACTION_REENGAGEMENT = ACTION_CONTENT_RECOMMENDATION
ACTION_DISCOUNT_OFFER = ACTION_PAUSE_SUBSCRIPTION
ACTION_ONBOARDING = ACTION_STREAMING_CONCIERGE


def recommend_interventions(
    customer_features: Dict[str, Any],
    churn_probability: float,
    risk_tier: str,
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates OTT streaming subscriber telemetry and ML risk drivers against rule-based criteria.

    Returns:
        Dict containing:
        - primary_action: The highest priority suggested action
        - suggested_actions: Ordered list of applicable intervention actions
        - retention_strategy_summary: Comprehensive natural language recommendation
    """
    if top_risk_drivers is None:
        top_risk_drivers = []

    driver_features = [str(d.get("feature", "")).lower() for d in top_risk_drivers]

    # Normalize subscriber inputs
    plan = str(customer_features.get("subscription_plan", "Basic")).strip().capitalize()
    monthly_price = float(customer_features.get("monthly_price", 8.99) or 8.99)
    watch_hours = float(customer_features.get("watch_hours_last_30_days", 0.0) or 0.0)
    days_since_last_watch = int(customer_features.get("days_since_last_watch", 0) or 0)
    tickets = int(customer_features.get("customer_support_tickets", 0) or 0)
    payment_failures = int(customer_features.get("payment_failures", 0) or 0)
    free_trial = str(customer_features.get("free_trial_converted", "Yes")).strip().capitalize()
    tenure_months = int(customer_features.get("tenure_months", 1) or 1)

    matched_actions: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # RULE 1: Payment Failures -> Proactive Billing Support & Grace Recovery
    # -------------------------------------------------------------------------
    has_payment_driver = any("payment" in f or "fail" in f for f in driver_features)
    if payment_failures >= 1 or has_payment_driver:
        urgency = "Immediate" if payment_failures >= 2 or churn_probability >= 0.70 else "High"
        matched_actions.append({
            "action_type": ACTION_BILLING_SUPPORT,
            "title": "Proactive Billing Support & 1-Click Payment Recovery",
            "description": (
                f"Subscriber experienced {payment_failures} billing failure(s) in the last 90 days. "
                "Trigger automated smart dunning email and an in-app modal prompt offering 7 days of "
                "grace streaming while updating credit card or linking alternative payment (PayPal/Apple Pay)."
            ),
            "urgency": urgency,
            "recommended_channel": "in-app modal & email",
            "triggered_by": [f"payment_failures: {payment_failures} declines (involuntary churn risk)"],
            "rationale": "Over 40% of streaming churn is involuntary due to expired cards; proactive grace recovery saves the account immediately.",
            "_priority_weight": 110 if urgency == "Immediate" else 95
        })

    # -------------------------------------------------------------------------
    # RULE 2: Critical Churn Risk / About to Cancel -> Pause Subscription Offer
    # -------------------------------------------------------------------------
    if risk_tier == "Critical" or churn_probability >= 0.85:
        matched_actions.append({
            "action_type": ACTION_PAUSE_SUBSCRIPTION,
            "title": "Pause Subscription Offer (1–3 Months Free Hold)",
            "description": (
                f"Subscriber exhibits extreme churn probability ({churn_probability*100:.1f}%, Critical Tier). "
                "Instead of full cancellation, deploy the 'Pause & Save' retention modal offering up to 3 months "
                "account hold with zero billing, keeping their saved watchlists, profiles, and history intact."
            ),
            "urgency": "Immediate",
            "recommended_channel": "in-app modal",
            "triggered_by": [f"risk_tier: {risk_tier} ({churn_probability*100:.1f}% churn probability)"],
            "rationale": "Pause offers retain over 30% of subscribers who would otherwise permanently churn to competitor streaming platforms.",
            "_priority_weight": 105
        })

    # -------------------------------------------------------------------------
    # RULE 3: Low Watch Hours / High Dormancy -> Personalized Content Recommendation
    # -------------------------------------------------------------------------
    has_watch_driver = any("watch" in f or "day" in f for f in driver_features)
    is_dormant = days_since_last_watch >= 12 or watch_hours < 8.0
    if is_dormant or has_watch_driver:
        urgency = "High" if days_since_last_watch >= 20 else "Medium"
        triggers = []
        if days_since_last_watch >= 12:
            triggers.append(f"days_since_last_watch: {days_since_last_watch} days inactive")
        if watch_hours < 8.0:
            triggers.append(f"watch_hours_last_30_days: {watch_hours} hrs (low engagement)")

        matched_actions.append({
            "action_type": ACTION_CONTENT_RECOMMENDATION,
            "title": "Personalized Content Discovery & Trending Watchlist Email",
            "description": (
                f"Subscriber has been inactive for {days_since_last_watch} days with only {watch_hours:.1f} watch hours this month. "
                "Trigger a personalized 'New Releases For You' email push featuring top-rated series and movie releases "
                "matching their past viewing genre preferences."
            ),
            "urgency": urgency,
            "recommended_channel": "email & push notification",
            "triggered_by": triggers or ["streaming inactivity signals"],
            "rationale": "Re-surfacing compelling content sparks spontaneous evening streaming sessions and restores active habituation.",
            "_priority_weight": 90 if urgency == "High" else 75
        })

    # -------------------------------------------------------------------------
    # RULE 4: Price-Sensitive Basic Plan Users -> Annual Plan Switch Loyalty Discount
    # -------------------------------------------------------------------------
    has_price_driver = any("price" in f or "basic" in f or "plan" in f for f in driver_features)
    is_basic_sensitive = plan == "Basic" and (churn_probability >= 0.50 or has_price_driver)
    if is_basic_sensitive or (plan in ["Basic", "Standard"] and monthly_price >= 8.99 and churn_probability >= 0.60):
        matched_actions.append({
            "action_type": ACTION_ANNUAL_DISCOUNT,
            "title": "Annual Plan Switch Loyalty Discount (25% Savings)",
            "description": (
                f"Subscriber is on the {plan} plan (${monthly_price:.2f}/mo) and shows price sensitivity. "
                "Offer a 25% discount to switch to an annual billing cycle, locking in 12 months of uninterrupted streaming "
                "for the price of 9 months."
            ),
            "urgency": "High" if churn_probability >= 0.70 else "Medium",
            "recommended_channel": "email",
            "triggered_by": [f"subscription_plan: {plan} with elevated churn probability ({churn_probability*100:.1f}%)"],
            "rationale": "Annual commitments eliminate monthly cancellation decisions and drastically lower voluntary subscriber churn.",
            "_priority_weight": 85 if churn_probability >= 0.70 else 70
        })

    # -------------------------------------------------------------------------
    # RULE 5: High Support Tickets -> Streaming Quality & Playback Concierge
    # -------------------------------------------------------------------------
    has_ticket_driver = any("ticket" in f or "support" in f for f in driver_features)
    if tickets >= 2 or has_ticket_driver:
        matched_actions.append({
            "action_type": ACTION_STREAMING_CONCIERGE,
            "title": "VIP Streaming Quality & Device Optimization Outreach",
            "description": (
                f"Subscriber opened {tickets} technical support tickets in recent months. "
                "Deploy a priority concierge outreach email offering guided 4K/HDR bandwidth optimization, "
                "device cache clearing steps, and a complimentary 1-month speed/quality trial credit."
            ),
            "urgency": "High" if tickets >= 3 else "Medium",
            "recommended_channel": "email & priority ticket follow-up",
            "triggered_by": [f"customer_support_tickets: {tickets} opened tickets"],
            "rationale": "Resolving buffering and video playback friction restores faith in service reliability.",
            "_priority_weight": 80
        })

    # -------------------------------------------------------------------------
    # RULE 6: Low Risk -> VIP Early Access & Screening Perks
    # -------------------------------------------------------------------------
    if not matched_actions or risk_tier == "Low" or churn_probability < 0.40:
        matched_actions.append({
            "action_type": ACTION_VIP_LOYALTY,
            "title": "VIP Early Access Premiere Screening & Referral Pass",
            "description": (
                "Subscriber demonstrates healthy streaming activity and high engagement. "
                "Send an exclusive invite to virtual early premiere screenings, offer a free extra household profile, "
                "and share a VIP referral link rewarding 1 month free for every friend enrolled."
            ),
            "urgency": "Low",
            "recommended_channel": "in-app notification & email",
            "triggered_by": ["Healthy streaming engagement (Low churn tier)"],
            "rationale": "Rewarding high-value advocates deepens brand loyalty and turns satisfied subscribers into organic brand promoters.",
            "_priority_weight": 50
        })

    # Sort actions by priority weight
    matched_actions.sort(key=lambda x: x.get("_priority_weight", 50), reverse=True)

    # Clean internal priority keys
    for a in matched_actions:
        a.pop("_priority_weight", None)

    primary_action = matched_actions[0] if matched_actions else None

    # Natural language retention strategy summary
    if risk_tier == "Critical":
        strategy_summary = (
            f"URGENT: Subscriber is at CRITICAL churn risk ({churn_probability*100:.1f}%). "
            f"Primary trigger: {primary_action['title']}. Immediately deploy {primary_action['recommended_channel']} "
            "intervention before month-end billing."
        )
    elif risk_tier == "High":
        strategy_summary = (
            f"HIGH RISK: Churn probability at {churn_probability*100:.1f}%. Recommend immediate execution of "
            f"{primary_action['title']} to re-engage subscriber."
        )
    elif risk_tier == "Medium":
        strategy_summary = (
            f"MODERATE RISK: Subscriber shows churn vulnerability ({churn_probability*100:.1f}%). "
            f"Deploy {primary_action['title']} to strengthen engagement."
        )
    else:
        strategy_summary = (
            f"HEALTHY: Subscriber is actively engaged ({churn_probability*100:.1f}% churn risk). "
            "Focus on loyalty perks and organic referral growth."
        )

    return {
        "primary_action": primary_action,
        "suggested_actions": matched_actions,
        "retention_strategy_summary": strategy_summary
    }
