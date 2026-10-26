"""
Pydantic Schemas for OTT Streaming Churn Prediction Platform.
Defines data contracts for inference requests, customer directory, interventions, and analytics KPIs.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, ConfigDict


class CustomerInput(BaseModel):
    """Payload representing an OTT streaming subscriber profile for churn inference."""
    customer_id: Optional[str] = Field(None, description="Unique Subscriber Identifier, e.g. 'OTT-84920'")
    subscription_plan: str = Field("Basic", description="Tier plan: 'Basic', 'Standard', 'Premium'")
    monthly_price: float = Field(8.99, ge=0.0, description="Monthly recurring subscription cost ($)")
    watch_hours_last_30_days: float = Field(0.0, ge=0.0, description="Total viewing hours in last 30 days")
    days_since_last_watch: int = Field(0, ge=0, description="Inactivity metric: days since last stream")
    number_of_devices: int = Field(1, ge=1, le=10, description="Number of linked streaming devices")
    number_of_profiles: int = Field(1, ge=1, le=10, description="Number of household viewer profiles")
    downloads_count: int = Field(0, ge=0, description="Offline downloads in last 30 days")
    customer_support_tickets: int = Field(0, ge=0, description="Support tickets opened (playback/billing)")
    payment_failures: int = Field(0, ge=0, description="Billing decline events in last 90 days")
    free_trial_converted: str = Field("Yes", description="Trial conversion status: 'Yes' or 'No'")
    tenure_months: int = Field(1, ge=0, description="Subscriber tenure with platform in months")

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # 1. customer_id
        if not d.get("customer_id") and d.get("CustomerID"):
            d["customer_id"] = str(d["CustomerID"])

        # 2. subscription_plan / contract_type / plan
        if "subscription_plan" not in d:
            if "plan" in d:
                d["subscription_plan"] = str(d["plan"]).strip().capitalize()
            elif "contract_type" in d:
                contract = str(d["contract_type"]).strip().lower()
                if "two year" in contract:
                    d["subscription_plan"] = "Premium"
                elif "one year" in contract:
                    d["subscription_plan"] = "Standard"
                else:
                    d["subscription_plan"] = "Basic"

        # 3. monthly_price / monthly_charges
        if "monthly_price" not in d and "monthly_charges" in d:
            d["monthly_price"] = float(d["monthly_charges"])

        # 4. tenure_months / tenure
        if "tenure_months" not in d and "tenure" in d:
            d["tenure_months"] = int(d["tenure"])

        # 5. customer_support_tickets / support_tickets
        if "customer_support_tickets" not in d and "support_tickets" in d:
            d["customer_support_tickets"] = int(d["support_tickets"])

        # 6. payment_failures / payment_declines
        if "payment_failures" not in d and "payment_declines" in d:
            d["payment_failures"] = int(d["payment_declines"])

        # 7. days_since_last_watch / inactivity_days / last_login_days_ago
        if "days_since_last_watch" not in d:
            if "inactivity_days" in d:
                d["days_since_last_watch"] = int(d["inactivity_days"])
            elif "last_login_days_ago" in d:
                d["days_since_last_watch"] = int(d["last_login_days_ago"])

        # 8. watch_hours_last_30_days / usage_frequency
        if "watch_hours_last_30_days" not in d and "usage_frequency" in d:
            freq = str(d["usage_frequency"]).strip().lower()
            if "daily" in freq:
                d["watch_hours_last_30_days"] = 60.0
            elif "weekly" in freq:
                d["watch_hours_last_30_days"] = 25.0
            else:
                d["watch_hours_last_30_days"] = 1.0

        return d

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "customer_id": "OTT-78421",
                "subscription_plan": "Standard",
                "monthly_price": 14.99,
                "watch_hours_last_30_days": 18.5,
                "days_since_last_watch": 12,
                "number_of_devices": 3,
                "number_of_profiles": 2,
                "downloads_count": 4,
                "customer_support_tickets": 2,
                "payment_failures": 1,
                "free_trial_converted": "Yes",
                "tenure_months": 5
            }
        }
    )


class BatchPredictRequest(BaseModel):
    """Batch list of subscriber profiles to evaluate."""
    customers: List[CustomerInput]


class FactorImpact(BaseModel):
    """Individual feature contribution to churn probability."""
    feature: str
    impact: float
    transformed_value: Optional[float] = None
    description: Optional[str] = None


class InterventionSuggestion(BaseModel):
    """Targeted retention intervention rule result."""
    action_type: str
    title: str
    description: str
    urgency: str
    recommended_channel: str
    triggered_by: List[str]
    rationale: str


class PredictResponse(BaseModel):
    """Comprehensive churn prediction, explainability breakdown, and recommended retention actions."""
    customer_id: str
    churn_probability: float
    churn_probability_pct: str
    risk_tier: str       # Critical, High, Medium, Low
    predicted_churn: bool
    top_risk_drivers: List[FactorImpact]
    top_protective_factors: List[FactorImpact]
    explanation_summary: str
    recommended_action: str
    suggested_interventions: List[InterventionSuggestion]
    raw_features: Optional[Dict[str, Any]] = None


class InterveneRequest(BaseModel):
    """Payload to log and trigger a retention intervention."""
    customer_id: str
    action_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None
    urgency: Optional[str] = "High"
    recommended_channel: Optional[str] = None
    channel: Optional[str] = None
    status: Optional[str] = "Sent"
    notes: Optional[str] = None
    performed_by: Optional[str] = "Automated System"


class InterveneResponse(BaseModel):
    """Stored intervention confirmation."""
    id: int
    customer_id: str
    action_type: str
    title: str
    description: str
    urgency: str
    recommended_channel: str
    channel: Optional[str] = None
    status: str
    notes: Optional[str] = None
    performed_by: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None


class CustomerListItem(BaseModel):
    """Compact summary of subscriber for table directory listing."""
    customer_id: str
    subscription_plan: Optional[str] = "Basic"
    monthly_price: Optional[float] = 8.99
    watch_hours_last_30_days: Optional[float] = 0.0
    days_since_last_watch: Optional[int] = 0
    number_of_devices: Optional[int] = 1
    number_of_profiles: Optional[int] = 1
    customer_support_tickets: Optional[int] = 0
    payment_failures: Optional[int] = 0
    free_trial_converted: Optional[str] = "Yes"
    tenure_months: Optional[int] = 1
    churn_probability: Optional[float] = None
    churn_probability_pct: Optional[str] = None
    risk_tier: Optional[str] = None
    predicted_churn: Optional[bool] = None
    latest_scored_at: Optional[str] = None
    interventions_count: int = 0
    latest_intervention: Optional[Dict[str, Any]] = None


class CustomerListResponse(BaseModel):
    """Paginated customer directory response."""
    total_customers: int
    limit: int
    offset: int
    customers: List[CustomerListItem]


class CustomerDetailResponse(BaseModel):
    """Full 360-degree view of a subscriber."""
    customer: Dict[str, Any]
    latest_score: Optional[Dict[str, Any]] = None
    score_history: List[Dict[str, Any]] = []
    interventions: List[Dict[str, Any]] = []
    current_recommendations: Optional[Dict[str, Any]] = None


class InterventionUpdate(BaseModel):
    """Payload to update an intervention's status or notes."""
    status: Optional[str] = Field(None, description="Status: 'Pending', 'Sent', 'Responded', 'Resolved', 'Cancelled'")
    notes: Optional[str] = Field(None, description="Updated notes from retention agent or system")


class AnalyticsOverviewResponse(BaseModel):
    """High-level analytics overview metrics for frontend dashboard with 4 risk tiers."""
    total_customers: int
    critical_risk_count: int = 0
    critical_risk_pct: float = 0.0
    high_risk_count: int = 0
    high_risk_pct: float = 0.0
    moderate_risk_count: int = 0
    moderate_risk_pct: float = 0.0
    medium_risk_count: int = 0
    medium_risk_pct: float = 0.0
    low_risk_count: int = 0
    low_risk_pct: float = 0.0
    revenue_at_risk: float = 0.0
    total_monthly_revenue: float = 0.0
    avg_churn_probability: float = 0.0
    risk_distribution: Dict[str, int]
    subscription_plan_distribution: Dict[str, Any] = {}
    contract_distribution: Optional[Dict[str, Any]] = None  # Backward-compatible alias
    interventions_status_breakdown: Dict[str, int]
    trend_over_time: List[Dict[str, Any]]


class CSVUploadResponse(BaseModel):
    """Response returned upon batch CSV upload and scoring."""
    filename: str
    rows_processed: int
    critical_risk_detected: int = 0
    high_risk_detected: int = 0
    moderate_risk_detected: int = 0
    medium_risk_detected: int = 0
    low_risk_detected: int = 0
    total_revenue_at_risk: float
    sample_scored_customers: List[PredictResponse]


class EventIngestRequest(BaseModel):
    """Payload representing a real-time subscriber activity event."""
    customer_id: str = Field(..., description="Subscriber identifier, e.g. 'OTT-78421'")
    event_type: str = Field(..., description="Event type: 'watch_session', 'login', 'cancel_click', 'payment_failure', 'support_ticket', 'download'")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of event occurrence")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Event-specific metadata (e.g., duration_minutes, reason, plan_id)")


class EventIngestResponse(BaseModel):
    """Result of real-time event processing and immediate churn re-scoring."""
    status: str
    customer_id: str
    event_type: str
    previous_score: Dict[str, Any]
    new_score: Dict[str, Any]
    features_updated: Dict[str, Any]
    escalated_to_critical_queue: bool
    message: str


class SimulatorStatusResponse(BaseModel):
    """State of the live event simulation daemon."""
    is_running: bool
    events_generated: int
    interval_seconds: float
    last_event: Optional[Dict[str, Any]] = None

