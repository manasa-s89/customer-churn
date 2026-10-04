"""
Pydantic Schemas for OTT Streaming Churn Prediction Platform.
Defines data contracts for inference requests, customer directory, interventions, and analytics KPIs.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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

    class Config:
        json_schema_extra = {
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
    title: str
    description: str
    urgency: Optional[str] = "High"
    recommended_channel: Optional[str] = "email"
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

