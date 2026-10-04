"""
SQLAlchemy ORM Models for OTT Streaming Churn Intelligence Platform.
Persists subscriber profiles, model inference scores (with 4 risk tiers), and intervention tracking logs.
"""

from datetime import datetime, timezone
import json
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    ForeignKey
)
from sqlalchemy.orm import relationship
from database import Base


class CustomerRecord(Base):
    """
    Subscribers table storing OTT viewing activity, plan details, and billing health.
    """
    __tablename__ = "customers"

    customer_id = Column(String(64), primary_key=True, index=True)
    subscription_plan = Column(String(20), nullable=False, default="Basic")    # Basic, Standard, Premium
    monthly_price = Column(Float, nullable=False, default=8.99)
    watch_hours_last_30_days = Column(Float, nullable=False, default=0.0)
    days_since_last_watch = Column(Integer, nullable=False, default=0)
    number_of_devices = Column(Integer, nullable=False, default=1)
    number_of_profiles = Column(Integer, nullable=False, default=1)
    downloads_count = Column(Integer, nullable=False, default=0)
    customer_support_tickets = Column(Integer, nullable=False, default=0)
    payment_failures = Column(Integer, nullable=False, default=0)
    free_trial_converted = Column(String(10), nullable=False, default="Yes")   # Yes, No
    tenure_months = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    scores = relationship(
        "ChurnScoreRecord",
        back_populates="customer",
        cascade="all, delete-orphan",
        order_by="desc(ChurnScoreRecord.created_at)"
    )
    interventions = relationship(
        "InterventionRecord",
        back_populates="customer",
        cascade="all, delete-orphan",
        order_by="desc(InterventionRecord.created_at)"
    )

    def to_dict(self):
        return {
            "customer_id": self.customer_id,
            "subscription_plan": self.subscription_plan,
            "monthly_price": self.monthly_price,
            "watch_hours_last_30_days": self.watch_hours_last_30_days,
            "days_since_last_watch": self.days_since_last_watch,
            "number_of_devices": self.number_of_devices,
            "number_of_profiles": self.number_of_profiles,
            "downloads_count": self.downloads_count,
            "customer_support_tickets": self.customer_support_tickets,
            "payment_failures": self.payment_failures,
            "free_trial_converted": self.free_trial_converted,
            "tenure_months": self.tenure_months,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChurnScoreRecord(Base):
    """
    Historical model churn predictions and feature attributions.
    Supports 4 tiers: Critical (>=90%), High (70-89%), Medium (40-69%), Low (<40%).
    """
    __tablename__ = "churn_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    churn_probability = Column(Float, nullable=False)
    risk_tier = Column(String(20), nullable=False)  # Critical, High, Medium, Low
    predicted_churn = Column(Boolean, nullable=False)
    
    # Serialized JSON lists of feature attributions
    top_risk_drivers_json = Column(Text, nullable=True)
    top_protective_factors_json = Column(Text, nullable=True)

    explanation_summary = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    customer = relationship("CustomerRecord", back_populates="scores")

    def to_dict(self):
        drivers = []
        if self.top_risk_drivers_json:
            try:
                drivers = json.loads(self.top_risk_drivers_json)
            except Exception:
                drivers = []

        protective = []
        if self.top_protective_factors_json:
            try:
                protective = json.loads(self.top_protective_factors_json)
            except Exception:
                protective = []

        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "churn_probability": self.churn_probability,
            "churn_probability_pct": f"{self.churn_probability * 100:.1f}%",
            "risk_tier": self.risk_tier,
            "predicted_churn": self.predicted_churn,
            "top_risk_drivers": drivers,
            "top_protective_factors": protective,
            "explanation_summary": self.explanation_summary,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class InterventionRecord(Base):
    """
    Retention intervention tracking: pause offers, personalized recommendations, billing grace recovery.
    """
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    action_type = Column(String(64), nullable=False)
    title = Column(String(128), nullable=False)
    description = Column(Text, nullable=False)
    urgency = Column(String(20), default="Medium")
    recommended_channel = Column(String(32), default="email")
    status = Column(String(32), default="Pending")  # Pending, Sent, Responded, Resolved, Cancelled
    notes = Column(Text, nullable=True)
    performed_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    customer = relationship("CustomerRecord", back_populates="interventions")

    def to_dict(self):
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "action_type": self.action_type,
            "title": self.title,
            "description": self.description,
            "urgency": self.urgency,
            "recommended_channel": self.recommended_channel,
            "status": self.status,
            "notes": self.notes,
            "performed_by": self.performed_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
