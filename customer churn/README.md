# 🎬 RetainPulse AI — OTT Streaming Churn Intelligence Platform

An end-to-end Machine Learning, Decision Intelligence, and **Simulated Real-Time Event Ingestion** platform designed for **OTT Video Streaming Services** (e.g. Netflix, Disney+, HBO Max, Prime Video). 

RetainPulse detects subscriber churn risk, explains underlying drivers using **SHAP / feature attributions**, executes automated streaming retention interventions, classifies subscribers into **4 risk tiers** (including an emergency **Critical Attention Queue**), and ingests live streaming events in real-time.

---

> [!NOTE]
> **Simulated Real-Time Event Pipeline**: This platform features a built-in event ingestion layer (`POST /events/ingest`) and background stream simulator (`/simulate/live-events/start`). This is a **simulated real-time pipeline for demonstration and testing purposes**, allowing live observation of behavioral shifts and instant ML score updates without requiring external production webhook credentials.

---

## 📑 Table of Contents
- [OTT Domain Architecture](#-ott-domain-architecture)
- [4-Tier Risk Classification](#-4-tier-risk-classification)
- [Machine Learning Champion Benchmark](#-machine-learning-champion-benchmark)
- [OTT Streaming Behavioral Features](#-ott-streaming-behavioral-features)
- [Rule-Based Retention Playbooks](#-rule-based-retention-playbooks)
- [Real-Time Event Ingestion Pipeline](#-real-time-event-ingestion-pipeline)
- [REST API Reference](#-rest-api-reference)
- [Executive Dashboard Overview](#-executive-dashboard-overview)
- [Quickstart Guide](#-quickstart-guide)
- [Automated Verification Tests](#-automated-verification-tests)

---

## 🏛 OTT Domain Architecture

```mermaid
flowchart TD
    subgraph Real-Time Telemetry & Event Ingestion
        EVT["Live Events (watch_session, payment_failure, cancel_click, download, login)"]
        INGEST["POST /events/ingest: Real-Time Event Processor"]
        SIM["Live Event Simulator (/simulate/live-events/*)"]
        EVT --> INGEST
        SIM --> INGEST
    end

    subgraph Data & ML Layer (OTT Domain)
        DATA["data/ott_customer_churn.csv (7,500 Subscribers)"] --> PIPE["data_pipeline.py: Median Imputer + Scaler + OneHotEncoder"]
        PIPE --> TRAIN["train.py: 5-Fold Stratified Cross-Validation"]
        TRAIN --> MODEL["models/best_model.joblib (Logistic Regression 0.872 AUC)"]
        MODEL --> EXP["predict_and_explain.py: Feature Attribution & Risk Drivers"]
    end

    subgraph Decision Engine & Playbooks
        EXP --> REC["recommender.py: OTT Retention Playbook Engine"]
    end

    subgraph Backend API (FastAPI)
        API["main.py: FastAPI Server (Port 8000)"]
        INGEST --> API
        EXP & REC --> API
        API <--> DB[("churn.db: SQLite Persistence")]
    end

    subgraph Frontend Dashboard (SPA)
        UI["static/index.html & app.js: Executive Dark Mode SPA"]
        API <--> UI
        POLL["Silent Auto-Refresh Polling (4s Interval)"] --> UI
    end
```

---

## 🎯 4-Tier Risk Classification

All churn probabilities are mapped to 4 distinct risk tiers configured in [`config.py`](file:///c:/Users/Sampath%20S/Desktop/New%20Folder/customer%20churn/config.py):

| Risk Tier | Churn Probability Threshold | Color Hex | Recommended SLA | Target Queue |
| :--- | :---: | :---: | :---: | :--- |
| **Critical** | $\mathbf{\ge 90\%}$ | `#ef4444` | Immediate (&lt; 2 hours) | **Critical Attention Queue** (Emergency) |
| **High** | $\mathbf{70\% - 89\%}$ | `#f97316` | &lt; 24 hours | High Priority Retention List |
| **Medium** | $\mathbf{40\% - 69\%}$ | `#eab308` | &lt; 72 hours | Nurture & Re-engagement Pool |
| **Low** | $\mathbf{< 40\%}$ | `#10b981` | Standard cycle | Loyalty & Upsell Cohort |

- **Centralized Configuration**: Thresholds, labels, and color palettes live strictly in [`config.py`](file:///c:/Users/Sampath%20S/Desktop/New%20Folder/customer%20churn/config.py).
- **Critical Attention Queue**: The executive dashboard prominently surfaces subscribers with $\ge 90\%$ churn probability requiring emergency intervention before cancellation.

---

## 🧠 Machine Learning Champion Benchmark

Models were trained on 7,500 realistic synthetic OTT subscriber records using **5-Fold Stratified Cross-Validation**:

| Candidate Model | Mean CV AUC | Holdout ROC-AUC | F1-Score | Precision | Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **0.8643** | **0.8718** | **0.4323** | **0.2941** | **81.56%** | 🏆 **Champion** |
| Random Forest Classifier | 0.8541 | 0.8605 | 0.3541 | 0.3120 | 41.13% | Benchmark |
| XGBoost Classifier | 0.8410 | 0.8504 | 0.3804 | 0.2965 | 53.19% | Benchmark |

- **Why Logistic Regression Won**: High recall ($81.6\%$) ensures maximum churners are flagged early for intervention, with calibrated probabilities and reliable feature coefficients.
- Model artifacts are saved in [`models/`](file:///c:/Users/Sampath%20S/Desktop/New%20Folder/customer%20churn/models/) with metadata in [`models/model_metadata.json`](file:///c:/Users/Sampath%20S/Desktop/New%20Folder/customer%20churn/models/model_metadata.json).

---

## 📊 OTT Streaming Behavioral Features

| Feature Name | Type | Description | Churn Impact Direction |
| :--- | :--- | :--- | :---: |
| `subscription_plan` | Categorical | `Basic` ($8.99), `Standard` ($14.99), `Premium` ($20.99) | Basic plans churn faster |
| `monthly_price` | Numeric | Monthly recurring fee in USD | Higher price slightly raises churn risk |
| `watch_hours_last_30_days` | Numeric | Total streaming playback hours in past 30 days | **Strong Protective Factor** (more viewing = loyal) |
| `days_since_last_watch` | Numeric | Days since subscriber last streamed content | **High Risk Catalyst** (dormancy indicates churn) |
| `number_of_devices` | Numeric | Connected devices (TV, mobile, tablet, console) | Multi-device households retain better |
| `number_of_profiles` | Numeric | Active family / sub-profiles on account | Multi-profile accounts are stickier |
| `downloads_count` | Numeric | Offline downloads for on-the-go viewing | Engagement sign (reduces churn) |
| `customer_support_tickets` | Numeric | Support tickets logged in last 90 days | $\ge 2$ tickets indicates friction |
| `payment_failures` | Numeric | Declined billing cycles / card failures | **Severe Churn Catalyst** |
| `free_trial_converted` | Categorical | `Yes` or `No` | Converted users retain significantly longer |
| `tenure_months` | Numeric | Lifetime tenure in months | Newer accounts (&lt; 3 months) most vulnerable |

---

## 🎯 Rule-Based Retention Playbooks

The automated playbook engine in [`recommender.py`](file:///c:/Users/Sampath%20S/Desktop/New%20Folder/customer%20churn/recommender.py) matches subscriber risk drivers to OTT retention plays:

| Trigger Condition | Recommended Action Type | Retention Strategy | Recommended Channel | Urgency |
| :--- | :--- | :--- | :--- | :--- |
| `payment_failures >= 1` | `proactive_billing_support` | 7-day streaming grace period + 1-click update link | In-App Banner & SMS | **Critical** |
| `days_since_last_watch >= 25` | `pause_subscription_offer` | Offer 1–3 months free hold instead of cancel | In-App Cancel Modal | **High** |
| `watch_hours < 8.0` | `content_recommendation_email` | AI-curated watchlist based on past binge genres | Email & Push Notification | **Medium** |
| `monthly_price >= $15` & `tenure >= 6` | `annual_plan_switch_discount` | Switch to Annual with 2 months free ($16% discount) | Account Portal Banner | **Medium** |
| `customer_support_tickets >= 2` | `streaming_quality_concierge` | Direct 4K playback diagnostic & VIP live chat | In-App Concierge Chat | **High** |
| `watch_hours >= 40.0` & `tenure >= 12` | `vip_early_access_screening` | Early premiere screening ticket & loyalty badge | Email & In-App VIP Card | **Low** (Loyalty) |

---

## ⚡ Real-Time Event Ingestion Pipeline

### 1. Ingesting an Event (`POST /events/ingest`)
Accepts subscriber activities, updates their behavioral telemetry, immediately recalculates churn probability, and alerts if they escalate into the Critical Attention Queue:

```bash
curl -X POST "http://127.0.0.1:8000/events/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "OTT-97805",
       "event_type": "payment_failure",
       "metadata": {"reason": "Card declined on auto-renewal"}
     }'
```

**Response Example:**
```json
{
  "status": "success",
  "customer_id": "OTT-97805",
  "event_type": "payment_failure",
  "previous_score": { "churn_probability": 0.8897, "risk_tier": "High" },
  "new_score": { "churn_probability": 0.9411, "risk_tier": "Critical" },
  "features_updated": { "payment_failures": 1 },
  "escalated_to_critical_queue": true,
  "message": "Event 'payment_failure' processed for subscriber OTT-97805. Churn Risk shifted from High (89.0%) to Critical (94.1%) [+5.1%]. ATTENTION: Subscriber has escalated to the Critical Attention Queue (≥90% churn probability)!"
}
```

### 2. Live Simulator Endpoints
- `POST /simulate/live-events/start`: Generates continuous realistic events across existing subscribers every 3.5s.
- `POST /simulate/live-events/stop`: Pauses simulation.
- `POST /simulate/live-events/step`: Triggers exactly one instant realistic event across subscribers.
- `GET /simulate/live-events/status`: Simulator state and last generated event.

### 3. Dashboard Auto-Polling
The frontend polls `/simulate/live-events/status`, `/analytics/overview`, and `/customers` every 4 seconds. When events occur:
- The top banner flashes the live score delta.
- The 4-tier donut and KPIs update live.
- The Critical Attention Queue automatically populates when an account crosses the $\ge 90\%$ threshold!

---

## 🔌 REST API Reference

Swagger documentation is accessible at [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs):

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | API health, database status, and ML model artifact check |
| `GET` | `/dashboard` | Interactive frontend single-page application |
| `POST` | `/predict` | Single or batch OTT subscriber churn scoring and risk explanation |
| `POST` | `/events/ingest` | Ingest real-time event, update features, and re-score immediately |
| `POST` | `/simulate/live-events/start` | Start background simulated event generator |
| `POST` | `/simulate/live-events/stop` | Stop background simulated event generator |
| `POST` | `/simulate/live-events/step` | Manually trigger 1 simulated event on existing customer |
| `GET` | `/simulate/live-events/status` | Current status of live event simulation |
| `GET` | `/customers` | Filterable & sortable directory of subscribers |
| `GET` | `/customers/{id}` | Subscriber 360 profile, score history, and intervention logs |
| `POST` | `/intervene` | Record retention action taken on a subscriber |
| `GET` | `/analytics/overview` | Executive metrics, 4-tier risk counts, and plan distribution |
| `POST` | `/upload_csv` | Batch CSV upload scoring for bulk subscriber evaluation |

---

## 🚀 Quickstart Guide

### 1. Launch the Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Open the Executive Dashboard
Open your browser to:
👉 **`http://127.0.0.1:8000/dashboard`**

### 3. Live Demonstration Controls
- Click **"⚡ Sample Event"** in the top navigation bar to inject a single event and witness immediate score shift.
- Click **"Start Live Stream"** to begin background event simulation with live ticker updates.

---

## 🧪 Automated Verification Tests

Run the complete test suite verifying preprocessing, model inference, 4-tier classification, and real-time event ingestion:

```bash
python run_all_tests.py
```

**Results:**
```text
Total Unit Tests Discovered: 16
Ran 16 tests in 6.95s
OK — ALL 16 TESTS PASSED SUCCESSFULLY!
```
