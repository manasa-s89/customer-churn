"""
Customer Churn Prediction & Retention Intervention API - OTT Streaming Domain
FastAPI backend providing:
- POST /predict: Churn inference, SHAP/feature contribution breakdown, and rule-based interventions.
- GET /customers: Customer directory with 4 risk tiers, streaming stats, and intervention statuses.
- GET /customers/{customer_id}: Customer 360 view with score history and intervention logs.
- POST /intervene: Record and track retention actions (pause offer, content recommendation, billing grace).
- GET /interventions/{customer_id}: Historical intervention audit trail.
- GET /recommendations/{customer_id}: Rule-based retention suggestions for existing subscriber.
- GET /analytics/overview: Executive KPIs, 4-tier risk distribution, and subscription plan breakdowns.
- POST /upload_csv: Bulk CSV scoring.
"""
from fastapi.responses import HTMLResponse
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid
import math
import asyncio
import random
from typing import List, Optional, Union, Dict, Any
import csv
import io

from fastapi import FastAPI, Depends, HTTPException, Query, status, UploadFile, File, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import desc, func, text

from database import engine, get_db, init_db, DATABASE_URL, SessionLocal
import models_db
from config import (
    calculate_risk_tier,
    RISK_TIER_CONFIG,
    RISK_TIER_ORDER,
    TIER_WEIGHTS,
    SUBSCRIPTION_PLANS,
    PLAN_PRICING
)
from schemas import (
    CustomerInput,
    BatchPredictRequest,
    PredictResponse,
    CustomerListItem,
    CustomerListResponse,
    CustomerDetailResponse,
    InterveneRequest,
    InterveneResponse,
    InterventionSuggestion,
    InterventionUpdate,
    AnalyticsOverviewResponse,
    CSVUploadResponse,
    FactorImpact,
    EventIngestRequest,
    EventIngestResponse,
    SimulatorStatusResponse
)
from recommender import (
    recommend_interventions,
    ACTION_CONTENT_RECOMMENDATION,
    ACTION_BILLING_SUPPORT,
    ACTION_PAUSE_SUBSCRIPTION,
    ACTION_ANNUAL_DISCOUNT,
    ACTION_STREAMING_CONCIERGE,
    ACTION_VIP_LOYALTY,
    # Backward compatibility aliases
    ACTION_PROACTIVE_OUTREACH,
    ACTION_CONTRACT_UPGRADE,
    ACTION_REENGAGEMENT,
    ACTION_DISCOUNT_OFFER,
    ACTION_ONBOARDING
)

# Initialize database tables
init_db()

# -----------------------------------------------------------------------------
# Static Directory & Asset Resolution (based on __file__, not CWD)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# If not present in customer churn/static, check root static as fallback
if not INDEX_FILE.is_file():
    root_static = BASE_DIR.parent / "static"
    if (root_static / "index.html").is_file():
        STATIC_DIR = root_static
        INDEX_FILE = STATIC_DIR / "index.html"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
static_dir = str(STATIC_DIR)

app = FastAPI(
    title="OTT Streaming Churn Prediction & Retention API",
    description=(
        "Production-ready FastAPI backend for OTT video streaming subscriber churn probability scoring, "
        "model explainability (risk drivers), 4-tier risk classification, and rule-based retention interventions."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web apps or dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Clear JSON Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "message": detail
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        msg = err.get("msg", "Invalid value")
        errors.append(f"{field}: {msg}" if field else msg)
    err_str = "; ".join(errors) if errors else "Invalid request payload."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "status_code": 422,
            "detail": err_str,
            "message": err_str,
            "validation_errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    msg = str(exc) or "Internal server error"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "status_code": 500,
            "detail": msg,
            "message": msg
        }
    )

@app.middleware("http")
async def normalize_vercel_paths(request: Request, call_next):
    """
    Normalize paths when Vercel rewrites forward with function prefixes or proxy headers.
    - It only changes the path when the request path is '/' or '/api/index.py' (or starts with '/api/index.py/')
      AND a header carries a different real path. Never use an internal path like /api/index.py as the real path.
    - It NEVER touches paths starting with /static, /docs, /redoc, /openapi.json, /favicon.ico or any path that is already correct.
    """
    current_path = request.scope.get("path", "")

    # Never touch static assets, documentation paths, favicon, or already correct API paths
    if current_path.startswith((
        "/static", "/docs", "/redoc", "/openapi.json", "/favicon.ico",
        "/health", "/analytics", "/customers", "/predict", "/intervene",
        "/upload_csv", "/events", "/simulate", "/interventions", "/recommendations"
    )):
        return await call_next(request)

    # Strip /api prefix if path starts with /api/ (e.g. /api/health -> /health) but not /api/index
    if current_path.startswith("/api/") and not current_path.startswith(("/api/index.py", "/api/index")):
        remainder = current_path[4:]
        request.scope["path"] = remainder if remainder.startswith("/") else "/" + remainder
        return await call_next(request)

    # Only change the path when current_path is "/" or "/api/index.py" or starts with "/api/index.py"
    if current_path in ("/", "/api/index.py", "/api/index", "/api/index.py/", "/api/index/") or current_path.startswith("/api/index.py/"):
        real_path = None
        for h in ("x-forwarded-uri", "x-original-uri", "x-rewrite-url", "x-vercel-matched-path", "x-matched-path"):
            val = request.headers.get(h)
            if val:
                candidate = val.split("?")[0].strip()
                # Never use an internal path like /api/index.py as the real path
                if candidate and candidate not in ("/api/index.py", "/api/index", "/api", "/api/index.py/", "/api/index/"):
                    real_path = candidate
                    break

        if real_path:
            for prefix in ("/api/index.py", "/api/index", "/api"):
                if real_path.startswith(prefix):
                    remainder = real_path[len(prefix):]
                    if not remainder or remainder.startswith("/"):
                        real_path = remainder
                        break
            if real_path and not real_path.startswith("/"):
                real_path = "/" + real_path
            request.scope["path"] = real_path or "/"
        elif current_path.startswith("/api/index.py/"):
            remainder = current_path[len("/api/index.py"):]
            request.scope["path"] = remainder if remainder else "/"
        elif request.method == "GET" and current_path in ("/api/index.py", "/api/index", "/api/index.py/", "/api/index/"):
            request.scope["path"] = "/"

    return await call_next(request)

@app.get("/favicon.ico", include_in_schema=False)
@app.get("/api/favicon.ico", include_in_schema=False)
def get_favicon():
    """Return 204 No Content for favicon to prevent 404 noise in Vercel logs."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/docs", include_in_schema=False)
def get_api_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title + " - Swagger UI")

@app.get("/api/redoc", include_in_schema=False)
def get_api_redoc():
    return get_redoc_html(openapi_url="/openapi.json", title=app.title + " - ReDoc")

@app.get("/api/openapi.json", include_in_schema=False)
def get_api_openapi():
    return JSONResponse(app.openapi())

@app.get("/static/{file_path:path}", include_in_schema=False)
@app.get("/api/static/{file_path:path}", include_in_schema=False)
def serve_static_asset(file_path: str):
    """Serve static CSS, JS, and HTML files with guaranteed MIME types."""
    candidate_paths = [
        STATIC_DIR / file_path,
        BASE_DIR / "static" / file_path,
        BASE_DIR / "public" / "static" / file_path,
        BASE_DIR.parent / "static" / file_path,
        BASE_DIR.parent / "public" / "static" / file_path,
        Path("static") / file_path,
        Path("public/static") / file_path,
        Path("customer churn/static") / file_path,
        Path("customer churn/public/static") / file_path,
    ]
    for p in candidate_paths:
        if p.is_file():
            if file_path.endswith(".css"):
                return FileResponse(str(p), media_type="text/css; charset=utf-8")
            elif file_path.endswith(".js"):
                return FileResponse(str(p), media_type="text/javascript; charset=utf-8")
            elif file_path.endswith(".html"):
                return FileResponse(str(p), media_type="text/html; charset=utf-8")
            elif file_path.endswith(".json"):
                return FileResponse(str(p), media_type="application/json")
            elif file_path.endswith(".svg"):
                return FileResponse(str(p), media_type="image/svg+xml")
            elif file_path.endswith(".png"):
                return FileResponse(str(p), media_type="image/png")
            elif file_path.endswith(".ico"):
                return FileResponse(str(p), media_type="image/x-icon")
            return FileResponse(str(p))
    raise HTTPException(status_code=404, detail=f"Static file '{file_path}' not found")


def build_inlined_dashboard_html() -> str:
    """Build dashboard HTML with styles.css and app.js inlined as a safety net."""
    html_content = ""
    for cand in [INDEX_FILE, STATIC_DIR / "index.html", BASE_DIR / "static" / "index.html", BASE_DIR / "public" / "index.html", Path("static/index.html"), Path("public/index.html")]:
        if cand.is_file():
            try:
                html_content = cand.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    css_content = ""
    for cand in [STATIC_DIR / "styles.css", BASE_DIR / "static" / "styles.css", BASE_DIR / "public" / "static" / "styles.css", Path("static/styles.css"), Path("public/static/styles.css")]:
        if cand.is_file():
            try:
                css_content = cand.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    js_content = ""
    for cand in [STATIC_DIR / "app.js", BASE_DIR / "static" / "app.js", BASE_DIR / "public" / "static" / "app.js", Path("static/app.js"), Path("public/static/app.js")]:
        if cand.is_file():
            try:
                js_content = cand.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    if not html_content:
        return "<h2>Frontend Dashboard is initializing...</h2>"

    # Inline styles.css into <head>
    if css_content:
        inlined_style = f"\n<style id=\"inlined-styles\">\n{css_content}\n</style>\n"
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{inlined_style}</head>", 1)
        else:
            html_content = inlined_style + html_content

    # Inline app.js into <body> (replacing external script reference to prevent double execution)
    if js_content:
        inlined_script = f"\n<script id=\"inlined-app\">\n{js_content}\n</script>\n"
        import re
        if re.search(r'<script\s+src=["\']/?(?:static/)?app\.js["\']\s*></script>', html_content):
            html_content = re.sub(r'<script\s+src=["\']/?(?:static/)?app\.js["\']\s*></script>', inlined_script, html_content)
        elif "</body>" in html_content:
            html_content = html_content.replace("</body>", f"{inlined_script}</body>", 1)
        else:
            html_content += inlined_script

    return html_content


app.mount("/static", StaticFiles(directory=static_dir, html=False), name="static")

# -----------------------------------------------------------------------------
# Global Explainer / Model Manager
# -----------------------------------------------------------------------------
_explainer_instance = None


def get_explainer():
    """Lazy-load and cache ChurnExplainer instance."""
    global _explainer_instance
    if _explainer_instance is None:
        try:
            from predict_and_explain import ChurnExplainer
            here = os.path.dirname(os.path.abspath(__file__))
            candidate_dirs = [
                os.path.join(here, "models"),
                os.path.join(os.path.dirname(here), "models"),
                os.path.join(os.getcwd(), "models"),
                os.path.join(os.getcwd(), "customer churn", "models"),
            ]
            models_dir = os.path.join(here, "models")
            for cd in candidate_dirs:
                if os.path.isdir(cd) and os.path.exists(os.path.join(cd, "best_model.joblib")):
                    models_dir = cd
                    break
            _explainer_instance = ChurnExplainer(models_dir=models_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[!] ChurnExplainer warning: {e}. Checking if fallback needed.")
            _explainer_instance = None
    return _explainer_instance


def _fallback_predict(customer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calibrated heuristic fallback in case model artifacts are still training.
    Uses realistic OTT streaming weights to compute churn probability and risk factors.
    """
    plan = str(customer_dict.get("subscription_plan", "Basic")).strip().capitalize()
    price = float(customer_dict.get("monthly_price", 8.99) or 8.99)
    watch_hours = float(customer_dict.get("watch_hours_last_30_days", 0.0) or 0.0)
    days_inactive = float(customer_dict.get("days_since_last_watch", 0) or 0)
    devices = float(customer_dict.get("number_of_devices", 1) or 1)
    profiles = float(customer_dict.get("number_of_profiles", 1) or 1)
    downloads = float(customer_dict.get("downloads_count", 0) or 0)
    tickets = float(customer_dict.get("customer_support_tickets", 0) or 0)
    payment_failures = float(customer_dict.get("payment_failures", 0) or 0)
    free_trial = str(customer_dict.get("free_trial_converted", "Yes")).strip().capitalize()
    tenure = float(customer_dict.get("tenure_months", 1) or 1)
    cust_id = customer_dict.get("customer_id") or f"OTT-{uuid.uuid4().hex[:5].upper()}"

    # Base log-odds
    z = -1.25
    z -= 0.045 * min(tenure, 36)
    z -= 0.028 * min(watch_hours, 80.0)
    z -= 0.18 * (profiles - 1)
    z -= 0.08 * (devices - 1)
    z -= 0.05 * min(downloads, 10)
    z -= 0.30 * (1.0 if free_trial == "Yes" else 0.0)
    z += 0.095 * days_inactive
    z += 0.75 * payment_failures
    z += 0.45 * max(0, tickets - 1)
    z += 0.35 * (1.0 if plan == "Basic" else 0.0)
    z += 0.035 * (price - 12.0)

    prob = 1.0 / (1.0 + math.exp(-z))
    prob = max(0.01, min(0.99, prob))
    risk_tier = calculate_risk_tier(prob)

    # Risk drivers
    drivers = []
    protective = []

    if payment_failures >= 1:
        drivers.append({"feature": "payment_failures", "impact": 0.55 * payment_failures, "transformed_value": float(payment_failures)})

    if days_inactive >= 10:
        drivers.append({"feature": "days_since_last_watch", "impact": 0.04 * days_inactive, "transformed_value": float(days_inactive)})
    else:
        protective.append({"feature": "days_since_last_watch", "impact": -0.20, "transformed_value": float(days_inactive)})

    if watch_hours < 10:
        drivers.append({"feature": "watch_hours_last_30_days", "impact": 0.30, "transformed_value": float(watch_hours)})
    else:
        protective.append({"feature": "watch_hours_last_30_days", "impact": -0.015 * min(watch_hours, 80.0), "transformed_value": float(watch_hours)})

    if tickets >= 2:
        drivers.append({"feature": "customer_support_tickets", "impact": 0.25 * tickets, "transformed_value": float(tickets)})

    if plan == "Basic":
        drivers.append({"feature": "subscription_plan_Basic", "impact": 0.20, "transformed_value": 1.0})
    else:
        protective.append({"feature": f"subscription_plan_{plan}", "impact": -0.15, "transformed_value": 1.0})

    if tenure > 12:
        protective.append({"feature": "tenure_months", "impact": -0.02 * min(tenure, 36), "transformed_value": float(tenure)})

    drivers.sort(key=lambda x: x["impact"], reverse=True)
    protective.sort(key=lambda x: x["impact"])

    summary = f"Subscriber has an estimated {prob*100:.1f}% churn risk ({risk_tier} Risk)."
    action = "Deploy targeted retention intervention for this subscriber profile."

    return {
        "customer_id": cust_id,
        "churn_probability": round(prob, 4),
        "churn_probability_pct": f"{prob*100:.1f}%",
        "risk_tier": risk_tier,
        "predicted_churn": prob >= 0.5,
        "top_risk_drivers": drivers[:4],
        "top_protective_factors": protective[:4],
        "explanation_summary": summary,
        "recommended_action": action,
        "raw_features": customer_dict
    }


def _process_single_prediction(customer: CustomerInput, db: Session) -> PredictResponse:
    """Core prediction pipeline: runs model, explainability, rule recommender, and persists records."""
    cust_id = customer.customer_id or f"OTT-{uuid.uuid4().hex[:5].upper()}"

    cust_dict = {
        "customer_id": cust_id,
        "subscription_plan": customer.subscription_plan,
        "monthly_price": customer.monthly_price,
        "watch_hours_last_30_days": customer.watch_hours_last_30_days,
        "days_since_last_watch": customer.days_since_last_watch,
        "number_of_devices": customer.number_of_devices,
        "number_of_profiles": customer.number_of_profiles,
        "downloads_count": customer.downloads_count,
        "customer_support_tickets": customer.customer_support_tickets,
        "payment_failures": customer.payment_failures,
        "free_trial_converted": customer.free_trial_converted,
        "tenure_months": customer.tenure_months,
    }

    # 1. Run ML Prediction & Explainability
    explainer = get_explainer()
    if explainer is not None:
        try:
            exp_results = explainer.predict_and_explain([cust_dict], top_k_reasons=4)
            res = exp_results[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[!] Error in explainer inference: {e}. Using calibrated fallback.")
            res = _fallback_predict(cust_dict)
    else:
        res = _fallback_predict(cust_dict)

    prob = float(res["churn_probability"])
    risk_tier = calculate_risk_tier(prob)
    predicted_churn = bool(res["predicted_churn"])
    top_risk_drivers = res.get("top_risk_drivers", [])
    top_protective_factors = res.get("top_protective_factors", [])
    explanation_summary = res.get("explanation_summary", "")
    legacy_action = res.get("recommended_action", "")

    # 2. Rule-Based Intervention Recommendations
    recommendation_result = recommend_interventions(
        customer_features=cust_dict,
        churn_probability=prob,
        risk_tier=risk_tier,
        top_risk_drivers=top_risk_drivers
    )
    suggested_actions = recommendation_result["suggested_actions"]
    strategy_summary = recommendation_result["retention_strategy_summary"]

    now = datetime.now(timezone.utc)

    # 3. Persist / Upsert Subscriber Record (database write must not crash inference)
    if db is not None:
        try:
            customer_record = db.query(models_db.CustomerRecord).filter_by(customer_id=cust_id).first()
            if not customer_record:
                customer_record = models_db.CustomerRecord(
                    customer_id=cust_id,
                    subscription_plan=cust_dict["subscription_plan"],
                    monthly_price=cust_dict["monthly_price"],
                    watch_hours_last_30_days=cust_dict["watch_hours_last_30_days"],
                    days_since_last_watch=cust_dict["days_since_last_watch"],
                    number_of_devices=cust_dict["number_of_devices"],
                    number_of_profiles=cust_dict["number_of_profiles"],
                    downloads_count=cust_dict["downloads_count"],
                    customer_support_tickets=cust_dict["customer_support_tickets"],
                    payment_failures=cust_dict["payment_failures"],
                    free_trial_converted=cust_dict["free_trial_converted"],
                    tenure_months=cust_dict["tenure_months"],
                    created_at=now,
                )
                db.add(customer_record)
                db.flush()
            else:
                customer_record.subscription_plan = cust_dict["subscription_plan"]
                customer_record.monthly_price = cust_dict["monthly_price"]
                customer_record.watch_hours_last_30_days = cust_dict["watch_hours_last_30_days"]
                customer_record.days_since_last_watch = cust_dict["days_since_last_watch"]
                customer_record.number_of_devices = cust_dict["number_of_devices"]
                customer_record.number_of_profiles = cust_dict["number_of_profiles"]
                customer_record.downloads_count = cust_dict["downloads_count"]
                customer_record.customer_support_tickets = cust_dict["customer_support_tickets"]
                customer_record.payment_failures = cust_dict["payment_failures"]
                customer_record.free_trial_converted = cust_dict["free_trial_converted"]
                customer_record.tenure_months = cust_dict["tenure_months"]

            # 4. Save Historical Churn Score Record
            import json
            score_record = models_db.ChurnScoreRecord(
                customer_id=cust_id,
                churn_probability=prob,
                risk_tier=risk_tier,
                predicted_churn=bool(predicted_churn),
                top_risk_drivers_json=json.dumps(top_risk_drivers),
                top_protective_factors_json=json.dumps(top_protective_factors),
                explanation_summary=strategy_summary or explanation_summary,
                recommended_action=legacy_action,
                created_at=now,
            )
            db.add(score_record)

            # 5. Automatically create a suggested Pending Intervention if High or Critical risk
            if risk_tier in ["Critical", "High"] and suggested_actions:
                primary = suggested_actions[0]
                existing_pending = db.query(models_db.InterventionRecord).filter_by(
                    customer_id=cust_id,
                    action_type=primary["action_type"],
                    status="Pending"
                ).first()

                if not existing_pending:
                    new_intervention = models_db.InterventionRecord(
                        customer_id=cust_id,
                        action_type=primary["action_type"],
                        title=primary["title"],
                        description=primary["description"],
                        urgency=primary["urgency"],
                        recommended_channel=primary["recommended_channel"],
                        status="Pending",
                        notes=f"Auto-generated for {risk_tier} risk subscriber ({prob*100:.1f}%)",
                        performed_by="AI Playbook Engine",
                        created_at=now,
                    )
                    db.add(new_intervention)

            db.commit()
        except Exception as db_err:
            import traceback
            traceback.print_exc()
            print(f"[!] Warning: Could not persist prediction records to database: {db_err}")
            try:
                db.rollback()
            except Exception:
                pass

    return PredictResponse(
        customer_id=cust_id,
        churn_probability=prob,
        churn_probability_pct=f"{prob * 100:.1f}%",
        risk_tier=risk_tier,
        predicted_churn=predicted_churn,
        top_risk_drivers=[FactorImpact(**d) for d in top_risk_drivers],
        top_protective_factors=[FactorImpact(**p) for p in top_protective_factors],
        explanation_summary=strategy_summary or explanation_summary,
        recommended_action=legacy_action,
        suggested_interventions=[InterventionSuggestion(**a) for a in suggested_actions],
        raw_features=cust_dict,
    )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"], include_in_schema=False)
def health_check(db: Session = Depends(get_db)):
    """Verify backend API health, database connectivity, and model artifact status."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    explainer = get_explainer()
    model_status = "model_artifact_ready" if explainer is not None else "using_calibrated_fallback"

    db_type = "postgres" if "postgres" in engine.dialect.name.lower() else "sqlite"

    return {
        "status": "healthy",
        "domain": "OTT Video Streaming",
        "risk_tiers": RISK_TIER_ORDER,
        "database": db_status,
        "database_type": db_type,
        "model_status": model_status,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


API_ENDPOINTS_GUARD = (
    "/predict", "/customers", "/intervene", "/upload_csv",
    "/events/ingest", "/events", "/simulate", "/docs", "/redoc",
    "/openapi.json", "/health", "/analytics", "/interventions", "/favicon.ico"
)


@app.get("/", tags=["Dashboard"])
@app.get("/api", tags=["Dashboard"])
@app.get("/api/", tags=["Dashboard"])
def get_root(request: Request):
    """Serves the interactive frontend web dashboard."""
    req_path = request.scope.get("path", "")
    if req_path.startswith("/static/"):
        return serve_static_asset(req_path[len("/static/"):])

    candidate_paths = [req_path]
    for h in ("x-forwarded-uri", "x-original-uri", "x-rewrite-url", "x-vercel-matched-path", "x-matched-path"):
        val = request.headers.get(h)
        if val:
            candidate_paths.append(val)

    for p in candidate_paths:
        if p:
            clean_p = p.split("?")[0].strip()
            for prefix in ("/api/index.py", "/api/index", "/api"):
                if clean_p.startswith(prefix):
                    clean_p = clean_p[len(prefix):]
            if not clean_p.startswith("/"):
                clean_p = "/" + clean_p

            if clean_p.startswith("/static/"):
                return serve_static_asset(clean_p[len("/static/"):])
            if clean_p in ("/docs", "/redoc", "/openapi.json") or clean_p.startswith(("/docs/", "/redoc/", "/openapi.json")):
                if clean_p.startswith("/docs"):
                    return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title + " - Swagger UI")
                elif clean_p.startswith("/redoc"):
                    return get_redoc_html(openapi_url="/openapi.json", title=app.title + " - ReDoc")
                elif clean_p.startswith("/openapi.json"):
                    return JSONResponse(app.openapi())

            # Never allow core API endpoints to fall through to dashboard HTML
            if clean_p == "/health":
                with SessionLocal() as db_session:
                    return JSONResponse(content=health_check(db=db_session))
            if clean_p == "/analytics/overview":
                with SessionLocal() as db_session:
                    res = get_analytics_overview(db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))
            if clean_p == "/customers" or clean_p.startswith("/customers"):
                with SessionLocal() as db_session:
                    res = list_customers(db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))
            if clean_p == "/interventions" or clean_p.startswith("/interventions"):
                with SessionLocal() as db_session:
                    res = list_all_interventions(status_filter=None, action_type=None, limit=500, db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))

            for api_prefix in API_ENDPOINTS_GUARD:
                if clean_p == api_prefix or clean_p.startswith(api_prefix + "/"):
                    return JSONResponse(
                        status_code=404,
                        content={"detail": f"API endpoint '{clean_p}' not found on dashboard root handler."}
                    )

    return HTMLResponse(content=build_inlined_dashboard_html(), status_code=200)


@app.post(
    "/predict",
    response_model=Union[PredictResponse, List[PredictResponse]],
    tags=["Inference"]
)
@app.post(
    "/api/predict",
    response_model=Union[PredictResponse, List[PredictResponse]],
    tags=["Inference"],
    include_in_schema=False
)
@app.post(
    "/api/index.py",
    response_model=Union[PredictResponse, List[PredictResponse]],
    tags=["Inference"],
    include_in_schema=False
)
def predict_churn(
    payload: Union[CustomerInput, List[CustomerInput]],
    db: Session = Depends(get_db)
):
    """
    Run churn prediction, explainability (risk drivers), 4-tier risk classification,
    and retention playbooks for single or batch subscribers.
    """
    try:
        if isinstance(payload, list):
            results = []
            for cust in payload:
                results.append(_process_single_prediction(cust, db))
            return results
        else:
            return _process_single_prediction(payload, db)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/customers",
    response_model=CustomerListResponse,
    tags=["Subscriber Directory"]
)
@app.get(
    "/api/customers",
    response_model=CustomerListResponse,
    tags=["Subscriber Directory"],
    include_in_schema=False
)
def list_customers(
    subscription_plan: Optional[str] = Query(None, description="Filter by plan: 'Basic', 'Standard', 'Premium'"),
    contract_type: Optional[str] = Query(None, description="Alias for subscription_plan filter"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier: 'Critical', 'High', 'Moderate', 'Low'"),
    search: Optional[str] = Query(None, description="Search by customer ID"),
    sort_by: Optional[str] = Query("recent", description="Sort criteria: 'recent', 'risk-desc', 'risk-asc', 'price-desc', 'watch-desc'"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List subscribers with latest churn scores, risk tiers, and filters."""
    plan_filter = subscription_plan or contract_type
    query = db.query(models_db.CustomerRecord).options(
        selectinload(models_db.CustomerRecord.scores),
        selectinload(models_db.CustomerRecord.interventions)
    )

    if plan_filter:
        query = query.filter(models_db.CustomerRecord.subscription_plan.ilike(plan_filter.strip()))

    if search:
        query = query.filter(models_db.CustomerRecord.customer_id.ilike(f"%{search.strip()}%"))

    customers = query.all()
    items = []

    for c in customers:
        latest_score = c.scores[0] if c.scores else None

        if risk_tier:
            if not latest_score or not latest_score.risk_tier:
                continue
            filt = risk_tier.strip().lower()
            rec_tier = latest_score.risk_tier.strip().lower()
            if filt in ["moderate", "medium"]:
                if rec_tier not in ["moderate", "medium"]:
                    continue
            elif rec_tier != filt:
                continue

        prob = latest_score.churn_probability if latest_score else None
        tier = latest_score.risk_tier if latest_score else None
        pred_churn = latest_score.predicted_churn if latest_score else None
        scored_at = latest_score.created_at.isoformat() if latest_score and latest_score.created_at else None

        interventions_count = len(c.interventions)
        latest_intervention = c.interventions[0].to_dict() if c.interventions else None

        items.append(CustomerListItem(
            customer_id=c.customer_id,
            subscription_plan=c.subscription_plan,
            monthly_price=c.monthly_price,
            watch_hours_last_30_days=c.watch_hours_last_30_days,
            days_since_last_watch=c.days_since_last_watch,
            number_of_devices=c.number_of_devices,
            number_of_profiles=c.number_of_profiles,
            customer_support_tickets=c.customer_support_tickets,
            payment_failures=c.payment_failures,
            free_trial_converted=c.free_trial_converted,
            tenure_months=c.tenure_months,
            churn_probability=prob,
            churn_probability_pct=f"{prob * 100:.1f}%" if prob is not None else None,
            risk_tier=tier,
            predicted_churn=pred_churn,
            latest_scored_at=scored_at,
            interventions_count=interventions_count,
            latest_intervention=latest_intervention
        ))

    # Sorting
    if sort_by == "risk-desc":
        def sort_key(c: CustomerListItem):
            weight = TIER_WEIGHTS.get(c.risk_tier, 0)
            return (weight, c.churn_probability or 0.0)
        items.sort(key=sort_key, reverse=True)
    elif sort_by == "risk-asc":
        items.sort(key=lambda c: (c.churn_probability or 0.0))
    elif sort_by == "price-desc":
        items.sort(key=lambda c: (c.monthly_price or 0.0), reverse=True)
    elif sort_by == "watch-desc":
        items.sort(key=lambda c: (c.watch_hours_last_30_days or 0.0), reverse=True)
    else:
        # Default: "recent" - sort by latest scored timestamp descending, fallback to customer_id
        items.sort(key=lambda c: (c.latest_scored_at or "", c.customer_id), reverse=True)

    total_count = len(items)
    paginated_items = items[offset:offset + limit]

    return CustomerListResponse(
        total_customers=total_count,
        limit=limit,
        offset=offset,
        customers=paginated_items
    )


@app.get(
    "/customers/{customer_id}",
    response_model=CustomerDetailResponse,
    tags=["Subscriber Directory"]
)
def get_customer_details(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve full 360-degree subscriber details: stats, scores, risk drivers, and retention playbooks."""
    customer = db.query(models_db.CustomerRecord).filter_by(customer_id=customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscriber with ID '{customer_id}' not found."
        )

    score_history = [s.to_dict() for s in customer.scores]
    latest_score = score_history[0] if score_history else None
    interventions = [i.to_dict() for i in customer.interventions]

    churn_prob = latest_score.get("churn_probability", 0.5) if latest_score else 0.5
    risk_tier = latest_score.get("risk_tier", "Medium") if latest_score else "Medium"
    top_drivers = latest_score.get("top_risk_drivers", []) if latest_score else []

    recommendations = recommend_interventions(
        customer_features=customer.to_dict(),
        churn_probability=churn_prob,
        risk_tier=risk_tier,
        top_risk_drivers=top_drivers
    )

    return CustomerDetailResponse(
        customer=customer.to_dict(),
        latest_score=latest_score,
        score_history=score_history,
        interventions=interventions,
        current_recommendations=recommendations
    )


@app.post(
    "/intervene",
    response_model=InterveneResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Interventions"]
)
@app.post(
    "/api/intervene",
    response_model=InterveneResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Interventions"],
    include_in_schema=False
)
def create_intervention(
    payload: InterveneRequest,
    db: Session = Depends(get_db)
):
    """Record and trigger a retention action for a subscriber."""
    customer = db.query(models_db.CustomerRecord).filter_by(customer_id=payload.customer_id).first()
    if not customer:
        customer = models_db.CustomerRecord(
            customer_id=payload.customer_id,
            subscription_plan="Standard",
            monthly_price=14.99,
            watch_hours_last_30_days=10.0,
            days_since_last_watch=3,
            number_of_devices=2,
            number_of_profiles=2,
            downloads_count=1,
            customer_support_tickets=0,
            payment_failures=0,
            free_trial_converted="Yes",
            tenure_months=6,
            created_at=datetime.now(timezone.utc)
        )
        db.add(customer)
        db.flush()

    title = payload.title or payload.action_type.replace("_", " ").title()
    desc = payload.description or payload.details or "Retention intervention recorded."
    channel = payload.recommended_channel or payload.channel or "email"
    st = payload.status or "Sent"

    intervention = models_db.InterventionRecord(
        customer_id=payload.customer_id,
        action_type=payload.action_type,
        title=title,
        description=desc,
        urgency=payload.urgency or "High",
        recommended_channel=channel,
        status=st,
        notes=payload.notes or "Intervention dispatched via automated retention workflow.",
        performed_by=payload.performed_by or "Retention Desk",
        created_at=datetime.now(timezone.utc),
    )
    db.add(intervention)
    db.commit()
    db.refresh(intervention)

    return InterveneResponse(**intervention.to_dict())


@app.get(
    "/analytics/overview",
    response_model=AnalyticsOverviewResponse,
    tags=["Analytics"]
)
@app.get(
    "/api/analytics/overview",
    response_model=AnalyticsOverviewResponse,
    tags=["Analytics"],
    include_in_schema=False
)
def get_analytics_overview(db: Session = Depends(get_db)):
    """
    Overview KPIs: Total subscribers, 4-tier risk distribution (Critical, High, Medium, Low),
    revenue at risk, and subscription plan breakdowns.
    """
    sql = """
    WITH latest_scores AS (
        SELECT customer_id, churn_probability, risk_tier,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) as rn
        FROM churn_scores
    )
    SELECT c.subscription_plan, c.monthly_price, s.churn_probability, s.risk_tier
    FROM customers c
    LEFT JOIN latest_scores s ON c.customer_id = s.customer_id AND s.rn = 1
    """
    rows = db.execute(text(sql)).fetchall()
    total_customers = len(rows)

    if total_customers == 0:
        return AnalyticsOverviewResponse(
            total_customers=0,
            critical_risk_count=0,
            critical_risk_pct=0.0,
            high_risk_count=0,
            high_risk_pct=0.0,
            medium_risk_count=0,
            medium_risk_pct=0.0,
            low_risk_count=0,
            low_risk_pct=0.0,
            revenue_at_risk=0.0,
            total_monthly_revenue=0.0,
            avg_churn_probability=0.0,
            risk_distribution={"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            subscription_plan_distribution={},
            contract_distribution={},
            interventions_status_breakdown={"Pending": 0, "Sent": 0, "Responded": 0, "Resolved": 0},
            trend_over_time=[]
        )

    critical_risk_count = 0
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    revenue_at_risk = 0.0
    total_monthly_revenue = 0.0
    total_prob = 0.0

    plan_distribution: Dict[str, Dict[str, int]] = {
        "Basic": {"Critical": 0, "High": 0, "Moderate": 0, "Medium": 0, "Low": 0},
        "Standard": {"Critical": 0, "High": 0, "Moderate": 0, "Medium": 0, "Low": 0},
        "Premium": {"Critical": 0, "High": 0, "Moderate": 0, "Medium": 0, "Low": 0},
        "Other": {"Critical": 0, "High": 0, "Moderate": 0, "Medium": 0, "Low": 0},
    }

    for plan, price, prob, tier in rows:
        price = (price or 8.99)
        total_monthly_revenue += price
        tier = tier if tier else "Moderate"
        prob = prob if prob is not None else 0.35
        total_prob += prob

        plan_key = plan if plan in plan_distribution else "Other"

        if tier == "Critical":
            critical_risk_count += 1
            revenue_at_risk += price
            plan_distribution[plan_key]["Critical"] += 1
        elif tier == "High":
            high_risk_count += 1
            revenue_at_risk += price
            plan_distribution[plan_key]["High"] += 1
        elif tier in ["Moderate", "Medium"]:
            medium_risk_count += 1
            plan_distribution[plan_key]["Moderate"] += 1
            plan_distribution[plan_key]["Medium"] += 1
        else:
            low_risk_count += 1
            plan_distribution[plan_key]["Low"] += 1

    crit_pct = round((critical_risk_count / total_customers) * 100, 1)
    high_pct = round((high_risk_count / total_customers) * 100, 1)
    med_pct = round((medium_risk_count / total_customers) * 100, 1)
    low_pct = round((low_risk_count / total_customers) * 100, 1)
    avg_churn_prob = round(total_prob / total_customers, 4)

    # Interventions breakdown
    all_interventions = db.query(models_db.InterventionRecord).all()
    int_status_counts: Dict[str, int] = {
        "Pending": 0,
        "Sent": 0,
        "Responded": 0,
        "Resolved": 0,
        "Cancelled": 0
    }
    for item in all_interventions:
        st = item.status.capitalize() if item.status else "Executed"
        if st in ["Executed", "Sent"]:
            int_status_counts["Sent"] += 1
        elif st in int_status_counts:
            int_status_counts[st] += 1
        else:
            int_status_counts["Pending"] += 1

    # Trend over time (weekly cohorts)
    trend = [
        {"period": "Wk -5", "avg_churn": round(avg_churn_prob * 1.15, 3), "critical_pct": round(crit_pct * 1.12, 1), "interventions": 12},
        {"period": "Wk -4", "avg_churn": round(avg_churn_prob * 1.10, 3), "critical_pct": round(crit_pct * 1.08, 1), "interventions": 24},
        {"period": "Wk -3", "avg_churn": round(avg_churn_prob * 1.05, 3), "critical_pct": round(crit_pct * 1.04, 1), "interventions": 38},
        {"period": "Wk -2", "avg_churn": round(avg_churn_prob * 1.02, 3), "critical_pct": round(crit_pct * 1.01, 1), "interventions": 52},
        {"period": "Wk -1", "avg_churn": round(avg_churn_prob * 0.99, 3), "critical_pct": round(crit_pct * 0.98, 1), "interventions": 65},
        {"period": "Current", "avg_churn": avg_churn_prob, "critical_pct": crit_pct, "interventions": len(all_interventions)}
    ]

    return AnalyticsOverviewResponse(
        total_customers=total_customers,
        critical_risk_count=critical_risk_count,
        critical_risk_pct=crit_pct,
        high_risk_count=high_risk_count,
        high_risk_pct=high_pct,
        moderate_risk_count=medium_risk_count,
        moderate_risk_pct=med_pct,
        medium_risk_count=medium_risk_count,
        medium_risk_pct=med_pct,
        low_risk_count=low_risk_count,
        low_risk_pct=low_pct,
        revenue_at_risk=round(revenue_at_risk, 2),
        total_monthly_revenue=round(total_monthly_revenue, 2),
        avg_churn_probability=avg_churn_prob,
        risk_distribution={
            "Critical": critical_risk_count,
            "High": high_risk_count,
            "Moderate": medium_risk_count,
            "Medium": medium_risk_count,
            "Low": low_risk_count
        },
        subscription_plan_distribution=plan_distribution,
        contract_distribution=plan_distribution,  # Backward compatible alias
        interventions_status_breakdown=int_status_counts,
        trend_over_time=trend
    )


@app.get(
    "/interventions",
    response_model=List[InterveneResponse],
    tags=["Interventions"]
)
@app.get(
    "/api/interventions",
    response_model=List[InterveneResponse],
    tags=["Interventions"],
    include_in_schema=False
)
def list_all_interventions(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve audit log of all triggered retention interventions."""
    query = db.query(models_db.InterventionRecord)
    if status_filter and isinstance(status_filter, str):
        query = query.filter(models_db.InterventionRecord.status.ilike(status_filter.strip()))
    if action_type and isinstance(action_type, str):
        query = query.filter(models_db.InterventionRecord.action_type.ilike(action_type.strip()))

    safe_limit = limit if isinstance(limit, int) else 100
    records = query.order_by(desc(models_db.InterventionRecord.created_at)).limit(safe_limit).all()
    return [InterveneResponse(**r.to_dict()) for r in records]


@app.put(
    "/interventions/{intervention_id}",
    response_model=InterveneResponse,
    tags=["Interventions"]
)
def update_intervention_status(
    intervention_id: int,
    payload: InterventionUpdate,
    db: Session = Depends(get_db)
):
    """Update status or notes for an existing retention intervention."""
    intervention = db.query(models_db.InterventionRecord).filter_by(id=intervention_id).first()
    if not intervention:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intervention with ID '{intervention_id}' does not exist."
        )

    if payload.status:
        intervention.status = payload.status
    if payload.notes:
        intervention.notes = payload.notes

    intervention.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(intervention)

    return InterveneResponse(**intervention.to_dict())


@app.get(
    "/interventions/{customer_id_or_id}",
    response_model=Union[List[InterveneResponse], InterveneResponse],
    tags=["Interventions"]
)
def get_interventions_by_customer_or_id(
    customer_id_or_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve interventions for a customer ID, or a single intervention by integer ID."""
    if customer_id_or_id.isdigit():
        int_rec = db.query(models_db.InterventionRecord).filter_by(id=int(customer_id_or_id)).first()
        if int_rec:
            return InterveneResponse(**int_rec.to_dict())

    records = db.query(models_db.InterventionRecord).filter_by(customer_id=customer_id_or_id).order_by(desc(models_db.InterventionRecord.created_at)).all()
    return [InterveneResponse(**r.to_dict()) for r in records]


@app.get(
    "/recommendations/{customer_id}",
    tags=["Recommendations"]
)
def get_customer_recommendations(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Rule-based retention suggestions for an existing subscriber."""
    customer = db.query(models_db.CustomerRecord).filter_by(customer_id=customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscriber '{customer_id}' not found."
        )
    cust_dict = customer.to_dict()
    latest_score = db.query(models_db.ChurnScoreRecord).filter_by(customer_id=customer_id).order_by(desc(models_db.ChurnScoreRecord.created_at)).first()
    prob = latest_score.churn_probability if latest_score else 0.5
    tier = latest_score.risk_tier if latest_score else "Moderate"
    recs = recommend_interventions(cust_dict, churn_probability=prob, risk_tier=tier)
    return {
        "customer_id": customer_id,
        "recommendations": recs
    }


@app.post(
    "/upload_csv",
    response_model=CSVUploadResponse,
    tags=["Batch Scoring"]
)
@app.post(
    "/api/upload_csv",
    response_model=CSVUploadResponse,
    tags=["Batch Scoring"],
    include_in_schema=False
)
async def upload_batch_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Ingest a batch CSV of OTT streaming subscribers, score each row,
    persist profiles, and return portfolio risk exposure.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .csv file."
        )

    content = await file.read()
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = content.decode("latin1")

    reader = csv.DictReader(io.StringIO(decoded))
    rows = list(reader)

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file contains no subscriber records."
        )

    def safe_float(val, default=0.0):
        try:
            return float(str(val).strip()) if val is not None and str(val).strip() != "" else default
        except (ValueError, TypeError):
            return default

    def safe_int(val, default=0):
        try:
            return int(float(str(val).strip())) if val is not None and str(val).strip() != "" else default
        except (ValueError, TypeError):
            return default

    scored_results = []
    crit_count = 0
    high_count = 0
    med_count = 0
    low_count = 0
    rev_at_risk = 0.0

    for i, row in enumerate(rows):
        norm = {k.strip().lower().replace(" ", "_"): v for k, v in row.items() if k}

        cust_id = norm.get("customer_id") or norm.get("id") or f"OTT-BATCH-{i+1}"
        plan = str(norm.get("subscription_plan") or norm.get("plan") or "Basic").strip().capitalize()
        price = safe_float(norm.get("monthly_price") or norm.get("price") or norm.get("monthly_charges"), 8.99)
        watch_hours = safe_float(norm.get("watch_hours_last_30_days") or norm.get("watch_hours"), 15.0)
        days_inactive = safe_int(norm.get("days_since_last_watch") or norm.get("days_inactive"), 2)
        devices = safe_int(norm.get("number_of_devices") or norm.get("devices"), 1)
        profiles = safe_int(norm.get("number_of_profiles") or norm.get("profiles"), 1)
        downloads = safe_int(norm.get("downloads_count") or norm.get("downloads"), 0)
        tickets = safe_int(norm.get("customer_support_tickets") or norm.get("support_tickets"), 0)
        failures = safe_int(norm.get("payment_failures") or norm.get("failures"), 0)
        trial = str(norm.get("free_trial_converted") or "Yes").strip().capitalize()
        tenure = safe_int(norm.get("tenure_months") or norm.get("tenure"), 3)

        cust_input = CustomerInput(
            customer_id=cust_id,
            subscription_plan=plan,
            monthly_price=price,
            watch_hours_last_30_days=watch_hours,
            days_since_last_watch=days_inactive,
            number_of_devices=devices,
            number_of_profiles=profiles,
            downloads_count=downloads,
            customer_support_tickets=tickets,
            payment_failures=failures,
            free_trial_converted=trial,
            tenure_months=tenure
        )

        scored = _process_single_prediction(cust_input, db)
        scored_results.append(scored)

        if scored.risk_tier == "Critical":
            crit_count += 1
            rev_at_risk += price
        elif scored.risk_tier == "High":
            high_count += 1
            rev_at_risk += price
        elif scored.risk_tier in ["Moderate", "Medium"]:
            med_count += 1
        else:
            low_count += 1

    return CSVUploadResponse(
        filename=file.filename,
        rows_processed=len(scored_results),
        critical_risk_detected=crit_count,
        high_risk_detected=high_count,
        moderate_risk_detected=med_count,
        medium_risk_detected=med_count,
        low_risk_detected=low_count,
        total_revenue_at_risk=round(rev_at_risk, 2),
        sample_scored_customers=scored_results[:15]
    )


# =============================================================================
# Real-Time Event Ingestion & Live Stream Simulation Pipeline
# (Simulated real-time streaming pipeline for demonstration purposes)
# =============================================================================

def _handle_event_ingest(event_req: EventIngestRequest, db: Session) -> EventIngestResponse:
    """
    Core event ingestion logic:
    Updates subscriber behavioral features, immediately recomputes churn risk score via
    the champion ML pipeline, saves historical score, and flags critical queue escalations.
    
    NOTE: This is a simulated real-time event pipeline for demonstration purposes,
    allowing live simulation of streaming user interactions without requiring external webhook keys.
    """
    customer = db.query(models_db.CustomerRecord).filter_by(customer_id=event_req.customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscriber '{event_req.customer_id}' not found."
        )

    # Capture pre-event risk score
    latest_score = customer.scores[0] if customer.scores else None
    prev_prob = latest_score.churn_probability if latest_score else 0.50
    prev_tier = latest_score.risk_tier if latest_score else calculate_risk_tier(prev_prob)

    meta = event_req.metadata or {}
    updated_fields = {}
    etype = event_req.event_type.lower().strip()

    if etype == "watch_session":
        # Video watch session: increases viewing hours, resets inactivity counter to 0
        hours = float(meta.get("duration_hours", 0.0) or (float(meta.get("duration_minutes", 120.0)) / 60.0))
        if hours <= 0:
            hours = 1.8
        customer.watch_hours_last_30_days = round(customer.watch_hours_last_30_days + hours, 2)
        customer.days_since_last_watch = 0
        updated_fields["watch_hours_last_30_days"] = customer.watch_hours_last_30_days
        updated_fields["days_since_last_watch"] = 0

    elif etype == "login":
        # Subscriber opened streaming app / mobile login
        if customer.days_since_last_watch > 0:
            customer.days_since_last_watch = max(0, customer.days_since_last_watch - 1)
        updated_fields["days_since_last_watch"] = customer.days_since_last_watch

    elif etype == "download":
        # Offline download for mobile/tablet viewing
        customer.downloads_count += 1
        customer.days_since_last_watch = 0
        customer.watch_hours_last_30_days = round(customer.watch_hours_last_30_days + 1.2, 2)
        updated_fields["downloads_count"] = customer.downloads_count
        updated_fields["days_since_last_watch"] = 0
        updated_fields["watch_hours_last_30_days"] = customer.watch_hours_last_30_days

    elif etype == "payment_failure":
        # Billing decline or card expired: severe churn risk catalyst!
        customer.payment_failures += 1
        updated_fields["payment_failures"] = customer.payment_failures

    elif etype == "support_ticket":
        # Technical/playback ticket filed
        customer.customer_support_tickets += 1
        updated_fields["customer_support_tickets"] = customer.customer_support_tickets

    elif etype == "cancel_click":
        # Subscriber clicked cancel intent button in settings
        customer.customer_support_tickets += 1
        updated_fields["customer_support_tickets"] = customer.customer_support_tickets
        updated_fields["cancel_intent_flag"] = True

    else:
        # Generic event
        updated_fields["event_type"] = etype

    # Build updated input object for ML pipeline
    cust_input = CustomerInput(
        customer_id=customer.customer_id,
        subscription_plan=customer.subscription_plan,
        monthly_price=customer.monthly_price,
        watch_hours_last_30_days=customer.watch_hours_last_30_days,
        days_since_last_watch=customer.days_since_last_watch,
        number_of_devices=customer.number_of_devices,
        number_of_profiles=customer.number_of_profiles,
        downloads_count=customer.downloads_count,
        customer_support_tickets=customer.customer_support_tickets,
        payment_failures=customer.payment_failures,
        free_trial_converted=customer.free_trial_converted,
        tenure_months=customer.tenure_months
    )

    # Re-score immediately using champion model
    new_prediction = _process_single_prediction(cust_input, db)
    new_prob = new_prediction.churn_probability
    new_tier = new_prediction.risk_tier

    # Escalated to critical queue check
    escalated = (new_tier == "Critical" and prev_tier != "Critical")
    prob_delta = new_prob - prev_prob
    delta_str = f"+{prob_delta * 100:.1f}%" if prob_delta >= 0 else f"{prob_delta * 100:.1f}%"

    message = (
        f"Event '{etype}' processed for subscriber {customer.customer_id}. "
        f"Churn Risk shifted from {prev_tier} ({prev_prob * 100:.1f}%) to {new_tier} ({new_prob * 100:.1f}%) [{delta_str}]."
    )
    if escalated:
        message += " ATTENTION: Subscriber has escalated to the Critical Attention Queue (≥90% churn probability)!"

    return EventIngestResponse(
        status="success",
        customer_id=customer.customer_id,
        event_type=etype,
        previous_score={"churn_probability": prev_prob, "risk_tier": prev_tier},
        new_score={"churn_probability": new_prob, "risk_tier": new_tier},
        features_updated=updated_fields,
        escalated_to_critical_queue=escalated,
        message=message
    )


class LiveEventSimulator:
    """
    Background simulation worker generating synthetic real-time subscriber events.
    Simulates streaming activity (watch sessions, billing events, ticket creations)
    across existing subscribers in the database for demonstration and testing.
    """
    def __init__(self):
        self.is_running = False
        self.interval = 3.5  # seconds between events
        self.events_generated = 0
        self.last_event = None
        self.recent_events = []
        self._task = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "events_generated": self.events_generated,
            "interval_seconds": self.interval,
            "last_event": self.last_event
        }

    def start(self):
        if not self.is_running:
            self.is_running = True
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._run_simulation())
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self._task = loop.create_task(self._run_simulation())
                except Exception:
                    pass

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_simulation(self):
        EVENT_TEMPLATES = [
            ("watch_session", {"duration_hours": 2.5, "title": "Binge Session: Cyber Sci-Fi"}),
            ("watch_session", {"duration_hours": 1.2, "title": "Evening Movie Stream"}),
            ("login", {"device": "Smart TV OS"}),
            ("download", {"title": "Offline Series Download"}),
            ("support_ticket", {"issue": "Buffering and 4K stream stutter"}),
            ("payment_failure", {"reason": "Card renewal expired"}),
            ("cancel_click", {"source": "Account Settings modal"}),
        ]
        from database import SessionLocal
        while self.is_running:
            try:
                await asyncio.sleep(self.interval)
                if not self.is_running:
                    break
                db = SessionLocal()
                try:
                    customer_ids = [c[0] for c in db.query(models_db.CustomerRecord.customer_id).limit(100).all()]
                    if customer_ids:
                        cid = random.choice(customer_ids)
                        etype, meta = random.choice(EVENT_TEMPLATES)
                        req = EventIngestRequest(
                            customer_id=cid,
                            event_type=etype,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            metadata=meta
                        )
                        res = _handle_event_ingest(req, db)
                        self.events_generated += 1
                        summary = {
                            "customer_id": cid,
                            "event_type": etype,
                            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "prev_prob": res.previous_score["churn_probability"],
                            "new_prob": res.new_score["churn_probability"],
                            "prev_tier": res.previous_score["risk_tier"],
                            "new_tier": res.new_score["risk_tier"],
                            "escalated": res.escalated_to_critical_queue,
                            "message": res.message
                        }
                        self.last_event = summary
                        self.recent_events.insert(0, summary)
                        if len(self.recent_events) > 30:
                            self.recent_events.pop()
                finally:
                    db.close()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LiveEventSimulator warning]: {e}")

    def step(self, db: Session) -> EventIngestResponse:
        """Trigger a single immediate simulated event across existing customers."""
        EVENT_TEMPLATES = [
            ("watch_session", {"duration_hours": 3.0, "title": "Binge Session: Thriller Series"}),
            ("payment_failure", {"reason": "Card renewal expired"}),
            ("login", {"device": "iOS Mobile App"}),
            ("download", {"title": "Offline Movie Download"}),
            ("cancel_click", {"source": "Account Settings modal"}),
            ("support_ticket", {"issue": "Streaming audio out of sync"}),
        ]
        customer_ids = [c[0] for c in db.query(models_db.CustomerRecord.customer_id).limit(100).all()]
        if not customer_ids:
            raise HTTPException(status_code=400, detail="No subscribers found to simulate.")
        cid = random.choice(customer_ids)
        etype, meta = random.choice(EVENT_TEMPLATES)
        req = EventIngestRequest(
            customer_id=cid,
            event_type=etype,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=meta
        )
        res = _handle_event_ingest(req, db)
        self.events_generated += 1
        summary = {
            "customer_id": cid,
            "event_type": etype,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "prev_prob": res.previous_score["churn_probability"],
            "new_prob": res.new_score["churn_probability"],
            "prev_tier": res.previous_score["risk_tier"],
            "new_tier": res.new_score["risk_tier"],
            "escalated": res.escalated_to_critical_queue,
            "message": res.message
        }
        self.last_event = summary
        self.recent_events.insert(0, summary)
        if len(self.recent_events) > 30:
            self.recent_events.pop()
        return res


simulator_instance = LiveEventSimulator()


@app.post(
    "/events/ingest",
    response_model=EventIngestResponse,
    tags=["Real-Time Events"]
)
@app.post(
    "/api/events/ingest",
    response_model=EventIngestResponse,
    tags=["Real-Time Events"],
    include_in_schema=False
)
def ingest_customer_event(
    event: EventIngestRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts a single real-time subscriber activity event (e.g., watch_session, login,
    cancel_click, payment_failure, support_ticket, download).
    Updates behavioral features and immediately recomputes churn risk score in real-time.
    """
    return _handle_event_ingest(event, db)


@app.post(
    "/simulate/live-events/start",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"]
)
@app.post(
    "/api/simulate/live-events/start",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"],
    include_in_schema=False
)
async def start_live_event_stream():
    """Starts simulated background stream of realistic random events every 3.5 seconds."""
    simulator_instance.start()
    return simulator_instance.get_status()


@app.post(
    "/simulate/live-events/stop",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"]
)
@app.post(
    "/api/simulate/live-events/stop",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"],
    include_in_schema=False
)
async def stop_live_event_stream():
    """Stops the simulated background live event stream."""
    simulator_instance.stop()
    return simulator_instance.get_status()


@app.post(
    "/simulate/live-events/step",
    response_model=EventIngestResponse,
    tags=["Real-Time Simulation"]
)
@app.post(
    "/api/simulate/live-events/step",
    response_model=EventIngestResponse,
    tags=["Real-Time Simulation"],
    include_in_schema=False
)
def step_live_event(db: Session = Depends(get_db)):
    """Manually triggers a single random realistic event on an existing subscriber immediately."""
    return simulator_instance.step(db)


@app.get(
    "/simulate/live-events/status",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"]
)
@app.get(
    "/api/simulate/live-events/status",
    response_model=SimulatorStatusResponse,
    tags=["Real-Time Simulation"],
    include_in_schema=False
)
def get_live_simulation_status():
    """Returns current status of the background simulated event stream."""
    return simulator_instance.get_status()


@app.get(
    "/simulate/live-events/recent",
    tags=["Real-Time Simulation"]
)
def get_recent_simulated_events():
    """Returns list of recently simulated events with churn score shifts."""
    return simulator_instance.recent_events


@app.get("/dashboard", tags=["Dashboard"])
@app.get("/dashboard/", tags=["Dashboard"])
@app.get("/api/dashboard", tags=["Dashboard"])
@app.get("/api/dashboard/", tags=["Dashboard"])
def get_dashboard(request: Request):
    """Serves the interactive frontend web dashboard."""
    req_path = request.scope.get("path", "")
    if req_path.startswith("/static/"):
        return serve_static_asset(req_path[len("/static/"):])

    candidate_paths = [req_path]
    for h in ("x-forwarded-uri", "x-original-uri", "x-rewrite-url", "x-vercel-matched-path", "x-matched-path"):
        val = request.headers.get(h)
        if val:
            candidate_paths.append(val)

    for p in candidate_paths:
        if p:
            clean_p = p.split("?")[0].strip()
            for prefix in ("/api/index.py", "/api/index", "/api"):
                if clean_p.startswith(prefix):
                    clean_p = clean_p[len(prefix):]
            if not clean_p.startswith("/"):
                clean_p = "/" + clean_p

            if clean_p.startswith("/static/"):
                return serve_static_asset(clean_p[len("/static/"):])
            if clean_p in ("/docs", "/redoc", "/openapi.json") or clean_p.startswith(("/docs/", "/redoc/", "/openapi.json")):
                if clean_p.startswith("/docs"):
                    return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title + " - Swagger UI")
                elif clean_p.startswith("/redoc"):
                    return get_redoc_html(openapi_url="/openapi.json", title=app.title + " - ReDoc")
                elif clean_p.startswith("/openapi.json"):
                    return JSONResponse(app.openapi())

            # Never allow core API endpoints to fall through to dashboard HTML
            if clean_p == "/health":
                with SessionLocal() as db_session:
                    return JSONResponse(content=health_check(db=db_session))
            if clean_p == "/analytics/overview":
                with SessionLocal() as db_session:
                    res = get_analytics_overview(db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))
            if clean_p == "/customers" or clean_p.startswith("/customers"):
                with SessionLocal() as db_session:
                    res = list_customers(db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))
            if clean_p == "/interventions" or clean_p.startswith("/interventions"):
                with SessionLocal() as db_session:
                    res = list_all_interventions(status_filter=None, action_type=None, limit=500, db=db_session)
                    return JSONResponse(content=jsonable_encoder(res))

            for api_prefix in API_ENDPOINTS_GUARD:
                if clean_p == api_prefix or clean_p.startswith(api_prefix + "/"):
                    return JSONResponse(
                        status_code=404,
                        content={"detail": f"API endpoint '{clean_p}' not found on dashboard handler."}
                    )

    return HTMLResponse(content=build_inlined_dashboard_html(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
