import sqlite3
import pandas as pd
import numpy as np
import joblib
import json

from predict_and_explain import ChurnExplainer

explainer = ChurnExplainer(models_dir="models")
print("=== MODEL INFO ===")
print("Model classes_:", explainer.model.classes_)
if hasattr(explainer.model, "coef_"):
    print("Model coef_ shape:", explainer.model.coef_.shape)
    print("Model intercept_:", explainer.model.intercept_)
    for f, c in zip(explainer.feature_names, explainer.model.coef_[0]):
        print(f"  {f:32s}: {c:+.4f}")

conn = sqlite3.connect("churn.db")
df_all = pd.read_sql("SELECT * FROM customers", conn)
print(f"\nTotal customers in DB: {len(df_all)}")

# Check latest scores in DB
c = conn.cursor()
c.execute("""
    SELECT c.customer_id, s.churn_probability, s.risk_tier, s.created_at
    FROM customers c
    LEFT JOIN churn_scores s ON s.id = (
        SELECT id FROM churn_scores WHERE customer_id = c.customer_id ORDER BY created_at DESC LIMIT 1
    )
""")
stored_scores = c.fetchall()
print(f"Total customers with joined latest score: {len(stored_scores)}")

probs = [s[1] for s in stored_scores if s[1] is not None]
tiers = [s[2] for s in stored_scores if s[2] is not None]

print("\n=== CURRENT STORED SCORES IN DB ===")
print(f"Min stored prob: {min(probs):.4f}, Max: {max(probs):.4f}, Mean: {np.mean(probs):.4f}, Median: {np.median(probs):.4f}")
print("Stored risk tier counts:", pd.Series(tiers).value_counts().to_dict())

# How many stored scores are >= 0.90 or >= 0.98?
print("Stored >= 0.98:", sum(1 for p in probs if p >= 0.98))
print("Stored >= 0.90:", sum(1 for p in probs if p >= 0.90))
print("Stored >= 0.80:", sum(1 for p in probs if p >= 0.80))
print("Stored >= 0.60 and < 0.80:", sum(1 for p in probs if 0.60 <= p < 0.80))
print("Stored >= 0.30 and < 0.60:", sum(1 for p in probs if 0.30 <= p < 0.60))
print("Stored < 0.30:", sum(1 for p in probs if p < 0.30))

print("\n=== NOW RUNNING ACTUAL MODEL ON ALL CUSTOMERS IN DB ===")
model_preds = explainer.predict_and_explain(df_all)
m_probs = [p["churn_probability"] for p in model_preds]
m_tiers = [p["risk_tier"] for p in model_preds]

print(f"Model Min prob: {min(m_probs):.4f}, Max: {max(m_probs):.4f}, Mean: {np.mean(m_probs):.4f}, Median: {np.median(m_probs):.4f}")
print("Model risk tier counts (with current calculate_risk_tier):", pd.Series(m_tiers).value_counts().to_dict())
print("Model >= 0.98:", sum(1 for p in m_probs if p >= 0.98))
print("Model >= 0.80:", sum(1 for p in m_probs if p >= 0.80))
print("Model 0.60 to 0.80:", sum(1 for p in m_probs if 0.60 <= p < 0.80))
print("Model 0.30 to 0.60:", sum(1 for p in m_probs if 0.30 <= p < 0.60))
print("Model < 0.30:", sum(1 for p in m_probs if p < 0.30))

print("\nSample 10 model predictions:")
for p in model_preds[:10]:
    print(f"  {p['customer_id']}: prob={p['churn_probability']:.4f} ({p['churn_probability_pct']}), risk={p['risk_tier']}")
