import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import joblib

from data_pipeline import prepare_features_and_target, clean_raw_data
from predict_and_explain import ChurnExplainer

# Load dataset
csv_path = "data/ott_customer_churn.csv"
if not os.path.exists(csv_path):
    print("CSV not found!")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Overall churn label mean (base rate): {df['churn_label'].mean():.4f}")

explainer = ChurnExplainer(models_dir="models")
X, y, _ = prepare_features_and_target(df)
X_proc = explainer.preprocessor.transform(X)
probs = explainer.model.predict_proba(X_proc)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y, probs)
brier = brier_score_loss(y, probs)
cm = confusion_matrix(y, preds)

print("\n=== MODEL EVALUATION ON DATASET ===")
print(f"ROC-AUC: {auc:.4f}")
print(f"Brier Score Loss: {brier:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y, preds, target_names=["Retained (0)", "Churned (1)"]))

# Probability distribution across bands
print("\n=== PROBABILITY DISTRIBUTION ===")
print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}, Median: {np.median(probs):.4f}")

p_below_30 = (probs < 0.30).sum()
p_30_60 = ((probs >= 0.30) & (probs < 0.60)).sum()
p_60_80 = ((probs >= 0.60) & (probs < 0.80)).sum()
p_above_80 = (probs >= 0.80).sum()
n = len(probs)

print(f"< 30% (Low Risk):       {p_below_30:5d} ({p_below_30/n*100:.1f}%)")
print(f"30%–60% (Moderate Risk): {p_30_60:5d} ({p_30_60/n*100:.1f}%)")
print(f"60%–80% (High Risk):     {p_60_80:5d} ({p_60_80/n*100:.1f}%)")
print(f">= 80% (Critical Risk):  {p_above_80:5d} ({p_above_80/n*100:.1f}%)")

# Calibration curve
prob_true, prob_pred = calibration_curve(y, probs, n_bins=5)
print("\n=== CALIBRATION (True churn rate vs Predicted prob) ===")
for pt, pp in zip(prob_true, prob_pred):
    print(f"  Predicted ~ {pp*100:5.1f}% -> Actual Churn Rate: {pt*100:5.1f}%")
