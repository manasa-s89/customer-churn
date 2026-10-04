"""
OTT Subscriber Churn Model Training & Evaluation Pipeline
- Trains and compares Logistic Regression, Random Forest, and XGBoost
- 5-Fold Stratified Cross-Validation
- Evaluates ROC-AUC, F1-Score, Precision, Recall, Accuracy
- Selects the champion model based on validation metrics
- Serializes champion model, preprocessor, and metadata to models/
"""

import json
import os
import sys
import warnings
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from data_pipeline import (
    build_preprocessor,
    clean_raw_data,
    get_feature_names,
    prepare_features_and_target,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from generate_data import generate_ott_churn_dataset

warnings.filterwarnings("ignore")

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def load_or_generate_ott_data(data_path: str = "data/ott_customer_churn.csv") -> pd.DataFrame:
    """Load existing dataset or generate realistic synthetic OTT streaming dataset."""
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    if os.path.exists(data_path):
        print(f"[*] Loading OTT dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("[*] Generating realistic synthetic OTT streaming dataset (7,500 subscribers)...")
        df = generate_ott_churn_dataset(n_samples=7500, random_state=42)
        df.to_csv(data_path, index=False)
        print(f"[OK] Saved generated dataset to {data_path}")
    return df


def get_candidate_models(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """Define candidate algorithms to benchmark."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    else:
        print("[!] xgboost not installed. Using Gradient Boosting as alternative.", flush=True)
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42
        )

    return models


def cross_validate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Perform Stratified K-Fold Cross Validation.
    Fits preprocessor strictly on train fold to prevent leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\n{'='*70}")
    print(f"RUNNING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION ACROSS OTT CANDIDATES")
    print(f"{'='*70}")

    for model_name, model in models.items():
        roc_aucs = []
        f1_scores = []
        precisions = []
        recalls = []
        accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit preprocessor on training fold
            preprocessor = build_preprocessor()
            X_tr_proc = preprocessor.fit_transform(X_tr)
            X_val_proc = preprocessor.transform(X_val)

            # Fit model
            model.fit(X_tr_proc, y_tr)

            # Predict
            y_pred_proba = model.predict_proba(X_val_proc)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_val, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_val, y_pred))

        mean_roc = float(np.mean(roc_aucs))
        std_roc = float(np.std(roc_aucs))
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        mean_prec = float(np.mean(precisions))
        mean_rec = float(np.mean(recalls))
        mean_acc = float(np.mean(accuracies))

        cv_results[model_name] = {
            "cv_roc_auc_mean": mean_roc,
            "cv_roc_auc_std": std_roc,
            "cv_f1_mean": mean_f1,
            "cv_f1_std": std_f1,
            "cv_precision_mean": mean_prec,
            "cv_recall_mean": mean_rec,
            "cv_accuracy_mean": mean_acc,
            "selection_score": 0.5 * mean_roc + 0.5 * mean_f1
        }

        print(f"--> {model_name:<20} | ROC-AUC: {mean_roc:.4f} (±{std_roc:.4f}) | F1: {mean_f1:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | Acc: {mean_acc:.4f}", flush=True)

    return cv_results


def train_and_evaluate(models_dir: str = "models", data_path: str = "data/ott_customer_churn.csv") -> Tuple[str, Dict[str, Any]]:
    """Complete end-to-end training, benchmarking, and serialization."""
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    df = load_or_generate_ott_data(data_path)
    X, y, _ = prepare_features_and_target(df)

    print(f"\n[*] Dataset Shape: {X.shape[0]} rows, {X.shape[1]} features")
    churn_rate = y.mean() * 100
    print(f"[*] Churn Distribution: {churn_rate:.1f}% positive (churned), {100 - churn_rate:.1f}% negative")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 3. Model Comparison via 5-Fold Cross Validation
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / max(pos_count, 1)

    candidate_models = get_candidate_models(scale_pos_weight=scale_pos)
    cv_metrics = cross_validate_models(candidate_models, X_train, y_train, n_splits=5)

    # 4. Select Champion Model
    best_name = max(cv_metrics.keys(), key=lambda m: cv_metrics[m]["selection_score"])
    print(f"\n{'='*70}")
    print(f">>> CHAMPION MODEL SELECTED: {best_name} <<<")
    print(f"{'='*70}")

    # 5. Fit Preprocessing Pipeline & Champion Model on Entire Train Set
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    champion_model = candidate_models[best_name]
    champion_model.fit(X_train_proc, y_train)

    # 6. Evaluate Champion on Holdout Test Set
    y_test_proba = champion_model.predict_proba(X_test_proc)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    test_auc = float(roc_auc_score(y_test, y_test_proba))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    print(f"\n[HOLDOUT TEST METRICS - {best_name}]")
    print(f"  * ROC-AUC Score : {test_auc:.4f}")
    print(f"  * F1-Score      : {test_f1:.4f}")
    print(f"  * Precision     : {test_prec:.4f}")
    print(f"  * Recall        : {test_rec:.4f}")
    print(f"  * Accuracy      : {test_acc:.4f}")
    print(f"  * Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 7. Create End-to-End Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", champion_model)
    ])

    # 8. Feature Importance / Coefficients
    feat_importances = {}
    if hasattr(champion_model, "feature_importances_"):
        for f, imp in zip(feature_names, champion_model.feature_importances_):
            feat_importances[f] = float(imp)
    elif hasattr(champion_model, "coef_"):
        for f, coef in zip(feature_names, champion_model.coef_[0]):
            feat_importances[f] = float(abs(coef))

    top_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n[TOP 10 KEY RISK DRIVERS (MODEL WEIGHTS)]")
    for feat, imp in top_feats:
        print(f"  - {feat:<35}: {imp:.4f}")

    # 9. Save Artifacts
    preprocessor_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    model_path = os.path.join(models_dir, "best_model.joblib")
    pipeline_path = os.path.join(models_dir, "full_churn_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(champion_model, model_path)
    joblib.dump(full_pipeline, pipeline_path)

    metadata = {
        "domain": "OTT / Video Streaming",
        "champion_model_name": best_name,
        "test_metrics": {
            "roc_auc": test_auc,
            "f1": test_f1,
            "precision": test_prec,
            "recall": test_rec,
            "accuracy": test_acc,
            "confusion_matrix": cm
        },
        "cv_results": cv_metrics,
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "top_feature_importances": top_feats
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[OK] Artifacts successfully serialized to '{models_dir}/'")
    return best_name, metadata


if __name__ == "__main__":
    best_model, meta = train_and_evaluate()
