"""
Data Preprocessing Pipeline for OTT Streaming Churn Prediction
Handles missing values, categorical encoding, and numeric scaling for OTT subscriber data.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# OTT Streaming Numeric Features
NUMERIC_FEATURES = [
    'monthly_price',
    'watch_hours_last_30_days',
    'days_since_last_watch',
    'number_of_devices',
    'number_of_profiles',
    'downloads_count',
    'customer_support_tickets',
    'payment_failures',
    'tenure_months'
]

# OTT Streaming Categorical Features
CATEGORICAL_FEATURES = [
    'subscription_plan',
    'free_trial_converted'
]

TARGET_COLUMN = 'churn_label'
ID_COLUMN = 'customer_id'


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw input OTT DataFrame:
    - Standardizes column names (lowercase, strips spaces, replace whitespace with _)
    - Coerces numeric features safely
    - Normalizes categorical string values
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip().lower().replace(" ", "_") for c in df_clean.columns]

    # Clean numeric features
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Normalize categorical columns
    if 'subscription_plan' in df_clean.columns:
        df_clean['subscription_plan'] = (
            df_clean['subscription_plan']
            .astype(str)
            .str.strip()
            .str.capitalize()
            .replace({'Nan': 'Basic', 'None': 'Basic', '': 'Basic'})
        )

    if 'free_trial_converted' in df_clean.columns:
        df_clean['free_trial_converted'] = (
            df_clean['free_trial_converted']
            .astype(str)
            .str.strip()
            .str.capitalize()
            .replace({'1': 'Yes', '0': 'No', 'True': 'Yes', 'False': 'No', 'Nan': 'No', '': 'No'})
        )

    # Ensure target column is integer 0/1 if present
    if TARGET_COLUMN in df_clean.columns:
        if not pd.api.types.is_integer_dtype(df_clean[TARGET_COLUMN]):
            mapped = df_clean[TARGET_COLUMN].astype(str).str.strip().str.lower().map({
                'yes': 1, '1': 1, 'true': 1, 't': 1,
                'no': 0, '0': 0, 'false': 0, 'f': 0
            })
            df_clean[TARGET_COLUMN] = pd.to_numeric(mapped, errors='coerce').fillna(0).astype(int)
        else:
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(int)

    return df_clean


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer preprocessing pipeline:
    - Numeric: Median Imputation -> Standard Scaling
    - Categorical: Most Frequent Imputation -> One-Hot Encoding (handle_unknown='ignore')
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract transformed feature names after fitting preprocessor."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'remainder' or trans == 'drop':
                continue
            if hasattr(trans, 'get_feature_names_out'):
                names.extend(list(trans.get_feature_names_out(cols)))
            elif hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                onehot = trans.named_steps['onehot']
                names.extend(list(onehot.get_feature_names_out(cols)))
            else:
                names.extend(cols)
        return names


def prepare_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Clean data and split into features (X), target (y), and customer_ids."""
    df_clean = clean_raw_data(df)

    customer_ids = df_clean[ID_COLUMN] if ID_COLUMN in df_clean.columns else None
    y = df_clean[TARGET_COLUMN] if TARGET_COLUMN in df_clean.columns else None

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df_clean.columns]
    X = df_clean[feature_cols]

    return X, y, customer_ids
