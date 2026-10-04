"""
Unit Tests for OTT Streaming Data Cleaning and Preprocessing Pipeline
Tests data standardization, numeric coercion, missing value handling, categorical encoding,
and feature matrix shapes for OTT streaming telemetry.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

from data_pipeline import (
    clean_raw_data,
    build_preprocessor,
    get_feature_names,
    prepare_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN
)


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        """Create sample raw dataframe mimicking real-world OTT subscriber telemetry."""
        self.raw_data = pd.DataFrame({
            'Customer ID': ['OTT-001', 'OTT-002', 'OTT-003', 'OTT-004'],
            'Subscription Plan': ['Basic', 'Standard', 'Premium', 'basic'],
            'Monthly Price': [8.99, 14.99, 20.99, '8.99'],
            'Watch Hours Last 30 Days': [1.5, 45.0, 72.5, '0.0'],
            'Days Since Last Watch': [28, 2, 0, 35],
            'Number Of Devices': [1, 3, 5, 2],
            'Number Of Profiles': [1, 2, 4, 1],
            'Downloads Count': [0, 6, 12, np.nan],
            'Customer Support Tickets': [3, 0, 1, 4],
            'Payment Failures': [2, 0, 0, 1],
            'Free Trial Converted': ['Yes', 'Yes', 'No', 'False'],
            'Tenure Months': [2, 18, 6, 1],
            'Churn Label': ['Yes', 'No', 'False', '1']
        })

    def test_01_clean_raw_data_columns_and_types(self):
        """Test column standardization and type coercion for OTT telemetry."""
        df_clean = clean_raw_data(self.raw_data)

        # 1. Check standardized column names
        self.assertIn('customer_id', df_clean.columns)
        self.assertIn('subscription_plan', df_clean.columns)
        self.assertIn('monthly_price', df_clean.columns)
        self.assertIn('watch_hours_last_30_days', df_clean.columns)
        self.assertIn('days_since_last_watch', df_clean.columns)
        self.assertIn('churn_label', df_clean.columns)

        # 2. Check numeric coercion
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean['monthly_price']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean['watch_hours_last_30_days']))

        # 3. Check categorical normalization
        self.assertEqual(df_clean.loc[df_clean['customer_id'] == 'OTT-004', 'subscription_plan'].values[0], 'Basic')

        # 4. Check target mapping to integer 0/1
        self.assertTrue(pd.api.types.is_integer_dtype(df_clean['churn_label']))
        self.assertEqual(list(df_clean['churn_label']), [1, 0, 0, 1])

    def test_02_build_preprocessor_structure(self):
        """Test preprocessor assembly with numeric and categorical sub-pipelines."""
        preprocessor = build_preprocessor()
        transformer_names = [name for name, _, _ in preprocessor.transformers]

        self.assertIn('num', transformer_names)
        self.assertIn('cat', transformer_names)

    def test_03_fit_transform_shape(self):
        """Verify feature transformation produces non-empty numeric array with expected columns."""
        df_clean = clean_raw_data(self.raw_data)
        preprocessor = build_preprocessor()

        X_proc = preprocessor.fit_transform(df_clean)
        self.assertIsInstance(X_proc, np.ndarray)
        self.assertEqual(X_proc.shape[0], len(df_clean))
        self.assertGreater(X_proc.shape[1], len(NUMERIC_FEATURES))

        # Check feature names retrieval
        names = get_feature_names(preprocessor)
        self.assertEqual(len(names), X_proc.shape[1])
        self.assertTrue(any('subscription_plan' in f for f in names))

    def test_04_missing_value_imputation(self):
        """Test median and most frequent imputation on missing OTT values."""
        df_missing = self.raw_data.copy()
        df_missing.loc[0, 'Watch Hours Last 30 Days'] = np.nan
        df_missing.loc[1, 'Downloads Count'] = np.nan

        df_clean = clean_raw_data(df_missing)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(df_clean)

        self.assertFalse(np.isnan(X_proc).any(), "Preprocessed matrix must contain zero NaN values")

    def test_05_unseen_category_robustness(self):
        """Verify OneHotEncoder handle_unknown='ignore' handles new subscription plans gracefully."""
        df_clean = clean_raw_data(self.raw_data)
        preprocessor = build_preprocessor()
        preprocessor.fit(df_clean)

        new_data = pd.DataFrame([{
            'subscription_plan': 'VIP_Ultra_4K',  # Unseen category
            'monthly_price': 29.99,
            'watch_hours_last_30_days': 50.0,
            'days_since_last_watch': 1,
            'number_of_devices': 6,
            'number_of_profiles': 5,
            'downloads_count': 10,
            'customer_support_tickets': 0,
            'payment_failures': 0,
            'free_trial_converted': 'Yes',
            'tenure_months': 12
        }])

        X_new_proc = preprocessor.transform(new_data)
        self.assertEqual(X_new_proc.shape[0], 1)
        self.assertFalse(np.isnan(X_new_proc).any())


if __name__ == '__main__':
    unittest.main()
