import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from src.utils.config import PREPROCESSOR_PATH
from src.utils.logger import logger


class CreditDataPreprocessor:
    """
    Enriched feature engineering pipeline for Home Credit Default Risk.
    LightGBM handles missing values natively, so we skip scaling and
    only label-encode categoricals. We focus on creating high-signal
    domain features that the Kaggle community found most predictive.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_cols: List[str] = []
        self.feature_names: List[str] = []
        self.medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── 1. External credit bureau scores (highest Kaggle predictive value) ──
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
        if ext_cols:
            df['EXT_SOURCES_MEAN']    = df[ext_cols].mean(axis=1)
            df['EXT_SOURCES_MIN']     = df[ext_cols].min(axis=1)
            df['EXT_SOURCES_MAX']     = df[ext_cols].max(axis=1)
            df['EXT_SOURCES_STD']     = df[ext_cols].std(axis=1)
            df['EXT_SOURCES_PROD']    = df[ext_cols].prod(axis=1)
            if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns:
                df['EXT_12_DIFF']     = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
            if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
                df['EXT_23_DIFF']     = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']

        # ── 2. Age & employment ──
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS']           = -df['DAYS_BIRTH'] / 365.25
            df['AGE_YEARS_SQ']        = df['AGE_YEARS'] ** 2

        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_FLAG']  = (df['DAYS_EMPLOYED'] == 365243).astype(int)  # anomaly → retired/unemployed
            df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['EMPLOYED_YEARS']      = -df['DAYS_EMPLOYED_CLEAN'] / 365.25
            if 'DAYS_BIRTH' in df.columns:
                df['EMPLOYMENT_TO_AGE'] = df['DAYS_EMPLOYED_CLEAN'] / (df['DAYS_BIRTH'] + 0.001)

        if 'DAYS_REGISTRATION' in df.columns:
            df['DAYS_REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365.25

        if 'DAYS_ID_PUBLISH' in df.columns:
            df['DAYS_ID_YEARS']       = -df['DAYS_ID_PUBLISH'] / 365.25

        # ── 3. Financial ratios ──
        df['CREDIT_TO_INCOME']        = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_TO_INCOME']       = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_TO_CREDIT']       = df['AMT_ANNUITY'] / (df['AMT_CREDIT']        + 1)
        df['CREDIT_TERM_MONTHS']      = df['AMT_CREDIT']  / (df['AMT_ANNUITY']       + 1)
        df['INCOME_PER_CHILD']        = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'].fillna(0) + 1)
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['INCOME_PER_FAM']      = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'].fillna(1) + 1)

        if 'AMT_GOODS_PRICE' in df.columns:
            df['GOODS_TO_CREDIT']     = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
            df['CREDIT_OVER_GOODS']   = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']

        # ── 4. Social circle risk ──
        for t in [30, 60]:
            obs = f'OBS_{t}_CNT_SOCIAL_CIRCLE'
            dflt = f'DEF_{t}_CNT_SOCIAL_CIRCLE'
            if obs in df.columns and dflt in df.columns:
                df[f'SOCIAL_DEF_RATE_{t}'] = df[dflt] / (df[obs] + 1)

        # ── 5. Bureau aggregates (if joined) ──
        for col, default_val in [
            ('TOTAL_BUREAU_LOANS',     0),
            ('ACTIVE_BUREAU_LOANS',    0),
            ('TOTAL_BUREAU_DEBT',      0),
            ('MAX_OVERDUE',            0),
            ('DAYS_CREDIT_MIN',        0),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
            else:
                df[col] = default_val

        if 'TOTAL_BUREAU_DEBT' in df.columns:
            df['BUREAU_DEBT_TO_INCOME'] = df['TOTAL_BUREAU_DEBT'] / (df['AMT_INCOME_TOTAL'] + 1)
            df['BUREAU_ACTIVE_RATIO']   = df['ACTIVE_BUREAU_LOANS'] / (df['TOTAL_BUREAU_LOANS'] + 1)

        # ── 6. Previous application aggregates (if joined) ──
        for col, default_val in [
            ('PREV_APPLICATIONS_COUNT', 0),
            ('PREV_REFUSED_COUNT',      0),
            ('PREV_APPROVED_COUNT',     0),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
            else:
                df[col] = default_val

        df['PREV_REFUSAL_RATE'] = df['PREV_REFUSED_COUNT'] / (df['PREV_APPLICATIONS_COUNT'] + 1)

        # ── 7. Document / flag counts ──
        flag_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        if flag_cols:
            df['DOCUMENT_COUNT'] = df[flag_cols].sum(axis=1)

        contact_flags = [c for c in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'] if c in df.columns]
        if contact_flags:
            df['CONTACT_COUNT'] = df[contact_flags].sum(axis=1)

        return df

    # ------------------------------------------------------------------
    # Fit-Transform
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, target_col: str = "TARGET") -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Fitting CreditDataPreprocessor (LightGBM-optimised)...")
        df_processed = self.engineer_features(df)

        y = None
        if target_col in df_processed.columns:
            y = df_processed[target_col].copy()
            df_processed = df_processed.drop(columns=[target_col])

        ignore_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']
        feature_df = df_processed.drop(columns=[c for c in ignore_cols if c in df_processed.columns])

        # Detect column types
        self.categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols        = feature_df.select_dtypes(include=['number']).columns.tolist()

        # Store medians for inference-time imputation of numeric cols
        for col in numerical_cols:
            med = feature_df[col].median()
            self.medians[col] = 0.0 if pd.isna(med) else float(med)

        # Label-encode categoricals
        for col in self.categorical_cols:
            feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            self.label_encoders[col] = le

        self.feature_names = feature_df.columns.tolist()
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Total features: {len(self.feature_names)}")
        return feature_df, y

    # ------------------------------------------------------------------
    # Transform (inference)
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df_processed = self.engineer_features(df)

        ignore_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'TARGET']
        feature_df  = df_processed.drop(columns=[c for c in ignore_cols if c in df_processed.columns], errors='ignore')

        # Ensure all fitted features are present using reindex to prevent fragmentation
        feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0.0)

        # Impute numerics with stored medians
        for col, med in self.medians.items():
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(med)

        # Encode categoricals
        for col in self.categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
                le = self.label_encoders[col]
                known = set(le.classes_)
                feature_df[col] = feature_df[col].apply(
                    lambda v: int(le.transform([v])[0]) if v in known else 0
                )

        return feature_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: Path = PREPROCESSOR_PATH):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = PREPROCESSOR_PATH) -> 'CreditDataPreprocessor':
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)
