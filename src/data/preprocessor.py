import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.config import PREPROCESSOR_PATH
from src.utils.logger import logger

class CreditDataPreprocessor:
    """
    Data cleaning, feature engineering, categorical encoding, and scaling pipeline 
    for Home Credit Default Risk dataset.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies domain-specific financial and risk feature transformations."""
        df = df.copy()
        
        # 1. Financial ratios
        df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1.0)
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1.0)
        df['PAYMENT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1.0)
        df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1.0)

        # 2. Demographics & Employment ratios
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365.25
        else:
            df['AGE_YEARS'] = 40.0

        if 'DAYS_EMPLOYED' in df.columns:
            # Clean anomalous employment days (365243 in Home Credit dataset represents retired/unemployed)
            df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['DAYS_EMPLOYED_YEARS'] = np.abs(df['DAYS_EMPLOYED_CLEAN'].fillna(0)) / 365.25
            df['EMPLOYMENT_TO_AGE_RATIO'] = df['DAYS_EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1e-5)
        else:
            df['DAYS_EMPLOYED_YEARS'] = 5.0
            df['EMPLOYMENT_TO_AGE_RATIO'] = 0.12

        # 3. Aggregated Bureau & External score indicators
        ext_cols = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if col in df.columns]
        if ext_cols:
            df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
            df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
            df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1).fillna(0)
        else:
            df['EXT_SOURCES_MEAN'] = 0.5
            df['EXT_SOURCES_MIN'] = 0.5
            df['EXT_SOURCES_STD'] = 0.0

        # Fill potential bureau merged feature missing values with defaults if present
        for col, default_val in [
            ('TOTAL_BUREAU_LOANS', 0),
            ('ACTIVE_BUREAU_LOANS', 0),
            ('TOTAL_BUREAU_DEBT', 0),
            ('PREV_APPLICATIONS_COUNT', 0),
            ('PREV_REFUSED_COUNT', 0)
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
            else:
                df[col] = default_val

        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = "TARGET") -> Tuple[pd.DataFrame, pd.Series]:
        """Fits preprocessor transformers and transforms dataset into feature matrix X and target y."""
        logger.info("Fitting CreditDataPreprocessor...")
        
        df_processed = self.engineer_features(df)
        
        # Extract target if present
        if target_col in df_processed.columns:
            y = df_processed[target_col].copy()
            df_processed = df_processed.drop(columns=[target_col])
        else:
            y = None
            
        # Drop ID columns
        ignore_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']
        feature_df = df_processed.drop(columns=[col for col in ignore_cols if col in df_processed.columns])
        
        # Categorize column types
        self.categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        
        # Impute numerical missing values with median
        for col in self.numerical_cols:
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
        # Encode categorical variables
        for col in self.categorical_cols:
            feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            self.label_encoders[col] = le
            
        self.feature_names = feature_df.columns.tolist()
        
        # Scale numerical features
        scaled_features = self.scaler.fit_transform(feature_df[self.numerical_cols])
        feature_df[self.numerical_cols] = scaled_features
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Transformed {feature_df.shape[1]} features.")
        
        return feature_df, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms unseen applicant data using fitted encoders and scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
            
        df_processed = self.engineer_features(df)
        
        # Select features
        ignore_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'TARGET']
        feature_df = df_processed.drop(columns=[col for col in ignore_cols if col in df_processed.columns], errors='ignore')
        
        # Ensure all fitted features exist
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
                
        feature_df = feature_df[self.feature_names].copy()
        
        # Impute missing numericals
        for col in self.numerical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0.0)
                
        # Encode categoricals
        for col in self.categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna("Unknown").astype(str)
                le = self.label_encoders[col]
                # Map unseen categories to closest/default index
                classes = list(le.classes_)
                feature_df[col] = feature_df[col].apply(lambda val: le.transform([val])[0] if val in classes else 0)

        # Scale numerical features
        scaled_features = self.scaler.transform(feature_df[self.numerical_cols])
        feature_df[self.numerical_cols] = scaled_features
        
        return feature_df

    def save(self, filepath: Path = PREPROCESSOR_PATH):
        """Serializes fitted preprocessor object."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = PREPROCESSOR_PATH) -> 'CreditDataPreprocessor':
        """Loads serialized preprocessor object."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)
