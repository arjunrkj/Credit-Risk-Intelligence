import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, EXPLAINER_PATH
from src.utils.helpers import calculate_risk_score, get_risk_band, get_risk_color, derive_business_rules
from src.utils.logger import logger

class CreditRiskPredictor:
    """Inference and scoring engine for credit risk assessment."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, preprocessor, and SHAP explainer."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}. Run training pipeline first.")
            
        logger.info("Loading ML model and preprocessor artifacts...")
        payload = joblib.load(MODEL_PATH)
        self.model = payload['model']
        self.feature_names = payload['feature_names']
        
        self.preprocessor = CreditDataPreprocessor.load()
        
        if EXPLAINER_PATH.exists():
            explainer_payload = joblib.load(EXPLAINER_PATH)
            self.explainer = explainer_payload['explainer']

    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts credit risk score, probability, risk band, SHAP values, 
        and derived business rules for a single loan applicant.
        """
        df = pd.DataFrame([applicant_data])
        X_processed = self.preprocessor.transform(df)
        
        # Default probability prediction
        prob = float(self.model.predict_proba(X_processed)[0, 1])
        score = calculate_risk_score(prob)
        band = get_risk_band(prob)
        color = get_risk_color(band)
        
        # Calculate SHAP values for local explainability
        shap_values_dict = {}
        if self.explainer is not None:
            try:
                # Get SHAP values for class 1 (default)
                shap_raw = self.explainer(X_processed)
                if hasattr(shap_raw, 'values'):
                    vals = shap_raw.values[0]
                    if len(vals.shape) == 2: # Binary output [N, 2]
                        vals = vals[:, 1]
                    shap_values_dict = dict(zip(self.feature_names, vals.tolist()))
            except Exception as e:
                logger.warning(f"SHAP calculation exception: {e}")
                
        # Generate feature raw values dict
        raw_feature_dict = dict(zip(self.feature_names, X_processed.iloc[0].tolist()))
        
        # Derive human-readable business rules
        decision_rules = derive_business_rules(raw_feature_dict, shap_values_dict)
        
        return {
            'default_probability': prob,
            'risk_score': score,
            'risk_band': band,
            'risk_color': color,
            'shap_values': shap_values_dict,
            'decision_rules': decision_rules,
            'feature_values': raw_feature_dict
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring function returning risk predictions for a DataFrame."""
        X_processed = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_processed)[:, 1]
        
        results_df = df.copy()
        results_df['DEFAULT_PROBABILITY'] = np.round(probs, 4)
        results_df['RISK_SCORE'] = [calculate_risk_score(p) for p in probs]
        results_df['RISK_BAND'] = [get_risk_band(p) for p in probs]
        
        return results_df
