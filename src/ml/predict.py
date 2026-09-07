import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

import shap

from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, EXPLAINER_PATH
from src.utils.helpers import calculate_risk_score, get_risk_band, get_risk_color, derive_business_rules
from src.utils.logger import logger


class CreditRiskPredictor:
    """Real-time and batch inference engine with SHAP explainability."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}. Run training pipeline first.")

        logger.info("Loading ML model, preprocessor and SHAP explainer...")
        payload = joblib.load(MODEL_PATH)
        self.model         = payload["model"]
        self.feature_names = payload["feature_names"]

        self.preprocessor = CreditDataPreprocessor.load()

        if EXPLAINER_PATH.exists():
            explainer_payload = joblib.load(EXPLAINER_PATH)
            self.explainer = explainer_payload["explainer"]

    # ------------------------------------------------------------------
    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Full prediction + SHAP + decision rules for one applicant."""
        df = pd.DataFrame([applicant_data])
        X_processed = self.preprocessor.transform(df)

        prob  = float(self.model.predict_proba(X_processed)[0, 1])
        score = calculate_risk_score(prob)
        band  = get_risk_band(prob)
        color = get_risk_color(band)

        shap_values_dict = {}
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(X_processed)
                # LightGBM TreeExplainer returns array of shape (n, features)
                # For binary classification it may return list of 2 arrays
                if isinstance(sv, list):
                    sv = sv[1]          # class-1 SHAP values
                if len(sv.shape) == 2:
                    sv = sv[0]          # first (only) row
                shap_values_dict = dict(zip(self.feature_names, sv.tolist()))
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")

        raw_feature_dict = dict(zip(self.feature_names, X_processed.iloc[0].tolist()))
        decision_rules   = derive_business_rules(raw_feature_dict, shap_values_dict)

        return {
            "default_probability": prob,
            "risk_score":          score,
            "risk_band":           band,
            "risk_color":          color,
            "shap_values":         shap_values_dict,
            "decision_rules":      decision_rules,
            "feature_values":      raw_feature_dict,
        }

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring — returns DataFrame with appended risk columns."""
        X_processed = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_processed)[:, 1]

        out = df.copy()
        out["DEFAULT_PROBABILITY"] = np.round(probs, 4)
        out["RISK_SCORE"]          = [calculate_risk_score(p) for p in probs]
        out["RISK_BAND"]           = [get_risk_band(p) for p in probs]
        return out
