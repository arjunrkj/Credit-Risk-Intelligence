import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from src.data.loader import load_and_join_datasets
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, MODELS_DIR
from src.utils.logger import logger

def evaluate_model() -> Dict[str, Any]:
    """Evaluates model performance metrics and saves summary to models/evaluation_results.json."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model payload missing. Train model first.")
        
    logger.info("Evaluating ML model performance...")
    payload = joblib.load(MODEL_PATH)
    model = payload['model']
    
    raw_df = load_and_join_datasets()
    preprocessor = CreditDataPreprocessor.load()
    X, y = preprocessor.fit_transform(raw_df, target_col="TARGET")
    
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_prob >= 0.35).astype(int)
    
    roc_auc = float(roc_auc_score(y, y_pred_prob))
    pr_auc = float(average_precision_score(y, y_pred_prob))
    gini = float(2 * roc_auc - 1)
    
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, output_dict=True)
    
    results = {
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'gini_index': round(gini, 4),
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    out_path = MODELS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Evaluation completed. Metrics saved to {out_path}")
    logger.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Gini: {gini:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_model()
