import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)
from src.data.loader import load_and_join_datasets
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, MODELS_DIR
from src.utils.logger import logger


from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Finds decision thresholds for Youden's J statistic and Maximum F1-score."""
    # Youden's J
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    idx_j = np.argmax(youden_j)
    thresh_youden = float(thresholds_roc[idx_j])

    # F1 Maximization
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    idx_f1 = np.argmax(f1_scores)
    thresh_f1 = float(thresholds_pr[idx_f1]) if idx_f1 < len(thresholds_pr) else 0.5

    return {
        "f1_optimal": thresh_f1,
        "youden_j": thresh_youden
    }


def evaluate_model() -> Dict[str, Any]:
    """Full evaluation on the complete dataset using the best saved model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model payload missing. Train model first.")

    logger.info("Evaluating ML model performance...")
    payload = joblib.load(MODEL_PATH)
    model   = payload["model"]

    raw_df      = load_and_join_datasets()
    preprocessor = CreditDataPreprocessor.load()
    X, y        = preprocessor.fit_transform(raw_df, target_col="TARGET")

    y_pred_prob = model.predict_proba(X)[:, 1]

    roc_auc = float(roc_auc_score(y, y_pred_prob))
    pr_auc  = float(average_precision_score(y, y_pred_prob))
    gini    = float(2 * roc_auc - 1)

    thresholds_dict = find_optimal_thresholds(np.array(y), y_pred_prob)
    opt_thresh = thresholds_dict["f1_optimal"]
    logger.info(f"Optimal threshold (Max F1): {opt_thresh:.4f} | Youden J: {thresholds_dict['youden_j']:.4f}")

    y_pred = (y_pred_prob >= opt_thresh).astype(int)
    cm     = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, output_dict=True)

    results = {
        "roc_auc":          round(roc_auc, 4),
        "pr_auc":           round(pr_auc,  4),
        "gini_index":       round(gini,    4),
        "optimal_threshold": round(opt_thresh, 4),
        "thresholds":       thresholds_dict,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    preds_path = MODELS_DIR / "eval_preds.joblib"
    joblib.dump({"y_prob": y_pred_prob, "y_true": np.array(y)}, preds_path)

    out_path = MODELS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Gini: {gini:.4f}")
    logger.info(f"Evaluation results saved -> {out_path}")
    return results


if __name__ == "__main__":
    evaluate_model()
