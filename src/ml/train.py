import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.loader import load_and_join_datasets, initialize_database
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, EXPLAINER_PATH, MODELS_DIR
from src.utils.logger import logger


# ──────────────────────────────────────────────────────────────────────
# LightGBM parameters tuned for Home Credit Default Risk
# ──────────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           ["auc", "average_precision"],
    "boosting_type":    "gbdt",
    "n_estimators":     1000,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 30,
    "min_child_weight":  1e-3,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    "reg_alpha":         1.0,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

N_FOLDS = 5


def train_credit_risk_model() -> Tuple[Any, Any, Dict[str, float]]:
    """
    5-Fold Stratified cross-validated LightGBM training with early stopping.
    Saves best fold model, SHAP explainer, and evaluation metrics.
    """
    logger.info("Initializing database and loading merged datasets...")
    initialize_database()
    raw_df = load_and_join_datasets()

    logger.info("Preprocessing features...")
    preprocessor = CreditDataPreprocessor()
    X, y = preprocessor.fit_transform(raw_df, target_col="TARGET")
    preprocessor.save()

    logger.info(f"Dataset: {X.shape[0]:,} rows × {X.shape[1]} features | "
                f"Default rate: {y.mean():.2%}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_preds    = np.zeros(len(y))
    fold_aucs    = []
    best_model   = None
    best_auc     = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)

        logger.info(f"Fold {fold}/{N_FOLDS} -> ROC-AUC: {fold_auc:.4f} | Best iter: {model.best_iteration_}")

        if fold_auc > best_auc:
            best_auc   = fold_auc
            best_model = model

    # ── OOF metrics ──────────────────────────────────────────────────
    oof_auc  = float(roc_auc_score(y, oof_preds))
    oof_pr   = float(average_precision_score(y, oof_preds))
    oof_gini = float(2 * oof_auc - 1)

    logger.info("=" * 55)
    logger.info(f"  OOF ROC-AUC : {oof_auc:.4f}")
    logger.info(f"  OOF PR-AUC  : {oof_pr:.4f}")
    logger.info(f"  OOF Gini    : {oof_gini:.4f}")
    logger.info(f"  Fold AUCs   : {[round(a, 4) for a in fold_aucs]}")
    logger.info("=" * 55)

    # ── Save artifacts ────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature importances
    fi = pd.Series(
        best_model.feature_importances_,
        index=preprocessor.feature_names
    ).sort_values(ascending=False)

    metrics = {
        "roc_auc":   round(oof_auc,  4),
        "pr_auc":    round(oof_pr,   4),
        "gini_index": round(oof_gini, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "top_features": fi.head(20).to_dict(),
    }

    payload = {
        "model":         best_model,
        "feature_names": preprocessor.feature_names,
        "metrics":       metrics,
    }
    joblib.dump(payload, MODEL_PATH)
    logger.info(f"Best model saved -> {MODEL_PATH}")

    # SHAP explainer on sample background
    logger.info("Fitting SHAP TreeExplainer...")
    sample_bg = X.sample(n=min(500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(best_model)

    joblib.dump({"explainer": explainer, "background_sample": sample_bg}, EXPLAINER_PATH)
    logger.info(f"SHAP explainer saved -> {EXPLAINER_PATH}")

    # Write evaluation JSON
    eval_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation metrics saved -> {eval_path}")
    return best_model, explainer, metrics


if __name__ == "__main__":
    train_credit_risk_model()
