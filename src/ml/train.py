import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.loader import load_and_join_datasets, initialize_database
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import MODEL_PATH, EXPLAINER_PATH, MODELS_DIR
from src.utils.logger import logger

def train_credit_risk_model(test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Any, Dict[str, float]]:
    """
    Trains credit risk classifier with class imbalance handling, fits SHAP explainer,
    and serializes model artifacts.
    """
    logger.info("Initializing database and loading merged datasets...")
    initialize_database()
    raw_df = load_and_join_datasets()
    
    logger.info("Preprocessing features and target...")
    preprocessor = CreditDataPreprocessor()
    X, y = preprocessor.fit_transform(raw_df, target_col="TARGET")
    preprocessor.save()

    logger.info(f"Class distribution: Non-Default (0)={sum(y==0)}, Default (1)={sum(y==1)} " 
                f"(Default rate: {sum(y==1)/len(y):.2%})")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model handling class imbalance
    logger.info("Training HistGradientBoostingClassifier with balanced class weights...")
    model = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, y_pred_prob))
    pr_auc = float(average_precision_score(y_test, y_pred_prob))
    gini = float(2 * roc_auc - 1)

    logger.info(f"Model Training Results: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, Gini={gini:.4f}")

    # Fit SHAP Explainer using background sample
    logger.info("Fitting SHAP explainer...")
    sample_background = X_train.sample(n=min(300, len(X_train)), random_state=random_state)
    
    # Use Tree/Kernel Explainer for probability predictions
    explainer = shap.Explainer(model.predict_proba, sample_background)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact_payload = {
        'model': model,
        'feature_names': preprocessor.feature_names,
        'metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'gini': gini
        }
    }
    
    joblib.dump(artifact_payload, MODEL_PATH)
    logger.info(f"Model payload saved to {MODEL_PATH}")

    joblib.dump({
        'explainer': explainer,
        'background_sample': sample_background
    }, EXPLAINER_PATH)
    logger.info(f"SHAP explainer saved to {EXPLAINER_PATH}")

    return model, explainer, artifact_payload['metrics']

if __name__ == "__main__":
    train_credit_risk_model()
