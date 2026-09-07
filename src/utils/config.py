import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCUMENTS_DIR = BASE_DIR / "documents"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SQL_DIR = BASE_DIR / "sql"

# Create directories if they don't exist
for folder in [DATA_DIR, MODELS_DIR, DOCUMENTS_DIR, NOTEBOOKS_DIR, SQL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database and Model Paths
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "credit_risk.db"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "credit_risk_model.joblib"))
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

# API Keys & LLM Settings
def _get_api_key():
    # 1. Check Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
            if "gemini_api_key" in st.secrets:
                return st.secrets["gemini_api_key"]
    except Exception:
        pass
    # 2. Check environment variables (.env or system env)
    return os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""

GEMINI_API_KEY = _get_api_key()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# Risk Scoring Thresholds
LOW_RISK_THRESHOLD = 0.15     # prob < 0.15 -> Low Risk (Score 750-1000)
MEDIUM_RISK_THRESHOLD = 0.35  # 0.15 <= prob < 0.35 -> Medium Risk (Score 500-749)
                              # prob >= 0.35 -> High Risk (Score 0-499)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
