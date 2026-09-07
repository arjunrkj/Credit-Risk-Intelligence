# AI-Powered Credit Risk Intelligence Platform - Implementation Plan

Building an end-to-end AI-powered credit risk platform using the Home Credit Default Risk dataset structure. The solution covers Exploratory Data Analysis (EDA), machine learning default prediction with class imbalance handling, Explainable AI (SHAP) and decision rules, a conversational Talk-to-Data (NL-to-SQL) system powered by LLM, an interactive multi-tab Streamlit user interface, Docker deployment, and complete project documentation including an executive PDF presentation.

## User Review Required

> [!IMPORTANT]
> **LLM API Key Configuration**: The Talk-to-Data system supports Google Gemini via `GEMINI_API_KEY`. It also includes an intelligent offline rule-based SQL generator fallback so that the chatbot operates seamlessly even without an API key provided.

> [!NOTE]
> **Dataset Ingestion**: To ensure the repository runs out-of-the-box without requiring manual downloads, a synthetic dataset generator script will populate `data/application_train.csv`, `data/bureau.csv`, and `data/previous_application.csv` following the authentic Home Credit schema. Real Kaggle dataset files can also be dropped directly into `data/`.

## Open Questions

None at present. All requirements, architecture patterns, and directory structures have been fully specified in `NeoStats_AI_Use_Case.md` and the user-provided code structure tree.

---

## Proposed Changes

### Project Root & Configuration

#### [NEW] [requirements.txt](file:///c:/neostats/requirements.txt)
- Specifies required Python packages: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, `streamlit`, `google-genai`, `reportlab`, `plotly`, `sqlalchemy`, `python-dotenv`, `joblib`, `matplotlib`, `seaborn`.

#### [NEW] [.env.example](file:///c:/neostats/.env.example)
- Documents environment variables (`GEMINI_API_KEY`, `DB_PATH`, `MODEL_PATH`, `LOG_LEVEL`).

#### [NEW] [.gitignore](file:///c:/neostats/.gitignore)
- Standard python, dataset, model artifact, bytecode, and environment ignore file.

#### [NEW] [Dockerfile](file:///c:/neostats/Dockerfile)
- Python 3.10 container setup exposing port 8501 for Streamlit.

#### [NEW] [docker-compose.yml](file:///c:/neostats/docker-compose.yml)
- Docker Compose service definition mounting `./data` and exposing port 8501.

#### [NEW] [README.md](file:///c:/neostats/README.md)
- Complete technical documentation, architecture diagram (Mermaid), setup instructions, ML rationale, evaluation metrics, prompt engineering strategy, and decision rules.

---

### Core Python Modules (`src/`)

#### [NEW] [src/utils/config.py](file:///c:/neostats/src/utils/config.py)
- Configuration manager for paths, database connections, model thresholds, and environment variables.

#### [NEW] [src/utils/logger.py](file:///c:/neostats/src/utils/logger.py)
- Centralized logger configuration with file and stream logging handlers.

#### [NEW] [src/utils/helpers.py](file:///c:/neostats/src/utils/helpers.py)
- Helper utilities for metric formatting, risk score calibration (0-1000 scale), risk band categorization, and business rule evaluation.

#### [NEW] [src/utils/docker_utils.py](file:///c:/neostats/src/utils/docker_utils.py)
- Docker path resolution utilities to handle containerized vs. local execution environments.

#### [NEW] [src/data/loader.py](file:///c:/neostats/src/data/loader.py)
- Loads CSV datasets from `data/`, executes relational joins across application, bureau, and previous application tables, and initializes the SQLite database `credit_risk.db`.

#### [NEW] [src/data/preprocessor.py](file:///c:/neostats/src/data/preprocessor.py)
- Data cleaning, missing value handling, categorical encoding, and domain feature engineering (Debt-to-Income, Credit-to-Annuity, Employment ratio, Bureau delinquency indicators).

#### [NEW] [src/ml/train.py](file:///c:/neostats/src/ml/train.py)
- ML training pipeline with class imbalance handling (`scale_pos_weight` / SMOTE), model hyperparameter tuning, model artifact saving to `models/model.joblib`.

#### [NEW] [src/ml/predict.py](file:///c:/neostats/src/ml/predict.py)
- Real-time and batch inference engine returning default probability, risk score, and risk band (Low / Medium / High).

#### [NEW] [src/ml/evaluate.py](file:///c:/neostats/src/ml/evaluate.py)
- Model evaluation module calculating ROC-AUC, PR-AUC, Gini coefficient, Confusion Matrix, and metric summary JSONs.

#### [NEW] [src/talk_to_data/prompt_templates.py](file:///c:/neostats/src/talk_to_data/prompt_templates.py)
- Versioned system prompts for natural language to SQL translation, including table schema definitions, SQL guidelines, safety filters, and response synthesis instructions.

#### [NEW] [src/talk_to_data/nl_to_sql.py](file:///c:/neostats/src/talk_to_data/nl_to_sql.py)
- NL-to-SQL translator integration using Google Gemini API with fallback smart SQL translation engine for offline operation.

#### [NEW] [src/talk_to_data/query_runner.py](file:///c:/neostats/src/talk_to_data/query_runner.py)
- Safe SQL execution against SQLite, validation, Pandas formatting, and natural-language business insight generation.

---

### Database & Artifacts

#### [NEW] [sql/schema.sql](file:///c:/neostats/sql/schema.sql)
- SQL DDL schemas for Home Credit tables (`application_train`, `bureau`, `previous_application`) and indexes.

#### [NEW] [models/README.md](file:///c:/neostats/models/README.md)
- Placeholder and documentation for serialized `.joblib` model artifacts.

---

### User Interface & Analytics

#### [NEW] [app.py](file:///c:/neostats/app.py)
- Modern multi-tab Streamlit dashboard containing:
  1. Executive Summary & Dataset Overview
  2. Exploratory Data Analysis (EDA) & Charts
  3. Applicant Credit Risk Scoring & Batch Assessment
  4. Explainable AI (SHAP Waterfall/Bar & Derived Decision Rules)
  5. Model Performance & Evaluation Metrics (ROC, PR, Confusion Matrix)
  6. Conversational Talk-to-Data Chatbot Interface

#### [NEW] [notebooks/eda.ipynb](file:///c:/neostats/notebooks/eda.ipynb)
- Jupyter notebook containing full Exploratory Data Analysis on the dataset.

#### [NEW] [notebooks/eda.py](file:///c:/neostats/notebooks/eda.py)
- Converted python script version of `eda.ipynb`.

---

### Documentation & Deliverables

#### [NEW] [documents/project_presentation.pdf](file:///c:/neostats/documents/project_presentation.pdf)
- Executive PDF Presentation summarizing the project scope, business impact, architecture, EDA, ML performance, SHAP explainability, decision rules, Talk-to-Data system, and setup instructions.

#### [NEW] [scripts/generate_synthetic_data.py](file:///c:/neostats/scripts/generate_synthetic_data.py)
- Generator script for creating realistic Home Credit sample datasets in `data/`.

#### [NEW] [scripts/generate_pdf.py](file:///c:/neostats/scripts/generate_pdf.py)
- ReportLab-based script to programmatically build the executive PDF presentation in `documents/project_presentation.pdf`.

---

## Verification Plan

### Automated Tests
1. **Pipeline & Data Ingestion Verification**: Run data loading and preprocessing scripts to confirm feature generation and SQLite database initialization.
2. **Model Training & Evaluation**: Execute `python -m src.ml.train` and `python -m src.ml.evaluate` to confirm model training, serialization to `models/model.joblib`, and metric computation (ROC-AUC > 0.70).
3. **Talk-to-Data Verification**: Run `python -m src.talk_to_data.query_runner` to test NL-to-SQL translation, SQL execution against SQLite DB, and response synthesis.
4. **PDF Presentation Verification**: Execute `python scripts/generate_pdf.py` and verify `documents/project_presentation.pdf` is generated properly.

### Manual Verification
1. Launch Streamlit application using `streamlit run app.py`.
2. Test interactive features in UI:
   - Browse EDA charts and statistics.
   - Enter applicant parameters and trigger live Risk Scoring & SHAP explanation.
   - Test derived Business Decision Rules view.
   - Run sample natural language queries in the Talk-to-Data Chatbot tab (e.g. "What is the default rate by income type?", "Show average credit amount for default vs non-default").
