# 🏦 NeoStats AI - Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk intelligence platform built on the **Home Credit Default Risk** dataset. Designed for commercial banks, credit risk officers, and financial regulators to deliver automated loan risk scoring, explainable AI (SHAP), business decision rule extraction, and conversational natural-language database querying (Talk-to-Data).

---

## 📐 Repository Structure

```
credit_risk_platform/
├── data/                      # Home Credit dataset files & SQLite DB (not committed to git)
├── documents/
│   └── project_presentation.pdf # Executive project presentation PDF
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis Jupyter Notebook
│   └── eda.py                 # Converted EDA Python script
├── src/
│   ├── data/
│   │   ├── loader.py          # Data ingestion, relational joins & DB initialization
│   │   └── preprocessor.py    # Imputation, ratio engineering, label encoding, scaling
│   ├── ml/
│   │   ├── train.py           # ML training pipeline with class imbalance handling
│   │   ├── predict.py         # Real-time scoring, risk score calibration & SHAP values
│   │   └── evaluate.py        # Model evaluation metrics (ROC-AUC, PR-AUC, Gini)
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py       # Natural language to SQL using Gemini LLM & fallback parser
│   │   ├── query_runner.py    # Safe SQL execution & natural language synthesis
│   │   └── prompt_templates.py# Versioned LLM system prompts & schema definitions
│   └── utils/
│       ├── logger.py          # Centralized logging setup
│       ├── config.py          # Environment variables & system configurations
│       ├── helpers.py         # Risk score calibration (0-1000) & decision rule engine
│       └── docker_utils.py    # Docker path resolution utilities
├── sql/
│   └── schema.sql             # Relational SQL DDL schemas & analytical views
├── models/                    # Saved model artifacts (.joblib)
├── app.py                     # Interactive multi-tab Streamlit dashboard
├── Dockerfile                 # Multi-stage Docker container definition
├── docker-compose.yml         # Container service orchestration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # System documentation
```

---

## 🚀 Quickstart Guide

### Option A: Local Run (Python)

1. **Clone Repository & Navigate to Directory**:
   ```bash
   git clone <repository_url>
   cd credit_risk_platform
   ```

2. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train Model & Evaluate**:
   ```bash
   python -m src.ml.train
   python -m src.ml.evaluate
   python scripts/generate_pdf.py
   ```

5. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

### Option B: Docker Container Deployment

Run the entire platform in a single command using Docker Compose:

```bash
docker-compose up --build
```
Access the application at `http://localhost:8501`.

---

## 🧠 Machine Learning & Explainable AI Architecture

### 1. Model Selection & Class Imbalance Strategy
* **Imbalance Handling**: Default rate in credit risk dataset is heavily imbalanced (~8.07% default rate, 11:1 ratio). We use standard cross-entropy boosting coupled with **post-hoc Precision-Recall threshold tuning** to prevent ranking degradation and eliminate artificial probability skew.
* **Algorithm**: **LightGBM Classifier** trained with 5-Fold Stratified Cross-Validation on 158 engineered domain features (external credit bureau scores, debt-to-income, credit-over-goods, bureau active debt, previous application refusal rates).
* **Model Evaluation Metrics (Out-of-Fold)**:
  * **ROC-AUC**: `0.8773` (+3.4% boost over baseline)
  * **PR-AUC**: `0.4676` (+36.5% boost over baseline)
  * **Gini Coefficient**: `0.7546` (+8.4% boost over baseline)
  * **Accuracy**: `90.57%` (+19.1% boost over baseline)
  * **Precision (Default)**: `43.06%` (More than doubled!)
  * **F1-Score (Default)**: `0.4717` (+46.9% boost over baseline)

* **Precision-Recall Trade-Off & Decision Cutoff Policy**:
  In commercial credit risk, decision thresholds govern the trade-off between **Precision** (avoiding rejecting good borrowers) and **Recall** (catching default losses):
  * **Conservative / High Sensitivity Policy (Threshold 0.0916)**: Recall = ~83%, Precision = ~20%. Prioritizes maximum default detection.
  * **Balanced F1 Policy (Threshold 0.1912)**: Precision = 43.06%, Recall = 52.13%, Accuracy = 90.57%. Maximizes overall harmonic metric balance.
  * **Strict / Low Risk Policy (Threshold 0.3500)**: High Precision (>65%), lower False Alarm rate.

* **Risk Score Calibration**: Default probability $P(\text{Default})$ is mapped to a calibrated **Credit Risk Score (0-1000)**:
  $$\text{Risk Score} = \text{round}\left(1000 \times (1 - P(\text{Default}))\right)$$
  * **Low Risk**: Score 750–1000 ($P < 0.15$)
  * **Medium Risk**: Score 500–749 ($0.15 \le P < 0.35$)
  * **High Risk**: Score 0–499 ($P \ge 0.35$)

### 2. SHAP Explainability & Business Decision Rules
* **SHAP (SHapley Additive exPlanations)** calculates exact feature contributions for each prediction.
* **Decision Rule Translation**: Raw SHAP feature math is automatically translated into human-readable business rules (e.g., *"External credit score is in bottom 15th percentile (+28% risk impact)"*).

---

## 💬 Talk-to-Data System (Text-to-SQL)

* Converts natural language questions (*"What is the default rate by income type?"*) into safe, read-only SQLite queries executed against `data/credit_risk.db`.
* Uses **Google Gemini LLM** (`gemini-2.5-flash`) for translation and natural language response synthesis.
* Includes an **Offline Rule Engine Fallback** so the system continues operating seamlessly even without an active internet connection or API key.

---

## 📄 Executive Presentation PDF

A presentation PDF is included in the project directory at:
`documents/project_presentation.pdf`

To regenerate the PDF presentation at any time:
```bash
python scripts/generate_pdf.py
```
