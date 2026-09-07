import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.utils.config import DATA_DIR, DB_PATH, SQL_DIR
from src.utils.logger import logger

def load_raw_datasets() -> Dict[str, pd.DataFrame]:
    """Loads raw Home Credit CSV datasets from data directory."""
    app_path = DATA_DIR / "application_train.csv"
    bureau_path = DATA_DIR / "bureau.csv"
    prev_path = DATA_DIR / "previous_application.csv"
    
    if not app_path.exists():
        raise FileNotFoundError(f"Dataset file missing: {app_path}. Please place CSV files in data/ directory.")
        
    logger.info("Loading raw CSV datasets...")
    datasets = {}
    datasets["application_train"] = pd.read_csv(app_path)
    logger.info(f"Loaded application_train: {datasets['application_train'].shape}")
    
    if bureau_path.exists():
        datasets["bureau"] = pd.read_csv(bureau_path)
        logger.info(f"Loaded bureau: {datasets['bureau'].shape}")
        
    if prev_path.exists():
        datasets["previous_application"] = pd.read_csv(prev_path)
        logger.info(f"Loaded previous_application: {datasets['previous_application'].shape}")
        
    return datasets

def initialize_database(force_recreate: bool = False) -> sqlite3.Connection:
    """
    Initializes SQLite database with schema DDL and populates tables from CSV files.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema_path = SQL_DIR / "schema.sql"
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if tables already exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='application_train';")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists or force_recreate:
        logger.info(f"Initializing SQLite database at {DB_PATH} using schema {schema_path}...")
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
                
        datasets = load_raw_datasets()
        for table_name, df in datasets.items():
            logger.info(f"Writing table '{table_name}' to SQLite database ({len(df)} rows)...")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
        logger.info("Database schema and tables initialized successfully.")
    else:
        logger.info(f"Database already initialized at {DB_PATH}")
        
    return conn

def load_and_join_datasets() -> pd.DataFrame:
    """
    Loads application_train and joins aggregated features from bureau and previous_application.
    Returns unified pandas DataFrame ready for preprocessing.
    """
    datasets = load_raw_datasets()
    app_df = datasets["application_train"].copy()
    
    # Aggregate bureau data if present
    if "bureau" in datasets:
        bureau = datasets["bureau"]
        logger.info("Aggregating bureau table features...")
        bureau_agg = bureau.groupby("SK_ID_CURR").agg(
            TOTAL_BUREAU_LOANS=('SK_ID_BUREAU', 'count'),
            ACTIVE_BUREAU_LOANS=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
            TOTAL_BUREAU_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
            MAX_OVERDUE=('AMT_CREDIT_MAX_OVERDUE', 'max'),
            DAYS_CREDIT_MIN=('DAYS_CREDIT', 'min')
        ).reset_index()
        
        app_df = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        app_df['TOTAL_BUREAU_LOANS'] = app_df['TOTAL_BUREAU_LOANS'].fillna(0)
        app_df['ACTIVE_BUREAU_LOANS'] = app_df['ACTIVE_BUREAU_LOANS'].fillna(0)
        app_df['TOTAL_BUREAU_DEBT'] = app_df['TOTAL_BUREAU_DEBT'].fillna(0)
    else:
        app_df['TOTAL_BUREAU_LOANS'] = 0
        app_df['ACTIVE_BUREAU_LOANS'] = 0
        app_df['TOTAL_BUREAU_DEBT'] = 0

    # Aggregate previous applications if present
    if "previous_application" in datasets:
        prev = datasets["previous_application"]
        logger.info("Aggregating previous application table features...")
        prev_agg = prev.groupby("SK_ID_CURR").agg(
            PREV_APPLICATIONS_COUNT=('SK_ID_PREV', 'count'),
            PREV_REFUSED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
            PREV_APPROVED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum())
        ).reset_index()
        
        app_df = app_df.merge(prev_agg, on="SK_ID_CURR", how="left")
        app_df['PREV_APPLICATIONS_COUNT'] = app_df['PREV_APPLICATIONS_COUNT'].fillna(0)
        app_df['PREV_REFUSED_COUNT'] = app_df['PREV_REFUSED_COUNT'].fillna(0)
        app_df['PREV_APPROVED_COUNT'] = app_df['PREV_APPROVED_COUNT'].fillna(0)
    else:
        app_df['PREV_APPLICATIONS_COUNT'] = 0
        app_df['PREV_REFUSED_COUNT'] = 0
        app_df['PREV_APPROVED_COUNT'] = 0

    logger.info(f"Final merged dataset shape: {app_df.shape}")
    return app_df

if __name__ == "__main__":
    initialize_database()
