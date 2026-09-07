"""
Exploratory Data Analysis (EDA) Script for Home Credit Default Risk Dataset
Analyzes demographics, financial metrics, target distributions, and missing values.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data.loader import load_and_join_datasets, initialize_database
from src.utils.config import DB_PATH

def run_eda():
    print("=" * 60)
    print("NEOSTATS AI - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # Load data
    initialize_database()
    df = load_and_join_datasets()
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Target distribution
    target_counts = df['TARGET'].value_counts()
    default_rate = df['TARGET'].mean() * 100
    print(f"\n2. TARGET DISTRIBUTION (CLASS IMBALANCE)")
    print(f"Non-Default (0): {target_counts.get(0, 0):,} ({100 - default_rate:.2f}%)")
    print(f"Default (1):     {target_counts.get(1, 0):,} ({default_rate:.2f}%)")
    
    # Financial Summary
    print(f"\n3. FINANCIAL METRICS SUMMARY")
    financial_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    print(df[financial_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # Income Type breakdown
    print(f"\n4. DEFAULT RATE BY INCOME TYPE")
    income_eda = df.groupby('NAME_INCOME_TYPE')['TARGET'].agg(['count', 'mean']).reset_index()
    income_eda.columns = ['Income Type', 'Applicant Count', 'Default Rate']
    income_eda['Default Rate'] = (income_eda['Default Rate'] * 100).round(2)
    print(income_eda.sort_values(by='Applicant Count', ascending=False).to_string(index=False))
    
    # Education Level breakdown
    print(f"\n5. DEFAULT RATE BY EDUCATION LEVEL")
    edu_eda = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['count', 'mean']).reset_index()
    edu_eda.columns = ['Education Level', 'Applicant Count', 'Default Rate']
    edu_eda['Default Rate'] = (edu_eda['Default Rate'] * 100).round(2)
    print(edu_eda.sort_values(by='Applicant Count', ascending=False).to_string(index=False))

    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    run_eda()
