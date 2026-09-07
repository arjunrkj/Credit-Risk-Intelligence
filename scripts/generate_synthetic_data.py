import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_DIR
from src.utils.logger import logger

def generate_home_credit_synthetic_data(num_records: int = 2500, random_seed: int = 42):
    """
    Generates synthetic datasets adhering to the Home Credit Default Risk schema.
    Saves application_train.csv, bureau.csv, and previous_application.csv into data/
    """
    np.random.seed(random_seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    app_path = DATA_DIR / "application_train.csv"
    bureau_path = DATA_DIR / "bureau.csv"
    prev_path = DATA_DIR / "previous_application.csv"

    logger.info(f"Generating synthetic Home Credit dataset ({num_records} records)...")

    # 1. Generate application_train.csv
    curr_ids = np.arange(100001, 100001 + num_records)
    
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed']
    income_probs = [0.52, 0.23, 0.18, 0.06, 0.01]
    
    education_types = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    education_probs = [0.70, 0.24, 0.03, 0.02, 0.01]
    
    family_statuses = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    housing_types = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment']
    occupations = ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants']

    incomes = np.random.lognormal(mean=11.8, sigma=0.5, size=num_records) # ~$130k median
    credits = incomes * np.random.uniform(1.5, 6.0, size=num_records)
    annuities = credits * np.random.uniform(0.04, 0.10, size=num_records)
    goods_prices = credits * np.random.uniform(0.85, 1.0, size=num_records)

    days_birth = np.random.randint(-23000, -7300, size=num_records) # 20 to 63 years old
    days_employed = np.random.randint(-12000, -100, size=num_records)
    
    ext_1 = np.random.uniform(0.1, 0.9, size=num_records)
    ext_2 = np.random.uniform(0.05, 0.85, size=num_records)
    ext_3 = np.random.uniform(0.05, 0.85, size=num_records)

    # Induce realistic ground truth target relationship
    # Higher debt-to-income and lower external scores increase default chance
    credit_to_income = credits / incomes
    risk_score_linear = (
        -3.0 * ext_2 
        - 2.5 * ext_3 
        + 0.3 * credit_to_income 
        + np.random.normal(0, 0.8, size=num_records)
    )
    prob_default = 1.0 / (1.0 + np.exp(-risk_score_linear))
    target = (prob_default > np.percentile(prob_default, 91.5)).astype(int) # ~8.5% default rate

    app_df = pd.DataFrame({
        'SK_ID_CURR': curr_ids,
        'TARGET': target,
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=num_records, p=[0.9, 0.1]),
        'CODE_GENDER': np.random.choice(['F', 'M'], size=num_records, p=[0.65, 0.35]),
        'FLAG_OWN_CAR': np.random.choice(['N', 'Y'], size=num_records, p=[0.66, 0.34]),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], size=num_records, p=[0.69, 0.31]),
        'CNT_CHILDREN': np.random.choice([0, 1, 2, 3], size=num_records, p=[0.7, 0.2, 0.08, 0.02]),
        'AMT_INCOME_TOTAL': np.round(incomes, 2),
        'AMT_CREDIT': np.round(credits, 2),
        'AMT_ANNUITY': np.round(annuities, 2),
        'AMT_GOODS_PRICE': np.round(goods_prices, 2),
        'NAME_INCOME_TYPE': np.random.choice(income_types, size=num_records, p=income_probs),
        'NAME_EDUCATION_TYPE': np.random.choice(education_types, size=num_records, p=education_probs),
        'NAME_FAMILY_STATUS': np.random.choice(family_statuses, size=num_records),
        'NAME_HOUSING_TYPE': np.random.choice(housing_types, size=num_records),
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'OCCUPATION_TYPE': np.random.choice(occupations, size=num_records),
        'CNT_FAM_MEMBERS': np.random.choice([1.0, 2.0, 3.0, 4.0], size=num_records),
        'REGION_RATING_CLIENT': np.random.choice([1, 2, 3], size=num_records, p=[0.15, 0.70, 0.15]),
        'EXT_SOURCE_1': np.round(ext_1, 4),
        'EXT_SOURCE_2': np.round(ext_2, 4),
        'EXT_SOURCE_3': np.round(ext_3, 4),
        'OBS_30_CNT_SOCIAL_CIRCLE': np.random.poisson(0.5, size=num_records),
        'DEF_30_CNT_SOCIAL_CIRCLE': np.random.poisson(0.1, size=num_records),
        'OBS_60_CNT_SOCIAL_CIRCLE': np.random.poisson(0.5, size=num_records),
        'DEF_60_CNT_SOCIAL_CIRCLE': np.random.poisson(0.08, size=num_records),
        'DAYS_LAST_PHONE_CHANGE': np.random.randint(-3000, 0, size=num_records),
        'AMT_REQ_CREDIT_BUREAU_YEAR': np.random.poisson(1.8, size=num_records)
    })

    app_df.to_csv(app_path, index=False)
    logger.info(f"Saved {app_path.name} with shape {app_df.shape}")

    # 2. Generate bureau.csv
    bureau_records = []
    bureau_id_counter = 5000000
    for cid in curr_ids:
        num_bureau_loans = np.random.randint(0, 5)
        for _ in range(num_bureau_loans):
            bureau_id_counter += 1
            bureau_records.append({
                'SK_ID_BUREAU': bureau_id_counter,
                'SK_ID_CURR': cid,
                'CREDIT_ACTIVE': np.random.choice(['Closed', 'Active', 'Sold'], p=[0.6, 0.38, 0.02]),
                'CREDIT_CURRENCY': 'currency 1',
                'DAYS_CREDIT': np.random.randint(-2800, -10),
                'CREDIT_DAY_OVERDUE': np.random.choice([0, 15, 30, 60], p=[0.92, 0.04, 0.03, 0.01]),
                'AMT_CREDIT_MAX_OVERDUE': np.round(np.random.exponential(500), 2),
                'CNT_CREDIT_PROLONG': 0,
                'AMT_CREDIT_SUM': np.round(np.random.exponential(150000), 2),
                'AMT_CREDIT_SUM_DEBT': np.round(np.random.exponential(50000), 2),
                'AMT_CREDIT_SUM_LIMIT': 0.0,
                'AMT_CREDIT_SUM_OVERDUE': 0.0,
                'CREDIT_TYPE': np.random.choice(['Consumer credit', 'Credit card', 'Car loan', 'Mortgage'], p=[0.6, 0.25, 0.1, 0.05])
            })
    bureau_df = pd.DataFrame(bureau_records)
    bureau_df.to_csv(bureau_path, index=False)
    logger.info(f"Saved {bureau_path.name} with shape {bureau_df.shape}")

    # 3. Generate previous_application.csv
    prev_records = []
    prev_id_counter = 2000000
    for cid in curr_ids:
        num_prev = np.random.randint(0, 4)
        for _ in range(num_prev):
            prev_id_counter += 1
            status = np.random.choice(['Approved', 'Refused', 'Canceled'], p=[0.65, 0.25, 0.10])
            prev_records.append({
                'SK_ID_PREV': prev_id_counter,
                'SK_ID_CURR': cid,
                'NAME_CONTRACT_TYPE': np.random.choice(['Consumer loans', 'Cash loans', 'Revolving loans']),
                'AMT_ANNUITY': np.round(np.random.exponential(10000), 2),
                'AMT_APPLICATION': np.round(np.random.exponential(100000), 2),
                'AMT_CREDIT': np.round(np.random.exponential(105000), 2),
                'AMT_DOWN_PAYMENT': np.round(np.random.exponential(5000), 2),
                'NAME_CONTRACT_STATUS': status,
                'DAYS_DECISION': np.random.randint(-2500, -10),
                'CODE_REJECT_REASON': 'LIMIT' if status == 'Refused' else 'XAP'
            })
    prev_df = pd.DataFrame(prev_records)
    prev_df.to_csv(prev_path, index=False)
    logger.info(f"Saved {prev_path.name} with shape {prev_df.shape}")

    return app_df, bureau_df, prev_df

if __name__ == "__main__":
    generate_home_credit_synthetic_data()
