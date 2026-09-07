import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import DB_PATH, MODEL_PATH, MODELS_DIR
from src.utils.helpers import calculate_risk_score, get_risk_band, get_risk_color, format_currency
from src.data.loader import load_and_join_datasets, initialize_database
from src.ml.predict import CreditRiskPredictor
from src.talk_to_data.query_runner import QueryRunner

# Page Configuration
st.set_page_config(
    page_title="NeoStats AI - Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .metric-lbl { font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .badge-low { background-color: #DCFCE7; color: #166534; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .badge-medium { background-color: #FEF9C3; color: #854D0E; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .badge-high { background-color: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .rule-box {
        border-left: 4px solid #3B82F6;
        background-color: #F0F9FF;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_dict_strip=True)

# Helper Functions with Caching
@st.cache_data
def get_dataset():
    initialize_database()
    return load_and_join_datasets()

@st.cache_resource
def get_predictor():
    return CreditRiskPredictor()

@st.cache_resource
def get_query_runner():
    return QueryRunner()

# Header Banner
st.markdown('<div class="main-title">🏦 NeoStats AI - Credit Risk Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End AI Credit Scoring, Explainable AI (SHAP), Decision Rules & Conversational Talk-to-Data</div>', unsafe_allow_html=True)

# Load data and models
try:
    df_raw = get_dataset()
    predictor = get_predictor()
    query_runner = get_query_runner()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/isometric/100/bank.png", width=70)
st.sidebar.title("Navigation & Controls")
st.sidebar.info("Home Credit Default Risk Analytics Platform")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary", 
    "🔍 Exploratory Data (EDA)", 
    "🎯 Applicant Risk Scoring", 
    "💡 Explainable AI & Rules", 
    "📈 Model Performance", 
    "💬 Talk-to-Data Chatbot"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.header("Portfolio Risk Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    total_applicants = len(df_raw)
    default_cnt = int(df_raw['TARGET'].sum())
    default_rate = (default_cnt / total_applicants) * 100
    avg_credit = df_raw['AMT_CREDIT'].mean()
    total_exposure = df_raw['AMT_CREDIT'].sum()

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_applicants:,}</div><div class="metric-lbl">Total Applicants</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #DC2626;">{default_rate:.2f}%</div><div class="metric-lbl">Portfolio Default Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{format_currency(avg_credit)}</div><div class="metric-lbl">Avg Loan Credit</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{format_currency(total_exposure/1e6):.1f}M</div><div class="metric-lbl">Total Credit Exposure</div></div>', unsafe_allow_html=True)

    st.subheader("Key Portfolio Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig_target = px.pie(
            df_raw, 
            names='TARGET', 
            title="Applicant Default Status (0 = Repaid, 1 = Defaulted)",
            color='TARGET',
            color_discrete_map={0: '#22C55E', 1: '#EF4444'}
        )
        st.plotly_chart(fig_target, use_container_width=True)
    with c2:
        fig_income = px.histogram(
            df_raw, 
            x='NAME_INCOME_TYPE', 
            color='TARGET', 
            barmode='group',
            title="Applicant Distribution by Income Type",
            color_discrete_map={0: '#3B82F6', 1: '#EF4444'}
        )
        st.plotly_chart(fig_income, use_container_width=True)

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    
    eda_option = st.selectbox("Select Analysis Focus", [
        "Financial Distribution Analysis",
        "Demographic & Education Breakdown",
        "External Bureau Scores Correlation"
    ])
    
    if eda_option == "Financial Distribution Analysis":
        col1, col2 = st.columns(2)
        with col1:
            fig_inc = px.box(df_raw, x='TARGET', y='AMT_INCOME_TOTAL', color='TARGET', title="Total Income by Target", log_y=True)
            st.plotly_chart(fig_inc, use_container_width=True)
        with col2:
            fig_cred = px.box(df_raw, x='TARGET', y='AMT_CREDIT', color='TARGET', title="Credit Amount by Target", log_y=True)
            st.plotly_chart(fig_cred, use_container_width=True)
            
    elif eda_option == "Demographic & Education Breakdown":
        edu_df = df_raw.groupby(['NAME_EDUCATION_TYPE', 'TARGET']).size().reset_index(name='count')
        fig_edu = px.bar(edu_df, x='NAME_EDUCATION_TYPE', y='count', color='TARGET', title="Education Level vs Loan Status", barmode='group')
        st.plotly_chart(fig_edu, use_container_width=True)
        
    elif eda_option == "External Bureau Scores Correlation":
        fig_ext = px.scatter(df_raw.sample(min(1000, len(df_raw))), x='EXT_SOURCE_2', y='EXT_SOURCE_3', color='TARGET', title="External Bureau Scores (2 vs 3)", color_continuous_scale='Reds')
        st.plotly_chart(fig_ext, use_container_width=True)

# ==========================================
# TAB 3: APPLICANT RISK SCORING
# ==========================================
with tab3:
    st.header("Interactive Applicant Credit Scoring")
    
    mode = st.radio("Scoring Mode", ["Select Existing Applicant", "Custom Loan Assessment", "Batch CSV Scoring"])
    
    if mode == "Select Existing Applicant":
        selected_id = st.selectbox("Select Applicant ID", df_raw['SK_ID_CURR'].head(50).tolist())
        applicant_row = df_raw[df_raw['SK_ID_CURR'] == selected_id].iloc[0].to_dict()
        
        result = predictor.predict_single(applicant_row)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Default Probability", f"{result['default_probability']:.2%}")
        with c2:
            st.metric("Credit Risk Score", f"{result['risk_score']} / 1000")
        with c3:
            band = result['risk_band']
            badge_class = "badge-low" if "Low" in band else ("badge-medium" if "Medium" in band else "badge-high")
            st.markdown(f"### Risk Band\n<span class='{badge_class}'>{band}</span>", unsafe_allow_html=True)

        # Risk Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['risk_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Credit Risk Score (Higher is Better)"},
            gauge = {
                'axis': {'range': [0, 1000]},
                'bar': {'color': "#1E293B"},
                'steps': [
                    {'range': [0, 500], 'color': "#FEE2E2"},
                    {'range': [500, 750], 'color': "#FEF9C3"},
                    {'range': [750, 1000], 'color': "#DCFCE7"}
                ],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    elif mode == "Custom Loan Assessment":
        st.subheader("Input Applicant Financial Parameters")
        with st.form("custom_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                income = st.number_input("Total Annual Income ($)", value=150000.0, step=5000.0)
                credit = st.number_input("Requested Credit Amount ($)", value=450000.0, step=10000.0)
                annuity = st.number_input("Monthly Annuity ($)", value=22500.0, step=1000.0)
            with c2:
                income_type = st.selectbox("Income Type", df_raw['NAME_INCOME_TYPE'].unique().tolist())
                education = st.selectbox("Education Level", df_raw['NAME_EDUCATION_TYPE'].unique().tolist())
                age = st.slider("Applicant Age (Years)", 21, 70, 38)
            with c3:
                ext_2 = st.slider("External Bureau Score 2", 0.0, 1.0, 0.45)
                ext_3 = st.slider("External Bureau Score 3", 0.0, 1.0, 0.50)
                employed_yrs = st.slider("Employment Duration (Years)", 0.0, 40.0, 6.0)

            submitted = st.form_submit_button("Score Application")

        if submitted:
            custom_data = {
                'AMT_INCOME_TOTAL': income,
                'AMT_CREDIT': credit,
                'AMT_ANNUITY': annuity,
                'AMT_GOODS_PRICE': credit * 0.95,
                'NAME_INCOME_TYPE': income_type,
                'NAME_EDUCATION_TYPE': education,
                'DAYS_BIRTH': -int(age * 365.25),
                'DAYS_EMPLOYED': -int(employed_yrs * 365.25),
                'EXT_SOURCE_2': ext_2,
                'EXT_SOURCE_3': ext_3
            }
            res = predictor.predict_single(custom_data)
            st.success(f"Risk Score: {res['risk_score']}/1000 | Band: {res['risk_band']} | Default Probability: {res['default_probability']:.2%}")

    elif mode == "Batch CSV Scoring":
        uploaded_file = st.file_uploader("Upload CSV containing applicant features", type=["csv"])
        if uploaded_file:
            batch_input = pd.read_csv(uploaded_file)
            scored_df = predictor.predict_batch(batch_input)
            st.dataframe(scored_df.head(20))
            st.download_button("Download Scored Results", scored_df.to_csv(index=False), "scored_applicants.csv", "text/csv")

# ==========================================
# TAB 4: EXPLAINABLE AI & DECISION RULES
# ==========================================
with tab4:
    st.header("Explainable AI (SHAP) & Business Decision Rules")
    
    st.markdown("This tab breaks down **why** an applicant received their specific risk score, deriving human-readable decision rules for audit compliance.")
    
    selected_id_shap = st.selectbox("Select Applicant for SHAP Explanation", df_raw['SK_ID_CURR'].head(30).tolist(), key="shap_select")
    applicant_row_shap = df_raw[df_raw['SK_ID_CURR'] == selected_id_shap].iloc[0].to_dict()
    
    result_shap = predictor.predict_single(applicant_row_shap)
    
    st.subheader(f"Applicant #{selected_id_shap} Risk Summary")
    st.info(f"Score: {result_shap['risk_score']} / 1000 | Band: {result_shap['risk_band']} | Default Prob: {result_shap['default_probability']:.2%}")

    st.subheader("Derived Business Decision Rules")
    for rule in result_shap['decision_rules']:
        st.markdown(f"""
        <div class="rule-box">
            <b>Feature:</b> {rule['feature']} | <b>Impact:</b> {rule['impact']} (SHAP: {rule['shap_score']})<br>
            <i>{rule['rule_description']}</i>
        </div>
        """, unsafe_allow_html=True)

    if result_shap['shap_values']:
        st.subheader("SHAP Feature Contribution Plot")
        shap_df = pd.DataFrame(list(result_shap['shap_values'].items()), columns=['Feature', 'SHAP_Impact'])
        shap_df = shap_df.sort_values(by='SHAP_Impact', key=abs, ascending=False).head(10)
        
        fig_shap = px.bar(
            shap_df, 
            y='Feature', 
            x='SHAP_Impact', 
            orientation='h',
            color='SHAP_Impact',
            color_continuous_scale='RdYlGn_r',
            title="Top 10 Feature Drivers (+ Increases Risk | - Decreases Risk)"
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# TAB 5: MODEL PERFORMANCE
# ==========================================
with tab5:
    st.header("Model Performance & Evaluation Metrics")
    
    try:
        eval_path = MODELS_DIR / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_metrics = json.load(f)
                
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ROC-AUC Score", f"{eval_metrics['roc_auc']:.4f}")
            with c2:
                st.metric("PR-AUC Score", f"{eval_metrics['pr_auc']:.4f}")
            with c3:
                st.metric("Gini Coefficient", f"{eval_metrics['gini_index']:.4f}")
                
            st.subheader("Confusion Matrix")
            st.write(pd.DataFrame(eval_metrics['confusion_matrix'], columns=['Pred Repaid (0)', 'Pred Default (1)'], index=['Actual Repaid (0)', 'Actual Default (1)']))
        else:
            st.info("Run `python -m src.ml.evaluate` to generate complete metric payload.")
    except Exception as e:
        st.warning(f"Evaluation metrics display error: {e}")

# ==========================================
# TAB 6: TALK-TO-DATA CHATBOT
# ==========================================
with tab6:
    st.header("💬 Talk-to-Data Conversational System (NL-to-SQL)")
    st.markdown("Ask natural-language questions about credit applicants, risk metrics, or loan default distributions.")
    
    st.markdown("**Sample Questions (Click to try):**")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    user_query = None
    if q_col1.button("What is default rate by income type?"):
        user_query = "What is default rate by income type?"
    elif q_col2.button("Show average credit amount by loan status"):
        user_query = "Show average credit amount by loan status"
    elif q_col3.button("Show top 10 highest credit loans"):
        user_query = "Show top 10 highest credit loans"
        
    text_input = st.text_input("Or enter your own question:", value=user_query if user_query else "", placeholder="e.g. Compare male vs female default rates")
    
    if st.button("Submit Question") or user_query:
        query_to_run = text_input if text_input else user_query
        if query_to_run:
            with st.spinner("Translating natural language question into SQL and querying database..."):
                response = query_runner.ask(query_to_run)
                
            if response['status'] == 'success':
                st.success("Query Executed Successfully!")
                
                st.subheader("AI Executive Insight")
                st.info(response['summary'])
                
                with st.expander("Inspect Executed SQL Query"):
                    st.code(response['sql_query'], language="sql")
                    
                st.subheader("Query Results Data")
                st.dataframe(response['data'])
            else:
                st.error(f"Failed to execute query: {response['error']}")
