import re
import os
from typing import Optional
from src.utils.config import GEMINI_API_KEY, LLM_MODEL_NAME
from src.talk_to_data.prompt_templates import SQL_GENERATION_SYSTEM_PROMPT
from src.utils.logger import logger

class NLToSQLConverter:
    """Translates natural language user questions into executable SQLite queries using Gemini API with offline fallback."""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.use_llm = False
        self.genai_client = None
        self._setup_llm()

    def _setup_llm(self):
        """Initializes Gemini client if API key is provided."""
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                # Try new google-genai SDK first
                from google import genai
                self.genai_client = genai.Client(api_key=self.api_key)
                self.use_llm = True
                logger.info("Initialized Google GenAI client for NL-to-SQL.")
            except Exception as e1:
                try:
                    # Fallback to google-generativeai SDK
                    import google.generativeai as ggi
                    ggi.configure(api_key=self.api_key)
                    self.genai_client = ggi.GenerativeModel(LLM_MODEL_NAME)
                    self.use_llm = True
                    logger.info("Initialized legacy Google Generative AI client.")
                except Exception as e2:
                    logger.warning(f"Could not initialize Gemini API: {e1} | {e2}. Using offline SQL fallback engine.")
        else:
            logger.info("No valid GEMINI_API_KEY found in .env. Using offline rule-based SQL translation engine.")

    def generate_sql(self, user_question: str) -> str:
        """Translates user natural language query into clean SQL string."""
        question_clean = user_question.strip()
        
        if self.use_llm and self.genai_client:
            try:
                prompt = f"{SQL_GENERATION_SYSTEM_PROMPT}\n\nUser Question: {question_clean}\nSQL Query:"
                
                if hasattr(self.genai_client, 'models'):
                    # New google-genai SDK
                    response = self.genai_client.models.generate_content(
                        model=LLM_MODEL_NAME,
                        contents=prompt
                    )
                    raw_sql = response.text
                else:
                    # Legacy google-generativeai SDK
                    response = self.genai_client.generate_content(prompt)
                    raw_sql = response.text

                # Clean markdown fences
                cleaned_sql = re.sub(r'```sql\s*|\s*```', '', raw_sql).strip()
                cleaned_sql = cleaned_sql.replace(';', '').strip() + ';'
                logger.info(f"Gemini Generated SQL: {cleaned_sql}")
                return cleaned_sql
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based SQL parser.")

        # Offline Rule-Based Fallback Engine
        return self._offline_fallback_sql(question_clean)

    def _offline_fallback_sql(self, question: str) -> str:
        """Rule-based regex engine translating common banking queries to SQL."""
        q = question.lower()
        
        if "default rate by income type" in q or "default rate per income" in q:
            return "SELECT NAME_INCOME_TYPE, COUNT(*) AS total_applicants, SUM(TARGET) AS total_defaults, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM application_train GROUP BY NAME_INCOME_TYPE ORDER BY default_rate_pct DESC;"
            
        elif "default rate by education" in q or "education level" in q:
            return "SELECT NAME_EDUCATION_TYPE, COUNT(*) AS total_applicants, SUM(TARGET) AS total_defaults, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM application_train GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC;"
            
        elif "average credit" in q or "average loan amount" in q:
            return "SELECT TARGET AS is_default, COUNT(*) AS count, ROUND(AVG(AMT_CREDIT), 2) AS avg_credit, ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income FROM application_train GROUP BY TARGET;"
            
        elif "top" in q and ("high risk" in q or "highest loan" in q or "credit" in q):
            return "SELECT SK_ID_CURR, TARGET, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, EXT_SOURCE_2 FROM application_train ORDER BY AMT_CREDIT DESC LIMIT 10;"
            
        elif "gender" in q or "male vs female" in q:
            return "SELECT CODE_GENDER, COUNT(*) AS total, SUM(TARGET) AS defaults, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct, ROUND(AVG(AMT_CREDIT), 2) AS avg_credit FROM application_train GROUP BY CODE_GENDER;"
            
        elif "occupation" in q or "job" in q:
            return "SELECT OCCUPATION_TYPE, COUNT(*) AS total, SUM(TARGET) AS defaults, ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM application_train WHERE OCCUPATION_TYPE IS NOT NULL GROUP BY OCCUPATION_TYPE ORDER BY total DESC LIMIT 10;"
            
        elif "bureau" in q or "active loans" in q:
            return "SELECT CREDIT_ACTIVE, COUNT(*) AS total_bureau_records, ROUND(AVG(AMT_CREDIT_SUM_DEBT), 2) AS avg_bureau_debt FROM bureau GROUP BY CREDIT_ACTIVE;"
            
        else:
            # Default fallback query
            return "SELECT TARGET AS is_default, COUNT(*) AS total_applicants, ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount, ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income FROM application_train GROUP BY TARGET;"
