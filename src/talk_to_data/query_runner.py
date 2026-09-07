import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple
from src.utils.config import DB_PATH, GEMINI_API_KEY, LLM_MODEL_NAME
from src.data.loader import initialize_database
from src.talk_to_data.nl_to_sql import NLToSQLConverter
from src.talk_to_data.prompt_templates import RESPONSE_SYNTHESIS_PROMPT
from src.utils.logger import logger

class QueryRunner:
    """Safe execution engine for SQL queries against SQLite database with natural language response synthesis."""
    
    def __init__(self):
        self.converter = NLToSQLConverter()
        self._ensure_db()

    def _ensure_db(self):
        """Ensures SQLite database exists and is populated."""
        if not DB_PATH.exists():
            initialize_database()

    def validate_sql(self, sql_query: str) -> bool:
        """Validates that query is safe and read-only."""
        sql = sql_query.strip().upper()
        if not sql.startswith("SELECT") and not sql.startswith("WITH"):
            return False
            
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        for kw in forbidden:
            if f" {kw} " in f" {sql} ":
                return False
        return True

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Executes validated SQL query against SQLite and returns Pandas DataFrame."""
        if not self.validate_sql(sql_query):
            raise ValueError("Forbidden SQL query detected. Only SELECT statements are permitted.")
            
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(sql_query, conn)
            return df
        finally:
            conn.close()

    def synthesize_response(self, question: str, sql_query: str, df: pd.DataFrame) -> str:
        """Synthesizes human-readable business insights from query results."""
        if df.empty:
            return "No matching records were found in the database for this query."
            
        # If Gemini client is active, call LLM
        if self.converter.use_llm and self.converter.genai_client:
            try:
                data_summary = df.head(10).to_string(index=False)
                prompt = RESPONSE_SYNTHESIS_PROMPT.format(
                    question=question,
                    sql_query=sql_query,
                    data_result=data_summary
                )
                
                if hasattr(self.converter.genai_client, 'models'):
                    response = self.converter.genai_client.models.generate_content(
                        model=LLM_MODEL_NAME,
                        contents=prompt
                    )
                    return response.text.strip()
                else:
                    response = self.converter.genai_client.generate_content(prompt)
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Response synthesis LLM error: {e}")

        # Rule-based fallback synthesis
        num_rows = len(df)
        cols = ", ".join(df.columns)
        first_row_summary = df.iloc[0].to_dict()
        return f"The database query returned {num_rows} record(s). Key fields analyzed include: {cols}. For example, the top record shows: {first_row_summary}."

    def ask(self, question: str) -> Dict[str, Any]:
        """Full end-to-end Talk-to-Data pipeline execution."""
        logger.info(f"Processing Talk-to-Data query: '{question}'")
        
        sql_query = self.converter.generate_sql(question)
        
        try:
            df_result = self.execute_query(sql_query)
            insight_summary = self.synthesize_response(question, sql_query, df_result)
            
            return {
                'question': question,
                'sql_query': sql_query,
                'data': df_result,
                'summary': insight_summary,
                'status': 'success',
                'error': None
            }
        except Exception as e:
            logger.error(f"SQL execution error for query '{sql_query}': {e}")
            return {
                'question': question,
                'sql_query': sql_query,
                'data': pd.DataFrame(),
                'summary': f"Error executing query: {str(e)}",
                'status': 'error',
                'error': str(e)
            }

if __name__ == "__main__":
    runner = QueryRunner()
    res = runner.ask("What is the default rate by income type?")
    print("Generated SQL:", res['sql_query'])
    print("Summary:", res['summary'])
    print(res['data'])
