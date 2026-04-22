import os

def generate_sql_from_intent(user_question: str) -> str:
    """
    Parses the natural language question and outputs a raw SQLite query.
    Extracts the intent (COUNT, DETECTION, SUMMARY) and relevant conditions.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Using fallback mock SQL.")
        return "SELECT * FROM events"

    from google import genai
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    prompt = f"""
    You are an AI that converts natural language into SQLite queries.
    Table schema: events(id, event, start_time, end_time, confidence)
    
    Rules:
    - If intent is COUNT, use SELECT COUNT(*)...
    - If intent is DETECTION, use SELECT * WHERE event='...'
    - If intent is SUMMARY, use SELECT * WHERE start_time BETWEEN ...
    - Do NOT wrap your answer in markdown code blocks like ```sql ... ```.
    - Return strictly the raw SQL string, and absolutely nothing else.
    
    Question: {user_question}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        sql_query = response.text.strip()
        # Clean up any potential markdown formatting the LLM might include despite instructions
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        return sql_query.strip()
    except Exception as e:
        print(f"Error generating SQL: {e}")
        # Fallback to a safe default if the LLM call fails
        return "SELECT * FROM events"
