import os

def generate_natural_language_answer(user_question: str, db_results: list) -> str:
    """
    Takes the raw SQL results and original question to formulate a clear natural language response.
    Returns an empty string when the database has no matching events so the caller
    can trigger a different fallback path (e.g. transcript-based Gemini).
    """
    # Empty results — signal the caller to use a different strategy
    if not db_results:
        return ""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Using raw fallback.")
        return f"Raw Data: {db_results}"

    from google import genai
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    prompt = f"""
    You are a helpful AI assistant connected to an audio event database.
    A user asked: "{user_question}".
    The database executed a query and returned the following raw data: {db_results}.

    Rules:
    - Formulate a clear, concise, and natural-sounding answer.
    - If the user asks for a count, and the data is just a number (e.g., [(3,)]), say "The event happened 3 times."
    - Do NOT mention the database, SQL, or "raw data" in your response. Just answer the user.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating natural language response: {e}")
        return ""
