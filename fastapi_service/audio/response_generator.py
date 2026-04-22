import os


def generate_natural_language_answer(user_question, db_results):
    """
    Format SQL results as a plain English answer using Gemini.
    Returns an empty string when there are no results so the caller
    can try a different approach.
    """
    if not db_results:
        return ""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"Raw result: {db_results}"

    from google import genai

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    prompt = (
        f'A user asked: "{user_question}".\n'
        f"The audio event database returned: {db_results}.\n\n"
        "Write a clear, concise answer in plain English. "
        "If the result is a count like [(3,)], say 'The event occurred 3 times.' "
        "Do not mention the database, SQL, or raw data."
    )
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[response_generator] error: {e}")
        return ""
