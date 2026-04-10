import os
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
from chat.models import Document

# ── RAG Tool Definition ──────────────────────────────────────────────────────
search_doc_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the uploaded knowledge base documents for information relevant "
                "to the user's question. Use this when the user asks a specific factual "
                "question that may be answered by the uploaded documents. "
                "Do NOT use this for greetings, thanks, or casual conversation."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query to look up in the documents"
                    ),
                    "document_id": types.Schema(
                        type=types.Type.STRING,
                        description="Optional UUID of a specific document to search within. Leave empty to search all."
                    )
                },
                required=["query"]
            )
        )
    ]
)

class AIAgentService:
    @staticmethod
    def process_query(query: str, document_id: str, user, chat_id: int) -> str:
        try:
            print(f"[{chat_id}] process_query: {query!r}")

            # Guard: No documents uploaded yet for THIS user
            if user and user.is_authenticated:
                has_docs = Document.objects.filter(user=user).exists()
            else:
                has_docs = Document.objects.exists()
                
            if not has_docs:
                return (
                    "📚 No documents uploaded yet.\n\n"
                    "Please go to **Documents → Upload** and add a document first. "
                    "I can only answer questions based on your uploaded knowledge base."
                )

            # Semantic Cache: READ
            cached = semantic_cache_get(query, document_id)
            if cached:
                print(f"[{chat_id}] ✅ Semantic Cache HIT")
                return cached

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            if document_id and str(document_id).strip():
                system_instruction = (
                    "You are a helpful AI assistant. The user has selected a SPECIFIC document to search. "
                    "You MUST call the 'search_documents' tool for EVERY question to retrieve context from that document. "
                    "Answer ONLY based on the retrieved document context. "
                    "If the answer is not found in the document, say: 'This information is not available in the selected document.' "
                    "Do NOT use your general knowledge — only use what the tool returns."
                )
            else:
                system_instruction = (
                    "You are a helpful AI assistant with access to an uploaded knowledge base. "
                    "You MUST call the 'search_documents' tool for EVERY question to search the uploaded documents. "
                    "Answer ONLY based on what the tool returns from the uploaded documents. "
                    "Do NOT use your general knowledge or training data under any circumstances. "
                    "If the documents do not contain the answer, say: "
                    "'I could not find this information in the uploaded documents. Please try uploading a relevant document.'"
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_doc_tool],
            )

            contents = [types.Content(role="user", parts=[types.Part(text=query)])]

            MAX_ROUNDS = 3
            for _ in range(MAX_ROUNDS):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )

                contents.append(response.candidates[0].content)

                fn_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "function_call") and part.function_call
                ]

                if not fn_calls:
                    break

                tool_response_parts = []
                for fn_call in fn_calls:
                    if fn_call.name == "search_documents":
                        args = dict(fn_call.args)
                        rag_query = args.get("query", query)
                        doc_id = args.get("document_id", document_id) or document_id

                        print(f"[{chat_id}] 🔍 RAG tool called: query={rag_query!r}")
                        rag_result = FastAPIClient.search_documents(rag_query, doc_id)
                        print(f"[{chat_id}] 📄 RAG result: {len(rag_result)} chars")

                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name="search_documents",
                                response={"result": rag_result}
                            )
                        )

                contents.append(types.Content(role="tool", parts=tool_response_parts))

            answer = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, "text") and part.text])
            answer = answer.strip()

            print(f"[{chat_id}] ✅ Answer: {answer[:80]!r}")

            # Semantic Cache: WRITE
            semantic_cache_set(query, answer, document_id)

            return answer if answer else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            print(f"[{chat_id}] ERROR: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
