import os
import time
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
from chat.models import Document, ChatHistory

# defining the tool for searching documents so the AI knows how to fetch context
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

            # make sure they actually have documents, otherwise no point in querying right now
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

            # check if we answered this before so we can just grab it and save some api calls
            user_id = user.id if user and user.is_authenticated else None
            cached = semantic_cache_get(query, document_id, user_id=user_id)
            if cached:
                print(f"[{chat_id}] ✅ Semantic Cache HIT")
                return cached

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            if document_id and str(document_id).strip():
                system_instruction = (
                    "You are a helpful AI assistant. The user has selected a specific document to focus on. "
                    "You MUST call the 'search_documents' tool for ANY question that is not a simple casual greeting (like 'hello' or 'who are you'). "
                    "Even for general knowledge questions (like 'what is AI'), you MUST search the document first. "
                    "If the answer is found in the document, answer based on the document. "
                    "If the user asks a specific question not covered in the document, kindly inform them that it's not in the selected document, "
                    "but you may provide general helpful information if appropriate."
                )
            else:
                system_instruction = (
                    "You are a helpful AI assistant with access to an uploaded knowledge base. "
                    "You MUST call the 'search_documents' tool for ANY informational or factual request, even general queries like 'what is AI'. "
                    "The ONLY time you should NOT call the 'search_documents' tool is if the user is just saying a casual greeting (e.g., 'hello', 'how are you'). "
                    "Answer based on what the tool returns from the documents when possible. "
                    "If the documents do not contain the answer, say: "
                    "'I couldn't find specific information about this in the uploaded documents, but here's what I know generally:' and provide a helpful answer."
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_doc_tool],
            )

            contents = []
            if chat_id:
                history = list(ChatHistory.objects.filter(session_id=chat_id).order_by('-created_at')[:8])
                history.reverse()
                for h in history:
                    if h.answer.startswith("The AI model is currently") or h.answer.startswith("The AI service quota") or h.answer.startswith("An unexpected error") or h.answer.startswith("📚 No documents") or h.answer.startswith("There was an issue"):
                        continue
                    contents.append(types.Content(role="user", parts=[types.Part(text=h.question)]))
                    contents.append(types.Content(role="model", parts=[types.Part(text=h.answer)]))
            
            contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

            used_rag = False

            MAX_ROUNDS = 3
            for _ in range(MAX_ROUNDS):
                response = None
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=contents,
                            config=config,
                        )
                        break
                    except Exception as api_e:
                        err_str = str(api_e)
                        if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str) and attempt < 2:
                            time.sleep(2)
                            continue
                        raise api_e

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
                        used_rag = True
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

            # we got a fresh answer, let's cache it for next time if it was a factual rag query, or if it's a very detailed answer
            if used_rag or len(answer) > 150:
                semantic_cache_set(query, answer, document_id, user_id=user_id)

            return answer if answer else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            print(f"[{chat_id}] ERROR: {e}")
            traceback.print_exc()
            error_msg = str(e)
            
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                return "The AI model is currently experiencing high demand. Spikes in demand are usually temporary. Please wait a few moments and try again."
            elif "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                return "The AI service quota has been exceeded. Please try again later."
            elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
                return "There was an issue processing your request. Please try rephrasing your question."
            else:
                return "An unexpected error occurred while generating a response. Please try again later."
