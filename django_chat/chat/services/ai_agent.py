import os
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
from chat.models import Document

# greetings and small talk that definitely don't need a document search
CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
    "good afternoon", "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you",
    "great", "nice", "cool", "awesome", "got it", "understood"
}

def _is_casual_message(text: str) -> bool:
    """Returns True only for clear greetings/small talk with no factual content."""
    cleaned = text.strip().lower().rstrip("!?.,:;")
    return cleaned in CASUAL_PHRASES


class AIAgentService:
    @staticmethod
    def process_query(query: str, document_id: str, user, chat_id: int, chat_history: list = None) -> str:
        try:
            user_id = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] process_query: {query!r}  doc_id={document_id!r}  user_id={user_id}")

            # only count documents that are fully embedded - pending ones have no Pinecone data
            if user and user.is_authenticated:
                has_docs = Document.objects.filter(user=user, embedding_status="done").exists()
            else:
                has_docs = Document.objects.filter(embedding_status="done").exists()

            if not has_docs:
                return (
                    "No embedded documents found.\n\n"
                    "Please go to **Documents → Upload** and add a document first. "
                    "If you just uploaded one, wait a moment for embedding to finish, then try again."
                )

            # skip cache for small talk - not worth storing
            if not _is_casual_message(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] Semantic Cache HIT")
                    return cached

            # ---------------------------------------------------------------
            # PRE-FETCH: search Pinecone directly before sending to Gemini.
            # This guarantees the document content is in the prompt whether
            # Gemini decides to call the tool or not.
            # ---------------------------------------------------------------
            rag_context = ""
            if not _is_casual_message(query):
                try:
                    rag_context = FastAPIClient.search_documents(query, document_id or "")
                    print(f"[{chat_id}] Pre-fetch RAG: {len(rag_context)} chars")
                except Exception as e:
                    print(f"[{chat_id}] Pre-fetch RAG failed: {e}")

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            # system prompt depends on whether we have document content
            if rag_context and "No relevant" not in rag_context and "No sufficiently" not in rag_context:
                system_instruction = (
                    "You are a helpful AI assistant. You have been given content from the user's uploaded documents. "
                    "You also have the full conversation history for this session - use it to remember anything the user "
                    "has told you (their name, previous questions, preferences). "
                    "IMPORTANT: Base your answer PRIMARILY on the document content provided below. "
                    "Do NOT say you could not find information in the documents - the content is right there. "
                    "If the document content is relevant to the question, use it directly to answer. "
                    "Only supplement with general knowledge if the documents truly do not cover the topic."
                )
            else:
                system_instruction = (
                    "You are a helpful AI assistant with access to a knowledge base. "
                    "You have the full conversation history for this session - use it to remember anything the user "
                    "has shared (name, topics, preferences). "
                    "For factual questions, try to answer from the uploaded documents if possible. "
                    "If you are making small talk or answering a general question, just respond naturally."
                )

            # build conversation history for context
            contents = []
            if chat_history:
                for past in chat_history:
                    contents.append(types.Content(role="user", parts=[types.Part(text=past["question"])]))
                    contents.append(types.Content(role="model", parts=[types.Part(text=past["answer"])]))

            # build the user message - attach document context inline if we have it
            if rag_context and "No relevant" not in rag_context and "No sufficiently" not in rag_context:
                user_message = (
                    f"Here is the relevant content from the uploaded documents:\n\n"
                    f"{rag_context}\n\n"
                    f"Based on the above document content, please answer this question:\n{query}"
                )
            else:
                user_message = query

            contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
            )

            # try gemini-2.5-flash first, fall back to 2.0-flash if empty output
            MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.0-flash"]
            answer = ""

            for model_name in MODELS_TO_TRY:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )

                    candidate = response.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        print(f"[{chat_id}] {model_name} returned empty parts, trying next model...")
                        continue

                    answer = "".join([
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]).strip()

                    if answer:
                        print(f"[{chat_id}] [{model_name}] Answer: {answer[:80]!r}")
                        break
                    else:
                        print(f"[{chat_id}] {model_name} gave no text, trying next model...")

                except Exception as model_err:
                    err_str = str(model_err)
                    if "must contain either output text or tool calls" in err_str or "empty" in err_str.lower():
                        print(f"[{chat_id}] {model_name} empty-output error, falling back: {model_err}")
                        continue
                    raise

            # cache successful answers that used document content
            doc_was_used = rag_context and "No relevant" not in rag_context and "No sufficiently" not in rag_context
            if doc_was_used and answer and user_id is not None:
                semantic_cache_set(query, answer, document_id, user_id=user_id)

            return answer if answer else "I'm sorry, I couldn't generate a response. Please try again."

        except Exception as e:
            print(f"[{chat_id}] ERROR: {e}")
            traceback.print_exc()
            error_msg = str(e)

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                return "The AI model is temporarily overloaded. Please wait a few seconds and try again."
            elif "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                return "The AI service quota has been exceeded. Please try again later."
            elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
                return "There was an issue processing your request. Please try rephrasing your question."
            else:
                return "An unexpected error occurred while generating a response. Please try again later."
