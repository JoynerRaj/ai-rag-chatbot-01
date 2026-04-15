import os
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
from chat.models import Document

# short messages that clearly don't need a document search
CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
    "good afternoon", "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you",
    "great", "nice", "cool", "awesome", "got it", "understood"
}

def _is_casual_message(text: str) -> bool:
    """True only for obvious greetings and small talk, not actual questions."""
    cleaned = text.strip().lower().rstrip("!?.,:;")
    return cleaned in CASUAL_PHRASES


class AIAgentService:
    @staticmethod
    def process_query(query: str, document_id: str, user, chat_id: int, chat_history: list = None) -> str:
        try:
            user_id = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] query={query!r}  doc_id={document_id!r}  user={user_id}")

            # make sure the user actually has documents that are ready in Pinecone
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

            # don't bother checking cache for hi/thanks, waste of time
            if not _is_casual_message(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] Cache HIT")
                    return cached

            # search Pinecone before calling Gemini so the answer is always
            # grounded in the actual uploaded documents, not just general knowledge
            rag_context = ""
            if not _is_casual_message(query):
                try:
                    rag_context = FastAPIClient.search_documents(query, document_id or "")
                    print(f"[{chat_id}] RAG context: {len(rag_context)} chars")
                except Exception as e:
                    print(f"[{chat_id}] RAG search failed: {e}")

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            has_real_context = rag_context and "No relevant" not in rag_context and "No sufficiently" not in rag_context

            if has_real_context:
                # tell Gemini to answer from the document, not from its training data
                system_instruction = (
                    "You are a helpful AI assistant. You have been given content from the user's uploaded documents. "
                    "You also have the conversation history for this session - use it to remember the user's name, "
                    "previous questions, and anything else they have shared. "
                    "Base your answer on the document content provided. "
                    "Do not say you couldn't find information - the content is there. Use it."
                )
            else:
                # no great document match, just be a normal assistant
                system_instruction = (
                    "You are a helpful AI assistant. "
                    "You have the conversation history for this session - remember anything the user has told you. "
                    "Answer the user's question as helpfully as you can."
                )

            # build conversation so Gemini has context from earlier in the chat
            contents = []
            if chat_history:
                for past in chat_history:
                    contents.append(types.Content(role="user", parts=[types.Part(text=past["question"])]))
                    contents.append(types.Content(role="model", parts=[types.Part(text=past["answer"])]))

            # attach the RAG context directly to the user's message so Gemini can't miss it
            if has_real_context:
                user_message = (
                    f"Here is the relevant content from the uploaded documents:\n\n"
                    f"{rag_context}\n\n"
                    f"Answer this question using the document content above:\n{query}"
                )
            else:
                user_message = query

            contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

            config = types.GenerateContentConfig(system_instruction=system_instruction)

            # try 2.5-flash first, fall back to 2.0 if it gives an empty response
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
                        print(f"[{chat_id}] {model_name} came back empty, trying next...")
                        continue

                    answer = "".join([
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]).strip()

                    if answer:
                        print(f"[{chat_id}] [{model_name}] {answer[:80]!r}")
                        break
                    else:
                        print(f"[{chat_id}] {model_name} returned no text, trying next...")

                except Exception as model_err:
                    err_str = str(model_err)
                    if "must contain either output text or tool calls" in err_str or "empty" in err_str.lower():
                        print(f"[{chat_id}] {model_name} empty-output error: {model_err}")
                        continue
                    raise

            # save to cache only when the answer came from our documents
            if has_real_context and answer and user_id is not None:
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
                return "An unexpected error occurred. Please try again."
