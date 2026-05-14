import os
import traceback

from google import genai
from google.genai import types

from chat.services import embedding_service
from chat.semantic_cache import semantic_cache_get, semantic_cache_set, should_cache
from chat.models import Document


CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
    "good afternoon", "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you",
    "great", "nice", "cool", "awesome", "got it", "understood",
}


def _is_casual(text):
    return text.strip().lower().rstrip("!?.,:;") in CASUAL_PHRASES


def _gemini_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"api_version": "v1alpha"},
    )


def _generate(client, system_instruction, user_message, chat_history=None, stream=False):
    contents = []
    if chat_history:
        for turn in chat_history:
            contents.append(types.Content(role="user",  parts=[types.Part.from_text(text=turn["question"])]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["answer"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[{"google_search": {}}],
    )

    for model in ("gemini-2.5-flash", "gemini-2.0-flash"):
        try:
            if stream:
                response = client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return
            else:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue
                text = "".join(
                    p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                ).strip()
                if text:
                    return text
        except Exception as e:
            if "empty" in str(e).lower() or "output text" in str(e):
                continue
            if stream:
                yield f"\n\n> Error: {str(e)}"
                return
            raise

    if not stream:
        return ""


def _get_memory_context(user_id, query):
    try:
        from chat.services.memory_service import build_memory_context
        return build_memory_context(user_id=user_id, query=query)
    except Exception as e:
        print(f"[ai_agent] memory fetch failed: {e}")
        return ""


def _build_rag_system_prompt(has_context: bool, memory_context: str = "") -> str:
    """
    Build the system prompt that instructs the LLM how to answer.
    """
    if has_context:
        instruction = (
            "You are a helpful and intelligent AI assistant. "
            "Use the knowledge provided to answer the user's question clearly and accurately. "
            "Do NOT mention that you were provided documents, excerpts, or a knowledge base. "
            "Simply answer the question naturally as if you inherently know the information. "
            "If the provided knowledge doesn't fully answer the question, supplement it with your general knowledge without making any distinctions."
        )
    else:
        instruction = (
            "You are a helpful and intelligent AI assistant. Answer the user's question clearly and helpfully. "
            "Do NOT mention any missing documents, databases, or search results."
        )

    if memory_context:
        instruction = memory_context + "\n\n" + instruction

    return instruction


class AIAgentService:

    @staticmethod
    def process_query(query, document_id, user, chat_id, chat_history=None, stream=False):
        try:
            user_id = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] query={query!r} doc={document_id!r} user={user_id}")

            if user and user.is_authenticated:
                has_docs = Document.objects.filter(user=user, embedding_status="done").exists()
            else:
                has_docs = Document.objects.filter(embedding_status="done").exists()

            # ── Semantic cache check ────────────────────────────────────────────
            if document_id and not _is_casual(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] cache hit  score={cached.get('score', '?'):.3f}")
                    answer  = cached.get("answer", "")
                    sources = cached.get("sources", [])
                    score   = cached.get("score", 0.0)
                    if stream:
                        # First chunk is a cache-hit marker — consumer strips it
                        yield f"__CACHE_HIT_{score:.2f}__"
                        for word in answer.split(" "):
                            yield word + " "
                        return
                    return answer

            # ── Memory context ──────────────────────────────────────────────────
            memory_context = ""
            if user_id and has_docs and not _is_casual(query):
                memory_context = _get_memory_context(user_id, query)

            # ── Audio document path ─────────────────────────────────────────────
            if document_id and str(document_id).startswith("audio_"):
                transcript = ""
                doc_title = ""
                try:
                    doc_pk = int(str(document_id).replace("audio_", ""))
                    doc_obj = Document.objects.get(id=doc_pk, user=user)
                    transcript = doc_obj.content or ""
                    doc_title = doc_obj.title or ""
                except Exception as e:
                    print(f"[{chat_id}] audio doc load error: {e}")

                client = _gemini_client()
                no_transcript = not transcript.strip() or transcript.startswith("Audio file")

                if not no_transcript:
                    system = (
                        "You are a helpful assistant. Answer based on the audio transcript below. "
                        "Be specific and quote relevant parts where helpful."
                    )
                    message = f"Transcript:\n\n{transcript}\n\nQuestion: {query}"
                else:
                    system = (
                        "You are a helpful assistant. Answer based on your general knowledge. "
                        "The audio transcript is not available."
                    )
                    message = f'Audio file: "{doc_title}"\n\nQuestion: {query}'

                if memory_context:
                    system = memory_context + "\n\n" + system

                answer = _generate(client, system, message)
                if answer:
                    if user_id is not None:
                        semantic_cache_set(
                            query, answer, document_id,
                            user_id=user_id, sources=[],
                        )
                    return answer

                return "I couldn't answer your question about this audio. Please try again."

            # ── RAG path with hybrid search ─────────────────────────────────────
            rag_context = ""
            sources = []
            if not _is_casual(query):
                try:
                    rag_context, sources = embedding_service.search_with_sources(query, document_id or None)
                    print(f"[{chat_id}] RAG context: {len(rag_context)} chars, sources: {sources}")
                except Exception as e:
                    print(f"[{chat_id}] Pinecone search error: {e}")

            client = _gemini_client()
            has_context = (
                rag_context
                and "No relevant" not in rag_context
                and "No sufficiently" not in rag_context
            )

            system  = _build_rag_system_prompt(has_context, memory_context)
            message = f"Document content:\n\n{rag_context}\n\nUser question:\n{query}" if has_context else query

            generator = _generate(client, system, message, chat_history=chat_history, stream=stream)

            if stream:
                full_answer = ""
                for chunk in generator:
                    full_answer += chunk
                    yield chunk

                if full_answer and has_context and should_cache(query, has_document_context=True) and user_id:
                    semantic_cache_set(
                        query, full_answer, document_id,
                        user_id=user_id, sources=sources,
                    )
                return

            else:
                answer = generator
                if answer and has_context and should_cache(query, has_document_context=True) and user_id:
                    semantic_cache_set(
                        query, answer, document_id,
                        user_id=user_id, sources=sources,
                    )
                return answer or "I'm sorry, I couldn't generate a response. Please try again."

        except Exception as e:
            print(f"[{chat_id}] unhandled error: {e}")
            traceback.print_exc()
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                return "The AI model is temporarily overloaded. Please try again in a few seconds."
            if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
                return "The AI service quota has been exceeded. Please try again later."
            if "400" in err or "INVALID_ARGUMENT" in err:
                return "There was an issue with your request. Please try rephrasing your question."
            return "An unexpected error occurred. Please try again."

