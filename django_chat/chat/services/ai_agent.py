"""
ai_agent.py

The core RAG pipeline. Given a user query, this module:
  1. Checks the semantic cache — if there's a sufficiently similar cached answer, return it.
  2. Retrieves relevant memories from past conversations (MemPalace-inspired) and injects
     them into the system prompt so the AI has personal context.
  3. Searches Pinecone for relevant document chunks.
  4. Sends chunks + memory context + question to Gemini and streams (or returns) the answer.
  5. Caches the answer if it was genuinely grounded in an uploaded document.

Audio documents are handled separately: the stored transcript is used as context
instead of going through Pinecone.
"""

import os
import traceback

from google import genai
from google.genai import types

from chat.services import embedding_service
from chat.semantic_cache import semantic_cache_get, semantic_cache_set, should_cache
from chat.models import Document


# Short phrases that don't warrant a document search or cache lookup
CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
    "good afternoon", "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you",
    "great", "nice", "cool", "awesome", "got it", "understood",
}


def _is_casual(text: str) -> bool:
    return text.strip().lower().rstrip("!?.,:;") in CASUAL_PHRASES


def _gemini_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"api_version": "v1alpha"},
    )


def _generate(client, system_instruction, user_message, chat_history=None, stream=False):
    """
    Build a contents list and call Gemini.
    Yields text chunks if stream=True, otherwise returns the full response string.
    Falls back from gemini-2.5-flash to gemini-2.0-flash if the first model errors out.
    """
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
                response  = client.models.generate_content(model=model, contents=contents, config=config)
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
                yield f"\n\n> ⚠️ *[Error: {str(e)}]*"
                return
            raise

    if not stream:
        return ""


def _get_memory_context(user_id, query: str) -> str:
    """
    Retrieve semantically relevant memories for this user and query.
    Returns an empty string if the memory service is unavailable or finds nothing.
    This is always non-critical — a failure here must never break the chat response.
    """
    try:
        from chat.services.memory_service import build_memory_context
        return build_memory_context(user_id=user_id, query=query)
    except Exception as e:
        print(f"[ai_agent] memory context fetch failed (non-critical): {e}")
        return ""


class AIAgentService:

    @staticmethod
    def process_query(query, document_id, user, chat_id, chat_history=None, stream=False):
        try:
            user_id  = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] query={query!r}  doc={document_id!r}  user={user_id}")

            # Make sure the user has at least one embedded document before doing anything
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

            # Cache lookup — only when a specific document is selected and the query isn't casual
            if document_id and not _is_casual(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] cache hit for doc={document_id!r}")
                    if stream:
                        for word in cached.split(" "):
                            yield word + " "
                        return
                    return cached

            # ── MemPalace-inspired memory injection ───────────────────────────
            # Retrieve semantically relevant facts from the user's past conversations.
            # These are injected as a preamble to the system prompt so Gemini can
            # personalise its answers (e.g. "the user previously mentioned they work
            # in finance" or "the user prefers concise answers").
            memory_context = ""
            if user_id and not _is_casual(query):
                memory_context = _get_memory_context(user_id, query)
                if memory_context:
                    print(f"[{chat_id}] injecting memory context ({len(memory_context)} chars)")
            # ─────────────────────────────────────────────────────────────────

            # Audio document — use the stored transcript as context
            if document_id and str(document_id).startswith("audio_"):
                transcript = ""
                doc_title  = ""
                try:
                    doc_pk     = int(str(document_id).replace("audio_", ""))
                    doc_obj    = Document.objects.get(id=doc_pk, user=user)
                    transcript = doc_obj.content or ""
                    doc_title  = doc_obj.title or ""
                except Exception as e:
                    print(f"[{chat_id}] could not load audio doc: {e}")

                client        = _gemini_client()
                no_transcript = not transcript.strip() or transcript.startswith("Audio file")

                if not no_transcript:
                    system  = (
                        "You are a helpful assistant. Answer the user's question based only on the "
                        "audio transcript provided. Be specific and quote relevant parts where helpful."
                    )
                    message = f"Transcript:\n\n{transcript}\n\nQuestion: {query}"
                else:
                    system  = (
                        "You are a helpful assistant. Answer based on your general knowledge. "
                        "The user has uploaded an audio file but the transcript is not available."
                    )
                    message = f'Audio file: "{doc_title}"\n\nQuestion: {query}'

                if memory_context:
                    system = memory_context + "\n\n" + system

                answer = _generate(client, system, message)
                if answer:
                    if user_id is not None:
                        semantic_cache_set(query, answer, document_id, user_id=user_id)
                    if no_transcript:
                        answer += (
                            "\n\n> ⚠️ *This answer is from general knowledge — "
                            "the audio transcript was not available.*"
                        )
                    return answer

                return "I couldn't answer your question about this audio. Please try again."

            # Regular document — search Pinecone, then ask Gemini
            rag_context = ""
            if not _is_casual(query):
                try:
                    rag_context = embedding_service.search_documents(query, document_id or None)
                    print(f"[{chat_id}] RAG context: {len(rag_context)} chars")
                except Exception as e:
                    print(f"[{chat_id}] Pinecone search error: {e}")

            client      = _gemini_client()
            has_context = (
                rag_context
                and "No relevant"    not in rag_context
                and "No sufficiently" not in rag_context
            )

            if has_context:
                system  = (
                    "You are a helpful AI assistant with access to the user's uploaded documents. "
                    "Answer the question using the document content below. "
                    "If the document content doesn't cover the question, answer using your general knowledge "
                    "without mentioning the document."
                )
                message = f"Document content:\n\n{rag_context}\n\nUser question:\n{query}"
            else:
                system  = (
                    "You are a helpful assistant. No relevant content was found in the user's uploaded "
                    "documents. Answer the question using your general knowledge."
                )
                message = query

            # Prepend memory context so the AI is aware of the user's personal history
            if memory_context:
                system = memory_context + "\n\n" + system

            generator = _generate(client, system, message, chat_history=chat_history, stream=stream)

            if stream:
                full_answer = ""
                for chunk in generator:
                    full_answer += chunk
                    yield chunk

                if full_answer:
                    if has_context and should_cache(query, has_document_context=True) and user_id is not None:
                        semantic_cache_set(query, full_answer, document_id, user_id=user_id)
                        print(f"[{chat_id}] cached doc answer")
                return

            else:
                answer = generator
                if answer:
                    if has_context and should_cache(query, has_document_context=True) and user_id is not None:
                        semantic_cache_set(query, answer, document_id, user_id=user_id)
                        print(f"[{chat_id}] cached doc answer")
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
