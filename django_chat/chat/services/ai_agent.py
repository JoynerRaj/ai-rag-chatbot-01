import os
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
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


def _generate(client, system_instruction, user_message, chat_history=None):
    """Build a contents list and call Gemini. Returns the answer text or ''."""
    contents = []
    if chat_history:
        for turn in chat_history:
            contents.append(types.Content(role="user", parts=[types.Part(text=turn["question"])]))
            contents.append(types.Content(role="model", parts=[types.Part(text=turn["answer"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    config = types.GenerateContentConfig(system_instruction=system_instruction)
    for model in ("gemini-2.5-flash", "gemini-2.0-flash"):
        try:
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
            raise
    return ""


class AIAgentService:

    @staticmethod
    def process_query(query, document_id, user, chat_id, chat_history=None):
        try:
            user_id = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] query={query!r}  doc={document_id!r}  user={user_id}")

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

            if not _is_casual(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] cache hit")
                    return cached

            # Audio document — try event RAG first, then fall back to the stored transcript
            if document_id and str(document_id).startswith("audio_"):
                try:
                    audio_answer = FastAPIClient.ask_audio(query)
                    if audio_answer and audio_answer.strip():
                        if user_id is not None:
                            semantic_cache_set(query, audio_answer, document_id, user_id=user_id)
                        return audio_answer
                except Exception as e:
                    print(f"[{chat_id}] audio event RAG error: {e}")

                # Load the transcript that was saved during upload
                transcript = ""
                doc_title = ""
                try:
                    doc_pk = int(str(document_id).replace("audio_", ""))
                    doc_obj = Document.objects.get(id=doc_pk, user=user)
                    transcript = doc_obj.content or ""
                    doc_title = doc_obj.title or ""
                except Exception as e:
                    print(f"[{chat_id}] could not load audio doc: {e}")

                client = _gemini_client()
                no_transcript = not transcript.strip() or transcript == "Audio file events mapped."

                if not no_transcript:
                    system = (
                        "You are a helpful assistant. Answer the user's question based only on the "
                        "audio transcript provided. Be specific and quote relevant parts where helpful."
                    )
                    message = f"Transcript:\n\n{transcript}\n\nQuestion: {query}"
                else:
                    system = (
                        "You are a helpful assistant. Answer based on your general knowledge. "
                        "The user has uploaded an audio file but the transcript is not available."
                    )
                    message = f'Audio file: "{doc_title}"\n\nQuestion: {query}'

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

            # Regular document — search Pinecone then ask Gemini
            rag_context = ""
            if not _is_casual(query):
                try:
                    rag_context = FastAPIClient.search_documents(query, document_id or "")
                    print(f"[{chat_id}] RAG: {len(rag_context)} chars")
                except Exception as e:
                    print(f"[{chat_id}] RAG search error: {e}")

            client = _gemini_client()
            has_context = (
                rag_context
                and "No relevant" not in rag_context
                and "No sufficiently" not in rag_context
            )

            if not has_context:
                if not _is_casual(query):
                    return "I couldn't find any information about that in the uploaded document."
                else:
                    system = "You are a helpful assistant. You have the conversation history for context. Answer the user's greeting."
                    message = query
            else:
                system = (
                    "You are a helpful assistant. You have content from the user's uploaded documents "
                    "and the conversation history. Base your answer on the document content provided."
                )
                message = (
                    f"Document content:\n\n{rag_context}\n\n"
                    f"Answer this using the content above:\n{query}"
                )

            answer = _generate(client, system, message, chat_history=chat_history)

            if answer and has_context and user_id is not None:
                semantic_cache_set(query, answer, document_id, user_id=user_id)

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
