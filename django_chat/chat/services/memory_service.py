"""
memory_service.py

MemPalace-inspired conversational memory management for the RAG chatbot.

Concepts adapted from MemPalace (https://github.com/MemPalace/mempalace):
  - Verbatim storage of important conversation facts
  - Semantic search over stored memories for relevant recall
  - Per-user isolation (wing == user in this context)
  - Non-blocking background storage — never slows the response path

Pipeline:
  1. After each substantive Q&A exchange, Gemini extracts 1-3 key facts.
  2. Each fact is embedded and upserted to Pinecone with metadata
     { type="mem", user_id=X, session_id=Y }.
  3. Before answering a new query, we search Pinecone for related memories
     and inject them into the system prompt as background context.
  4. A MemoryEntry row in Django DB mirrors every stored fact so users can
     view and manage their memories from the UI.
"""

import os
import uuid
import threading
import traceback

from google import genai
from chat.services.embedding_service import _embed_text, _get_pinecone_index


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MEMORY_TYPE_TAG  = "mem"       # Pinecone metadata tag separating memories from docs
MEMORY_TOP_K     = 5           # max memories to retrieve per query
MEMORY_THRESHOLD = 0.55        # slightly relaxed to catch more relevant context

# Short trivial exchanges we never bother to remember
_TRIVIAL_PHRASES = {
    "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you",
    "ok", "okay", "sure", "cool", "nice", "great", "good", "awesome",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_worth_remembering(question: str, answer: str) -> bool:
    """
    Cheap pre-filter: skip greetings and very short exchanges.
    """
    q = question.strip().lower().rstrip("!?.,:;")
    if q in _TRIVIAL_PHRASES:
        return False
    if len(question.split()) < 2:  # allow 2+ words
        return False
    if len(answer.strip()) < 15:   # allow 15+ chars
        return False
    return True



def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})


# ──────────────────────────────────────────────────────────────────────────────
# Memory extraction (Gemini)
# ──────────────────────────────────────────────────────────────────────────────

def extract_memories(question: str, answer: str) -> list:
    """
    Use Gemini to distil 1-3 concise, self-contained facts from a Q&A exchange.

    Returns a list of short factual strings suitable for later semantic recall.
    Returns an empty list silently on any failure (non-critical path).
    """
    try:
        client = _gemini_client()
        prompt = (
            "You are a memory extractor. From the conversation below, extract "
            "1 to 3 concise, self-contained factual statements that are worth "
            "remembering for future conversations. Each statement must be a "
            "single sentence that can be understood without additional context. "
            "Focus on facts, decisions, preferences, or key conclusions. "
            "Skip filler, greetings, and meta-commentary.\n\n"
            "Return ONLY the statements, one per line, no bullets, no numbering.\n\n"
            f"User asked: {question}\n\n"
            f"Assistant answered: {answer[:1500]}\n\n"
            "Facts (one per line):"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        text = response.text or ""
        facts = [line.strip() for line in text.strip().splitlines() if line.strip()]
        valid_facts = [f for f in facts if 10 < len(f) < 300][:3]  # cap at 3
        print(f"[memory] extracted {len(valid_facts)} facts from exchange")
        return valid_facts

    except Exception as e:
        print(f"[memory] extract_memories failed (non-critical): {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Pinecone storage
# ──────────────────────────────────────────────────────────────────────────────

def _store_single_memory(user_id: int, session_id: int, fact: str, question: str) -> str | None:
    """
    Embed a single fact and upsert it to Pinecone.
    Returns the vector ID on success, None on failure.
    """
    try:
        embedding = _embed_text(fact)
        index     = _get_pinecone_index()
        vid       = str(uuid.uuid4())

        index.upsert(vectors=[{
            "id":     vid,
            "values": embedding,
            "metadata": {
                "type":       MEMORY_TYPE_TAG,
                "user_id":    str(user_id),
                "session_id": str(session_id),
                "fact":       fact,
                "question":   question[:200],
            },
        }])

        print(f"[memory] stored → user={user_id} id={vid[:8]}… fact={fact[:60]!r}")
        return vid

    except Exception as e:
        print(f"[memory] _store_single_memory error: {e}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API — write path
# ──────────────────────────────────────────────────────────────────────────────

def store_exchange_async(user_id: int, session_id: int, question: str, answer: str) -> None:
    """
    Non-blocking memory storage.

    Spawns a daemon thread that:
      1. Checks whether the exchange is worth remembering.
      2. Calls Gemini to extract 1-3 key facts.
      3. Embeds each fact and upserts it to Pinecone.
      4. Creates a MemoryEntry row in Django DB for the management UI.

    Failures are printed but never raised — the chat response is already sent.
    """
    if not _is_worth_remembering(question, answer):
        print(f"[memory] skipping trivial exchange for user={user_id}")
        return

    def _run():
        # MUST close stale connections at thread start — Django connections are
        # thread-local, so the new thread inherits a closed/stale handle from
        # the parent. close_old_connections() forces Django to open a fresh one.
        from django.db import close_old_connections
        close_old_connections()
        try:
            from chat.models import MemoryEntry
            facts = extract_memories(question, answer)
            saved = 0
            for fact in facts:
                vid = _store_single_memory(user_id, session_id, fact, question)
                if vid:
                    MemoryEntry.objects.create(
                        user_id=user_id,
                        session_id=session_id,
                        content=fact,
                        source_question=question[:500],
                        pinecone_id=vid,
                    )
                    saved += 1
            print(f"[memory] saved {saved}/{len(facts)} memories for user={user_id}")
        except Exception as e:
            print(f"[memory] background thread error: {e}")
            traceback.print_exc()
        finally:
            close_old_connections()

    threading.Thread(target=_run, daemon=True).start()



# ──────────────────────────────────────────────────────────────────────────────
# Public API — read path
# ──────────────────────────────────────────────────────────────────────────────

def search_memories(user_id: int, query: str, n: int = MEMORY_TOP_K) -> list:
    """
    Semantic search over this user's stored memories in Pinecone.

    Returns a list of dicts: [{fact, question, session_id, score}, ...]
    Returns an empty list silently on failure.
    """
    try:
        query_vec = _embed_text(query)
        index     = _get_pinecone_index()

        results = index.query(
            vector=query_vec,
            top_k=n,
            include_metadata=True,
            filter={
                "type":    {"$eq": MEMORY_TYPE_TAG},
                "user_id": {"$eq": str(user_id)},
            },
        )

        memories = []
        for m in results.matches:
            if m.score >= MEMORY_THRESHOLD and m.metadata:
                memories.append({
                    "fact":       m.metadata.get("fact", ""),
                    "question":   m.metadata.get("question", ""),
                    "session_id": m.metadata.get("session_id", ""),
                    "score":      round(m.score, 3),
                })

        print(f"[memory] search user={user_id} → {len(memories)} relevant memories (query={query[:40]!r})")
        return memories

    except Exception as e:
        print(f"[memory] search_memories error: {e}")
        return []


def build_memory_context(user_id: int, query: str) -> str:
    """
    Build a concise memory block to prepend to the AI system prompt.

    Searches for memories semantically related to the current query and
    formats them as a short bullet list.  Returns empty string when no
    relevant memories exist (so the prompt stays clean for new users).
    """
    memories = search_memories(user_id, query)
    if not memories:
        return ""

    lines = ["[Relevant context from your past conversations with this user:]"]
    for mem in memories:
        lines.append(f"• {mem['fact']}")

    context = "\n".join(lines)
    print(f"[memory] built context block with {len(memories)} memories for user={user_id}")
    return context


def get_user_memory_entries(user_id: int) -> list:
    """
    Return all MemoryEntry rows for a user, newest first.
    Used by the memory management page.
    """
    try:
        from chat.models import MemoryEntry
        entries = MemoryEntry.objects.filter(user_id=user_id).order_by("-created_at")
        return list(entries)
    except Exception as e:
        print(f"[memory] get_user_memory_entries error: {e}")
        return []


def delete_memory_entry(memory_id: int, user_id: int) -> bool:
    """
    Delete a single MemoryEntry (and its Pinecone vector) for the given user.
    Returns True on success.
    """
    try:
        from chat.models import MemoryEntry
        entry = MemoryEntry.objects.get(id=memory_id, user_id=user_id)

        # Remove the Pinecone vector
        if entry.pinecone_id:
            try:
                index = _get_pinecone_index()
                index.delete(ids=[entry.pinecone_id])
                print(f"[memory] deleted Pinecone vector {entry.pinecone_id}")
            except Exception as e:
                print(f"[memory] Pinecone delete error (continuing): {e}")

        entry.delete()
        return True

    except Exception as e:
        print(f"[memory] delete_memory_entry error: {e}")
        return False


def clear_user_memories(user_id: int) -> int:
    """
    Wipe ALL memories for a user — both Pinecone vectors and DB rows.
    Returns the number of DB rows deleted.
    """
    try:
        # Remove from Pinecone
        index = _get_pinecone_index()
        index.delete(filter={
            "type":    {"$eq": MEMORY_TYPE_TAG},
            "user_id": {"$eq": str(user_id)},
        })
        print(f"[memory] cleared Pinecone memories for user={user_id}")
    except Exception as e:
        print(f"[memory] Pinecone clear error for user={user_id}: {e}")

    try:
        from chat.models import MemoryEntry
        count, _ = MemoryEntry.objects.filter(user_id=user_id).delete()
        print(f"[memory] deleted {count} MemoryEntry rows for user={user_id}")
        return count
    except Exception as e:
        print(f"[memory] DB clear error for user={user_id}: {e}")
        return 0
