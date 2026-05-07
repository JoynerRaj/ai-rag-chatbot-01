# Redis-backed cache for document RAG answers only.
# We intentionally skip caching for:
#   - greetings / small talk  (hi, thanks, bye ...)
#   - memory / history questions  (what did we talk about, summarize ...)
#   - anything that isn't grounded in an uploaded document
#
# Cache key format:  rag:doc:{user_id}:{uuid}
# Each entry expires after 2 hours and is private per user.

import json
import math
import uuid
import os

from .redis_client import redis_client

SIMILARITY_THRESHOLD = 0.92
RAG_KEY_PREFIX = "rag:doc:"
RAG_TTL = 7200  # 2 hours

# Greeting words — if any of these appear in the query, it is not a document question
_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay", "thanks", "thank",
}

# Single words that indicate a memory/history request — block caching immediately
_MEMORY_TRIGGER_WORDS = {
    "summarize", "summarise", "remind",
    # 'conversation' in a chatbot almost always means chat history, not document content
    "conversation", "conversations",
}

# Word pairs: if BOTH words appear in the query, treat it as a memory request.
_MEMORY_WORD_PAIRS = [
    {"previous", "convo"},
    {"previous", "chat"},
    {"last", "message"},
    {"last", "conversation"},
    {"earlier", "said"},
    {"our", "conversation"},
    {"our", "chat"},
    {"tell", "previous"},
]

# Prefix fragments that identify "previous" and its common typos.
# "previous" -> previ, "prvious" -> prvio, "previos" -> previ
_PREVIOUS_PREFIXES = {"previ", "prvio", "previo"}


def should_cache(query: str, has_document_context: bool) -> bool:
    """
    Strict document-only cache gate.

    Returns True only when ALL of the following are true:
      - The AI answer was grounded in an uploaded document
      - The query is at least 3 words long
      - The query contains no greeting words
      - The query does not reference conversation history
        (checked with exact words, word-pairs, AND fuzzy prefix for 'previous' typos)
    """
    if not has_document_context:
        return False

    q = query.strip().lower().rstrip("!?.,:;")
    word_set = set(q.split())

    # Too short to be a real document question
    if len(word_set) < 3:
        return False

    # Any greeting word anywhere → not a document question
    if word_set & _GREETING_WORDS:
        return False

    # Single memory trigger words (summarize, conversation, remind, etc.)
    if word_set & _MEMORY_TRIGGER_WORDS:
        return False

    # Fuzzy prefix check for "previous" and its typos (prvious, previos, previuos ...)
    if any(w[:5] in _PREVIOUS_PREFIXES for w in word_set if len(w) >= 5):
        return False

    # Word-pair check for other memory patterns
    for pair in _MEMORY_WORD_PAIRS:
        if pair.issubset(word_set):
            return False

    return True


def _build_key_prefix(user_id):
    return f"{RAG_KEY_PREFIX}{user_id}:"


def _get_embedding(text: str) -> list:
    """Get an embedding vector for the text using Google's embedding model."""
    try:
        from chat.services.embedding_service import _embed_text
        return _embed_text(text)
    except Exception as e:
        print(f"[SemanticCache] Embedding failed: {e}")
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Standard cosine similarity - no numpy needed."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def semantic_cache_get(query: str, document_id: str, user_id=None):
    """
    Check if we already have a good answer for this question.
    Returns the cached answer if similarity is high enough, otherwise None.
    """
    if redis_client is None:
        return None

    if user_id is None:
        return None

    query_emb = _get_embedding(query)
    if query_emb is None:
        return None

    try:
        prefix = _build_key_prefix(user_id)
        keys = redis_client.keys(f"{prefix}*")
        best_score  = -1.0
        best_answer = None

        for key in keys:
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except Exception:
                continue

            # different document means different content, skip it
            if entry.get("document_id") != document_id:
                continue

            stored_emb = entry.get("embedding")
            if not stored_emb:
                continue

            score = _cosine_similarity(query_emb, stored_emb)
            if score > best_score:
                best_score  = score
                best_answer = entry.get("answer")

        if best_score >= SIMILARITY_THRESHOLD and best_answer:
            print(f"[SemanticCache] HIT (score={best_score:.3f}) user={user_id}")
            return best_answer

        print(f"[SemanticCache] MISS (best={best_score:.3f}) user={user_id}")
        return None

    except Exception as e:
        print(f"[SemanticCache] Read error: {e}")
        return None


def semantic_cache_set(query: str, answer: str, document_id: str, user_id=None) -> None:
    """
    Save a question/answer pair to the cache for this user.
    Only call this when the answer actually came from the documents.
    """
    if redis_client is None or not answer:
        return

    if user_id is None:
        return

    emb = _get_embedding(query)
    if emb is None:
        return

    try:
        entry = json.dumps({
            "query":       query,
            "answer":      answer,
            "document_id": document_id,
            "user_id":     user_id,
            "embedding":   emb,
        })
        prefix = _build_key_prefix(user_id)
        key = f"{prefix}{uuid.uuid4().hex}"
        redis_client.set(key, entry, ex=RAG_TTL)
        print(f"[SemanticCache] Saved key={key!r} query={query!r} user={user_id}")
    except Exception as e:
        print(f"[SemanticCache] Write error: {e}")


def get_user_cache_entries(user_id) -> list:
    """
    Pull all cached entries for one user - used by the cache dashboard page.
    """
    if redis_client is None or user_id is None:
        return []

    entries = []
    try:
        prefix = _build_key_prefix(user_id)
        keys = redis_client.keys(f"{prefix}*")
        for k in keys:
            raw = redis_client.get(k)
            ttl = redis_client.ttl(k)
            if not raw:
                continue
            try:
                entry   = json.loads(raw)
                query   = entry.get("query", "")
                answer  = entry.get("answer", "")
                doc_id  = entry.get("document_id") or "(all documents)"
                preview = answer if len(answer) < 300 else answer[:300] + "..."
                entries.append({
                    "key":    k,
                    "query":  query,
                    "value":  preview,
                    "doc_id": doc_id,
                    "ttl":    ttl if ttl > 0 else "Expired",
                })
            except Exception:
                pass
    except Exception as e:
        print(f"[SemanticCache] Dashboard read error: {e}")

    return entries


def clear_user_cache(user_id) -> int:
    """Wipe all cached entries for a single user."""
    if redis_client is None or user_id is None:
        return 0
    try:
        prefix = _build_key_prefix(user_id)
        keys = redis_client.keys(f"{prefix}*")
        if keys:
            redis_client.delete(*keys)
            print(f"[SemanticCache] Cleared {len(keys)} entries for user={user_id}")
            return len(keys)
    except Exception as e:
        print(f"[SemanticCache] Clear error for user={user_id}: {e}")
    return 0


def invalidate_by_document(document_id: str, user_id=None) -> int:
    """
    Remove any cached entries that belong to a specific document.
    Pass user_id to limit this to one person's cache, or leave it None to clear all.
    """
    if redis_client is None:
        return 0

    deleted = 0
    try:
        # narrow it down to just this user if we know who they are
        prefix = _build_key_prefix(user_id) if user_id else EMB_KEY_PREFIX
        keys = redis_client.keys(f"{prefix}*")
        to_delete = []
        for key in keys:
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if entry.get("document_id") == document_id:
                to_delete.append(key)

        if to_delete:
            redis_client.delete(*to_delete)
            deleted = len(to_delete)
            print(f"[SemanticCache] Removed {deleted} entries for doc={document_id!r} user={user_id}")

    except Exception as e:
        print(f"[SemanticCache] Invalidation error: {e}")

    return deleted
