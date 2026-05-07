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

# phrases we never want to cache — matching any of these skips both read and write
_NO_CACHE_PHRASES = {
    "hi", "hello", "hey", "hii", "helo",
    "good morning", "good evening", "good afternoon",
    "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye",
    "great", "nice", "cool", "awesome", "got it",
}

# question patterns that refer to conversation history — never cache these
_HISTORY_KEYWORDS = (
    "previous conversation", "last message", "what did we talk",
    "summarize our", "summarize the chat", "earlier you said",
    "what was my", "remind me", "our conversation",
)


# Greeting words that make a query non-cacheable even if combined in a longer string.
# e.g. "hi how are you" contains "hi" so it must NOT be cached.
_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay",
}

# Very short queries (below this word count) are almost always small talk.
# "what is f1 score" = 4 words → cached. "hi there" = 2 words → not cached.
_MIN_WORDS_TO_CACHE = 3


def should_cache(query: str, has_document_context: bool) -> bool:
    """
    Returns True only when it makes sense to cache this answer.

    Rules (all must pass):
      1. The answer must have come from an actual uploaded document.
      2. The query must not be an exact match for a small-talk phrase.
      3. None of the individual words can be a greeting word.
      4. The query must be at least 3 words long.
      5. The query must not reference conversation history or summaries.
    """
    if not has_document_context:
        return False

    q = query.strip().lower().rstrip("!?.,:;")

    # exact full-phrase match
    if q in _NO_CACHE_PHRASES:
        return False

    words = q.split()

    # any greeting word anywhere in the query → skip
    if any(w in _GREETING_WORDS for w in words):
        return False

    # too short to be a real knowledge question
    if len(words) < _MIN_WORDS_TO_CACHE:
        return False

    # references to conversation history
    for pattern in _HISTORY_KEYWORDS:
        if pattern in q:
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
