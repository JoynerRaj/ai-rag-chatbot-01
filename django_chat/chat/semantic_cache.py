# Redis-backed semantic cache for document RAG answers.
#
# Fixes applied:
#   1. Query normalization  — lowercase + strip punctuation before embedding
#   2. Lower threshold      — 0.88 (was 0.92) so semantically similar phrasing hits cache
#   3. Duplicate prevention — skip storing if a similar entry already exists
#   4. Clean entry format   — answer and sources are separate flat fields (no nested dicts)
#
# Cache key format:  rag:doc:{user_id}:{uuid}
# Each entry expires after 2 hours and is private per user.

import json
import math
import uuid
import re

from .redis_client import redis_client

# ── Tunable constants ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.88   # lowered from 0.92 — catches paraphrases
RAG_KEY_PREFIX       = "rag:doc:"
RAG_TTL              = 7200   # 2 hours

# ── Guard words — same as before ────────────────────────────────────────────
_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay", "thanks", "thank",
    "fav", "favorite", "favourite",
    "color", "colour", "food", "hobby", "hobbies",
}

_MEMORY_TRIGGER_WORDS = {
    "summarize", "summarise", "remind",
    "conversation", "conversations",
}

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

_PREVIOUS_PREFIXES = {"previ", "prvio", "previo"}


# ── Query normalization ──────────────────────────────────────────────────────

def _normalize_query(query: str) -> str:
    """
    Normalize a query so that trivially different phrasings produce the
    same (or very close) embedding vector.

    Steps:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove punctuation (keeps letters, digits, spaces)
      4. Collapse multiple spaces into one

    Examples:
      "What is AI?"              -> "what is ai"
      "Artificial Intelligence!" -> "artificial intelligence"
      "explain about  AI  "      -> "explain about ai"
    """
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)   # remove punctuation
    q = re.sub(r"\s+", " ", q)       # collapse spaces
    return q.strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

def should_cache(query: str, has_document_context: bool) -> bool:
    """
    Gate: only cache document-grounded answers to substantive questions.
    Returns True only when ALL conditions are satisfied.
    """
    if not has_document_context:
        return False

    q = _normalize_query(query)
    word_set = set(q.split())

    if len(word_set) < 3:
        return False

    if word_set & _GREETING_WORDS:
        return False

    if word_set & _MEMORY_TRIGGER_WORDS:
        return False

    if any(w[:5] in _PREVIOUS_PREFIXES for w in word_set if len(w) >= 5):
        return False

    for pair in _MEMORY_WORD_PAIRS:
        if pair.issubset(word_set):
            return False

    return True


def _build_key_prefix(user_id) -> str:
    return f"{RAG_KEY_PREFIX}{user_id}:"


def _get_embedding(text: str):
    """Embed text using the project's Gemini embedding model."""
    try:
        from chat.services.embedding_service import _embed_text
        return _embed_text(text)
    except Exception as e:
        print(f"[SemanticCache] Embedding failed: {e}")
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity — no numpy dependency."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Core cache operations ────────────────────────────────────────────────────

def semantic_cache_get(query: str, document_id: str, user_id=None):
    """
    Look up a cached answer for this query.

    The query is normalized before embedding so "What is AI?" and
    "what is ai" produce the same lookup vector.

    Returns a dict {"answer": str, "sources": list, "score": float}
    if a similar cached entry exists, otherwise None.
    """
    if redis_client is None or user_id is None:
        return None

    # Normalize BEFORE embedding — key fix for case/punctuation mismatches
    normalized = _normalize_query(query)
    query_emb  = _get_embedding(normalized)
    if query_emb is None:
        return None

    try:
        prefix     = _build_key_prefix(user_id)
        keys       = redis_client.keys(f"{prefix}*")
        best_score = -1.0
        best_entry = None

        for key in keys:
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except Exception:
                continue

            # Each cache entry is per-document; skip mismatches
            if entry.get("document_id") != document_id:
                continue

            stored_emb = entry.get("embedding")
            if not stored_emb:
                continue

            score = _cosine_similarity(query_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= SIMILARITY_THRESHOLD and best_entry:
            # Safely extract answer — handle legacy nested-dict format
            answer = best_entry.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("answer", "")

            sources = best_entry.get("sources", [])
            print(
                f"[SemanticCache] HIT  score={best_score:.3f}  "
                f"user={user_id}  query={normalized!r}"
            )
            return {"answer": answer, "sources": sources, "score": best_score}

        print(f"[SemanticCache] MISS  best={best_score:.3f}  user={user_id}  query={normalized!r}")
        return None

    except Exception as e:
        print(f"[SemanticCache] Read error: {e}")
        return None


def semantic_cache_set(
    query: str,
    answer,
    document_id: str,
    user_id=None,
    sources: list = None,
) -> None:
    """
    Save a question/answer pair to the cache.

    Before saving, check if a semantically equivalent entry already exists
    (score >= SIMILARITY_THRESHOLD). If so, skip the insert to prevent
    duplicate cache entries for paraphrased questions.

    The stored format is flat and clean:
        {query, normalized_query, answer (str), sources (list),
         document_id, user_id, embedding}
    """
    if redis_client is None:
        return
    if user_id is None:
        return

    # Unwrap legacy nested-dict answers from previous code version
    if isinstance(answer, dict):
        sources = answer.get("sources", sources or [])
        answer  = answer.get("answer", "")
    if not answer:
        return

    sources    = sources or []
    normalized = _normalize_query(query)

    emb = _get_embedding(normalized)
    if emb is None:
        return

    try:
        # ── Duplicate-prevention check ────────────────────────────────────
        # Before writing a new entry, scan existing entries for this user+doc.
        # If one already has similarity >= threshold, skip the write entirely.
        prefix = _build_key_prefix(user_id)
        keys   = redis_client.keys(f"{prefix}*")

        for key in keys:
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                existing = json.loads(raw)
            except Exception:
                continue

            if existing.get("document_id") != document_id:
                continue

            stored_emb = existing.get("embedding")
            if not stored_emb:
                continue

            score = _cosine_similarity(emb, stored_emb)
            if score >= SIMILARITY_THRESHOLD:
                print(
                    f"[SemanticCache] SKIP duplicate  score={score:.3f}  "
                    f"existing={existing.get('normalized_query', '')!r}  "
                    f"new={normalized!r}"
                )
                return   # similar entry already exists — don't pollute the cache

        # ── Store new entry ───────────────────────────────────────────────
        entry = json.dumps({
            "query":            query,       # original text (for display)
            "normalized_query": normalized,  # used for similarity lookup
            "answer":           answer,      # always a plain string
            "sources":          sources,     # list of {file_name, page_num}
            "document_id":      document_id,
            "user_id":          user_id,
            "embedding":        emb,
        })
        key = f"{prefix}{uuid.uuid4().hex}"
        redis_client.set(key, entry, ex=RAG_TTL)
        print(
            f"[SemanticCache] SAVED  key={key!r}  "
            f"query={normalized!r}  user={user_id}"
        )

    except Exception as e:
        print(f"[SemanticCache] Write error: {e}")


# ── Dashboard helpers ────────────────────────────────────────────────────────

def get_user_cache_entries(user_id) -> list:
    """Return all cache entries for a user — used by the cache dashboard page."""
    if redis_client is None or user_id is None:
        return []

    entries = []
    try:
        prefix = _build_key_prefix(user_id)
        keys   = redis_client.keys(f"{prefix}*")
        for k in keys:
            raw = redis_client.get(k)
            ttl = redis_client.ttl(k)
            if not raw:
                continue
            try:
                entry = json.loads(raw)

                # Safely extract answer — handle both old and new format
                answer = entry.get("answer", "")
                if isinstance(answer, dict):
                    answer = answer.get("answer", "")

                query   = entry.get("query", "")
                doc_id  = entry.get("document_id") or "(all documents)"
                sources = entry.get("sources", [])
                preview = answer[:300] + "..." if len(answer) > 300 else answer

                entries.append({
                    "key":     k,
                    "query":   query,
                    "value":   preview,
                    "doc_id":  doc_id,
                    "sources": sources,
                    "ttl":     ttl if ttl > 0 else "Expired",
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
        keys   = redis_client.keys(f"{prefix}*")
        if keys:
            redis_client.delete(*keys)
            print(f"[SemanticCache] Cleared {len(keys)} entries for user={user_id}")
            return len(keys)
    except Exception as e:
        print(f"[SemanticCache] Clear error for user={user_id}: {e}")
    return 0


def invalidate_by_document(document_id: str, user_id=None) -> int:
    """Remove all cached entries that belong to a specific document."""
    if redis_client is None:
        return 0

    deleted = 0
    try:
        prefix    = _build_key_prefix(user_id) if user_id else RAG_KEY_PREFIX
        keys      = redis_client.keys(f"{prefix}*")
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
            print(
                f"[SemanticCache] Removed {deleted} entries for "
                f"doc={document_id!r} user={user_id}"
            )

    except Exception as e:
        print(f"[SemanticCache] Invalidation error: {e}")

    return deleted
