"""
semantic_cache.py
-----------------
Semantic Cache using SentenceTransformer embeddings + cosine similarity on Redis.

How it works:
  WRITE: After Gemini responds using actual document search, embed the query and store:
      redis key -> chat:emb:{user_id}:{uuid}
      fields    -> query, answer, document_id, user_id, embedding (JSON list)

  READ: On a new query, embed it and scan only THIS user's stored embeddings.
        If cosine similarity >= THRESHOLD -> return cached answer.
        Otherwise -> call Gemini as normal.

Important:
  - Cache is per-user. One user never sees another user's cached data.
  - We only cache answers that came from actual document search (not casual chat).
  - Invalidation removes only the affected user's entries for a given document.
"""

import json
import math
import uuid
import os

from .redis_client import redis_client

# how similar does a query have to be before we treat it as the same question?
SIMILARITY_THRESHOLD = 0.70

# key prefix includes the user_id slot so each user gets their own namespace
EMB_KEY_PREFIX = "chat:emb:"

# cache each answer for 1 hour
EMB_TTL = 3600


def _build_key_prefix(user_id):
    # separate cache namespace per user - users can't see each other's data
    return f"{EMB_KEY_PREFIX}{user_id}:"


def _get_embedding(text: str) -> list:
    """Call the FastAPI /embed endpoint to get an embedding vector for the text."""
    try:
        import requests
        fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
        embed_api = fastapi_url.replace("/upload", "") + "/embed"

        res = requests.post(embed_api, json={"text": text}, timeout=15)
        if res.ok:
            return res.json().get("embedding")
        else:
            print(f"[SemanticCache] FastAPI /embed returned {res.status_code}")
            return None
    except Exception as e:
        print(f"[SemanticCache] Remote embedding failed: {e}")
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity without numpy."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def semantic_cache_get(query: str, document_id: str, user_id=None):
    """
    Look for a semantically similar answer in this user's cache.
    Returns the cached answer string, or None if nothing matches well enough.
    """
    if redis_client is None:
        return None

    # without a user we have no namespace to search in
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

            # only match entries for the same document context
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
            print(f"[SemanticCache] Cache HIT (similarity={best_score:.3f}) user={user_id}")
            return best_answer

        print(f"[SemanticCache] Cache MISS (best={best_score:.3f}) user={user_id}")
        return None

    except Exception as e:
        print(f"[SemanticCache] Read error: {e}")
        return None


def semantic_cache_set(query: str, answer: str, document_id: str, user_id=None) -> None:
    """
    Store a question/answer pair in the semantic cache, scoped to this user.
    Only call this when the answer actually came from the document search tool.
    """
    if redis_client is None or not answer:
        return

    # no user means we can't scope it properly, skip
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
        redis_client.set(key, entry, ex=EMB_TTL)
        print(f"[SemanticCache] Stored key={key!r} query={query!r} user={user_id}")
    except Exception as e:
        print(f"[SemanticCache] Write error: {e}")


def get_user_cache_entries(user_id) -> list:
    """
    Fetch all cache entries belonging to a specific user.
    Used by the cache dashboard view.
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
                entry = json.loads(raw)
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
        print(f"[SemanticCache] Error reading cache dashboard entries: {e}")

    return entries


def clear_user_cache(user_id) -> int:
    """Delete all cached entries for a specific user only."""
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
    Remove cached entries tied to a specific document.
    If user_id is given, only clears that user's entries.
    """
    if redis_client is None:
        return 0

    deleted = 0
    try:
        # scope to this user if we know who they are, else scan everything (fallback)
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
            print(f"[SemanticCache] Invalidated {deleted} entries for document={document_id!r} user={user_id}")

    except Exception as e:
        print(f"[SemanticCache] Invalidation error: {e}")

    return deleted
