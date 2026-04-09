"""
semantic_cache.py
-----------------
Semantic Cache-Aside layer using SentenceTransformer (all-MiniLM-L6-v2) + cosine similarity.

Uses the SAME model as FastAPI/Pinecone so embeddings are consistent (dim=384).

How it works:
  WRITE: After Gemini generates an answer, embed the query and store:
      redis key  →  chat:emb:{uuid}
      fields     →  query, answer, document_id, embedding (JSON list)

  READ: On a new query, embed it and scan all stored embeddings.
        If cosine similarity >= THRESHOLD → return that cached answer.
        Otherwise → call Gemini normally.

Advantage over exact-match cache:
  "what is AI" and "can you explain about AI" both get the same cached answer.
"""

import json
import math
import uuid
import os

from .redis_client import redis_client

# ── Config ───────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70   # Queries scoring >= 0.70 are treated as same question
EMB_KEY_PREFIX       = "chat:emb:"
EMB_TTL              = 3600   # 1 hour (seconds)

# ── Remote Embedding ────────────────────────────────────────────────────────────
# We call FastAPI to get embeddings instead of loading SentenceTransformer here
# to save memory (Django free tier limits).

def _get_embedding(text: str) -> list:
    """Embed text by calling FastAPI /embed (dim=384)."""
    try:
        import requests
        # Get FastAPI URL, same as from views
        fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
        embed_api = fastapi_url.replace("/upload", "") + "/embed"

        res = requests.post(embed_api, json={"text": text}, timeout=15)
        if res.ok:
            return res.json().get("embedding")
        else:
            print(f"[SemanticCache] ⚠️ FastAPI /embed returned {res.status_code}")
            return None
    except Exception as e:
        print(f"[SemanticCache] ⚠️ Remote Embedding error: {e}")
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity — no numpy required."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Public API ─────────────────────────────────────────────────────────────────
def semantic_cache_get(query: str, document_id: str):
    """
    Try to find a semantically similar cached answer.
    Returns the cached answer string, or None if no good match found.
    """
    if redis_client is None:
        return None

    query_emb = _get_embedding(query)
    if query_emb is None:
        return None

    try:
        keys = redis_client.keys(f"{EMB_KEY_PREFIX}*")
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

            # Only match entries for the same document scope
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
            print(f"[SemanticCache] ✅ Cache HIT  (similarity={best_score:.3f})")
            return best_answer

        print(f"[SemanticCache] ❌ Cache MISS (best similarity={best_score:.3f})")
        return None

    except Exception as e:
        print(f"[SemanticCache] ⚠️  Read error: {e}")
        return None


def semantic_cache_set(query: str, answer: str, document_id: str) -> None:
    """Embed the query and store the answer in the semantic cache."""
    if redis_client is None or not answer:
        return

    emb = _get_embedding(query)
    if emb is None:
        return

    try:
        entry = json.dumps({
            "query":       query,
            "answer":      answer,
            "document_id": document_id,
            "embedding":   emb,
        })
        key = f"{EMB_KEY_PREFIX}{uuid.uuid4().hex}"
        redis_client.set(key, entry, ex=EMB_TTL)
        print(f"[SemanticCache] 💾 Stored  key={key!r}  query={query!r}")
    except Exception as e:
        print(f"[SemanticCache] ⚠️  Write error: {e}")


def invalidate_by_document(document_id: str) -> int:
    """
    Delete all semantic cache entries that belong to a specific document.
    Returns the number of keys deleted.
    """
    if redis_client is None:
        return 0

    deleted = 0
    try:
        keys = redis_client.keys(f"{EMB_KEY_PREFIX}*")
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
            print(f"[SemanticCache] 🗑️  Invalidated {deleted} cache entries for document_id={document_id!r}")

    except Exception as e:
        print(f"[SemanticCache] ⚠️  Invalidation error: {e}")

    return deleted
