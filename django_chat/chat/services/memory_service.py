import os
import uuid
import threading
import traceback

from google import genai
from chat.services.embedding_service import _embed_text, _get_pinecone_index


MEMORY_TYPE_TAG  = "mem"
MEMORY_TOP_K     = 5
MEMORY_THRESHOLD = 0.55

_TRIVIAL_PHRASES = {
    "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you",
    "ok", "okay", "sure", "cool", "nice", "great", "good", "awesome",
}


def _is_worth_remembering(question, answer):
    q = question.strip().lower().rstrip("!?.,:;")
    if q in _TRIVIAL_PHRASES:
        return False
    if len(question.split()) < 2:
        return False
    if len(answer.strip()) < 15:
        return False
    return True


def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})


def extract_memories(question, answer):
    try:
        client = _gemini_client()
        prompt = (
            "You are a memory extractor. From the conversation below, extract "
            "1 to 3 concise, self-contained factual statements worth remembering. "
            "Each statement must be a single sentence. Focus on facts, decisions, "
            "and preferences. Skip greetings and filler.\n\n"
            "Return only the statements, one per line, no bullets or numbers.\n\n"
            f"User asked: {question}\n\n"
            f"Assistant answered: {answer[:1500]}\n\n"
            "Facts:"
        )
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text or ""
        facts = [line.strip() for line in text.strip().splitlines() if line.strip()]
        valid = [f for f in facts if 10 < len(f) < 300][:3]
        print(f"[memory] extracted {len(valid)} fact(s) via Gemini")
        return valid

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            print("[memory] Gemini quota exceeded, using fallback")
            q = question.strip()
            if len(q.split()) >= 4 and len(q) < 300:
                return [q]
            return []
        print(f"[memory] extract_memories failed: {e}")
        return []


def _store_single_memory(user_id, session_id, fact, question):
    try:
        embedding = _embed_text(fact)
        index = _get_pinecone_index()
        vid = str(uuid.uuid4())
        index.upsert(vectors=[{
            "id": vid,
            "values": embedding,
            "metadata": {
                "type": MEMORY_TYPE_TAG,
                "user_id": str(user_id),
                "session_id": str(session_id),
                "fact": fact,
                "question": question[:200],
            },
        }])
        print(f"[memory] stored user={user_id} id={vid[:8]}")
        return vid
    except Exception as e:
        print(f"[memory] store error: {e}")
        traceback.print_exc()
        return None


def store_exchange_async(user_id, session_id, question, answer):
    if not _is_worth_remembering(question, answer):
        return

    def _run():
        import django.db
        django.db.connections.close_all()
        try:
            print(f"[memory] thread started user={user_id} q={ascii(question[:40])}")
            from chat.models import MemoryEntry
            facts = extract_memories(question, answer)
            saved = 0
            for fact in facts:
                vid = _store_single_memory(user_id, session_id, fact, question)
                if vid:
                    try:
                        MemoryEntry.objects.create(
                            user_id=user_id,
                            session_id=session_id,
                            content=fact,
                            source_question=question[:500],
                            pinecone_id=vid,
                        )
                        saved += 1
                        print(f"[memory] saved fact={ascii(fact[:50])}")
                    except Exception as db_err:
                        print(f"[memory] DB create failed: {db_err}")
                        traceback.print_exc()
            print(f"[memory] done saved={saved}/{len(facts)} user={user_id}")
        except Exception as e:
            print(f"[memory] thread error: {e}")
            traceback.print_exc()
        finally:
            django.db.connections.close_all()

    threading.Thread(target=_run, daemon=True).start()


def search_memories(user_id, query, n=MEMORY_TOP_K):
    try:
        query_vec = _embed_text(query)
        index = _get_pinecone_index()
        results = index.query(
            vector=query_vec,
            top_k=n,
            include_metadata=True,
            filter={
                "type": {"$eq": MEMORY_TYPE_TAG},
                "user_id": {"$eq": str(user_id)},
            },
        )
        memories = []
        for m in results.matches:
            if m.score >= MEMORY_THRESHOLD and m.metadata:
                memories.append({
                    "fact": m.metadata.get("fact", ""),
                    "question": m.metadata.get("question", ""),
                    "session_id": m.metadata.get("session_id", ""),
                    "score": round(m.score, 3),
                })
        print(f"[memory] search user={user_id} found={len(memories)}")
        return memories
    except Exception as e:
        print(f"[memory] search error: {e}")
        return []


def build_memory_context(user_id, query):
    memories = search_memories(user_id, query)
    if not memories:
        return ""
    lines = ["[Context from past conversations with this user:]"]
    for mem in memories:
        lines.append(f"- {mem['fact']}")
    return "\n".join(lines)


def get_user_memory_entries(user_id):
    try:
        from chat.models import MemoryEntry
        return list(MemoryEntry.objects.filter(user_id=user_id).order_by("-created_at"))
    except Exception as e:
        print(f"[memory] get entries error: {e}")
        return []


def delete_memory_entry(memory_id, user_id):
    try:
        from chat.models import MemoryEntry
        entry = MemoryEntry.objects.get(id=memory_id, user_id=user_id)
        if entry.pinecone_id:
            try:
                _get_pinecone_index().delete(ids=[entry.pinecone_id])
            except Exception as e:
                print(f"[memory] Pinecone delete error: {e}")
        entry.delete()
        return True
    except Exception as e:
        print(f"[memory] delete error: {e}")
        return False


def clear_user_memories(user_id):
    try:
        _get_pinecone_index().delete(filter={
            "type": {"$eq": MEMORY_TYPE_TAG},
            "user_id": {"$eq": str(user_id)},
        })
    except Exception as e:
        print(f"[memory] Pinecone clear error: {e}")

    try:
        from chat.models import MemoryEntry
        count, _ = MemoryEntry.objects.filter(user_id=user_id).delete()
        return count
    except Exception as e:
        print(f"[memory] DB clear error: {e}")
        return 0
