from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Document, ChatHistory, ChatSession
import requests
from django.http import HttpResponse

from django.utils.timezone import localtime

def health_check(request):
    return HttpResponse("OK")

@login_required
def chat_page(request):
    documents = Document.objects.filter(user=request.user)
    chats = ChatSession.objects.filter(user=request.user).order_by("-created_at")

    chat_id = request.GET.get("chat_id")

    chat_data = []

    for chat in chats:
        last = ChatHistory.objects.filter(session=chat).order_by("-created_at").first()

        if last:
            last_message = last.question[:40]
            time = localtime(last.created_at).strftime("%H:%M")
        else:
            last_message = "No messages yet"
            time = ""

        chat_data.append({
            "id": chat.id,
            "title": chat.title,
            "last_message": last_message,
            "time": time
        })

    return render(request, 'chat.html', {
        'documents': documents,
        'chats': chat_data,
        'current_chat_id': str(chat_id) if chat_id else ""
    })


@login_required
def upload_page(request):
    if request.method == "POST":
        title = request.POST.get("title")
        file = request.FILES.get("file")

        if not file:
            return render(request, "upload.html", {"error": "No file selected"})

        pinecone_id = ""
        content = ""

        # ── Step 1: Extract content on Django side (reliable, no extra deps) ──
        filename = file.name.lower()
        try:
            if filename.endswith(".txt"):
                file.seek(0)
                content = file.read().decode("utf-8", errors="ignore")
                file.seek(0)

            elif filename.endswith(".pdf"):
                import io
                from pypdf import PdfReader
                file.seek(0)
                reader = PdfReader(io.BytesIO(file.read()))
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                content = "\n".join(pages_text)
                file.seek(0)

            elif filename.endswith(".docx"):
                import zipfile, io, re
                file.seek(0)
                with zipfile.ZipFile(io.BytesIO(file.read())) as z:
                    if "word/document.xml" in z.namelist():
                        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                        content = re.sub(r"<[^>]+>", " ", xml)
                        content = re.sub(r"\s+", " ", content).strip()
                file.seek(0)

        except Exception as e:
            print("Django-side content extraction failed:", e)

        # ── Step 2: Send to FastAPI for Pinecone embedding ────────────────────
        import os
        fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
        try:
            file.seek(0)
            res = requests.post(fastapi_url, files={"file": file}, timeout=60)
            print("FastAPI status:", res.status_code)
            print("FastAPI response:", res.text[:300])

            if res.status_code == 200:
                res_json = res.json()
                pinecone_id = res_json.get("document_id", "")
                fastapi_text = res_json.get("text", "")
                # FastAPI uses pdfplumber which is far superior to pypdf for PDFs.
                # Always prefer FastAPI text if it returned something meaningful.
                if fastapi_text and fastapi_text.strip():
                    content = fastapi_text
                elif not content:
                    content = fastapi_text  # last-resort fallback

        except Exception as e:
            print("FastAPI upload failed:", e)

        doc = Document.objects.create(
            user=request.user,
            title=title,
            content=content,
            pinecone_id=pinecone_id
        )
        print("Saved to Django DB, id:", doc.id, "content length:", len(content))
        return redirect("documents")

    return render(request, "upload.html")


@login_required
def document_list(request):
    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {"docs": docs})


@login_required
def delete_document(request, id):
    from django.shortcuts import get_object_or_404
    doc = get_object_or_404(Document, id=id, user=request.user)

    try:
        if doc.pinecone_id:
            import requests, os
            fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
            delete_api = fastapi_url.replace("/upload", "") + f"/delete/{doc.pinecone_id}"
            res = requests.delete(delete_api, timeout=30)
            if not res.ok:
                print("Failed calling delete API on FastAPI:", res.text)
    except Exception as e:
        print("Pinecone delete error:", e)

    # ── Invalidate all semantic cache entries tied to this document ──────────
    try:
        from .semantic_cache import invalidate_by_document
        invalidate_by_document(str(doc.pinecone_id))
    except Exception as e:
        print("Cache invalidation error:", e)
    # ── End Cache Invalidation ───────────────────────────────────────────────

    doc.delete()
    return redirect("documents")


@login_required
def edit_document(request, id):
    from django.shortcuts import get_object_or_404
    doc = get_object_or_404(Document, id=id, user=request.user)

    if request.method == "POST":
        doc.title = request.POST.get("title")
        doc.content = request.POST.get("content")
        doc.save()
        return redirect("documents")

    return render(request, "edit.html", {"doc": doc})


def clear_history(request):
    session_key = request.session.session_key
    if session_key:
        ChatHistory.objects.filter(session_key=session_key).delete()
    return redirect("chat")

@login_required
def create_chat(request):
    from .models import ChatSession
    chat = ChatSession.objects.create(user=request.user, title="New Chat")
    return redirect(f"/?chat_id={chat.id}")

@login_required
def create_chat_ajax(request):
    """AJAX endpoint — creates a chat session titled from the first message."""
    from django.http import JsonResponse
    from .models import ChatSession
    if request.method == "POST":
        import json
        try:
            body = json.loads(request.body)
            first_msg = body.get("message", "New Chat")
        except Exception:
            first_msg = "New Chat"
        title = first_msg[:50] if first_msg else "New Chat"
        chat = ChatSession.objects.create(user=request.user, title=title)
        return JsonResponse({"chat_id": chat.id, "title": title})
    return JsonResponse({"error": "POST required"}, status=405)

@login_required
def delete_chat(request, chat_id):
    from django.shortcuts import get_object_or_404
    from .models import ChatSession
    chat = get_object_or_404(ChatSession, id=chat_id, user=request.user)
    chat.delete()
    return redirect("/")

@login_required
def cache_page(request):
    import json
    from .redis_client import redis_client

    cache_entries = []
    if redis_client is not None:
        try:
            keys = redis_client.keys("chat:emb:*")
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
                    # Truncate long answers for preview
                    answer_preview = answer if len(answer) < 300 else answer[:300] + "..."
                    cache_entries.append({
                        "key":     k,
                        "query":   query,
                        "value":   answer_preview,
                        "doc_id":  doc_id,
                        "ttl":     ttl if ttl > 0 else "Expired",
                    })
                except Exception:
                    pass  # skip malformed entries
        except Exception as e:
            print("Redis read error on cache page:", e)

    return render(request, "cache.html", {
        "cache_entries": cache_entries,
        "redis_available": redis_client is not None
    })

@login_required
def clear_cache(request):
    from .redis_client import redis_client
    if redis_client is not None:
        try:
            keys = redis_client.keys("chat:emb:*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            print("Redis delete error:", e)
    return redirect("cache")