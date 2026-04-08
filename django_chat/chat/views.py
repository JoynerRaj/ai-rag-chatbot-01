from django.shortcuts import render, redirect
from .models import Document, ChatHistory, ChatSession
import requests

from django.utils.timezone import localtime

def chat_page(request):
    documents = Document.objects.all()
    chats = ChatSession.objects.all().order_by("-created_at")

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
                # Use FastAPI text for PDFs (FastAPI has pdfplumber; Django doesn't)
                if not content and fastapi_text:
                    content = fastapi_text

        except Exception as e:
            print("FastAPI upload failed:", e)

        doc = Document.objects.create(
            title=title,
            content=content,
            pinecone_id=pinecone_id
        )
        print("Saved to Django DB, id:", doc.id, "content length:", len(content))
        return redirect("documents")

    return render(request, "upload.html")


def document_list(request):
    docs = Document.objects.all()
    return render(request, "documents.html", {"docs": docs})


def delete_document(request, id):
    doc = Document.objects.get(id=id)

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

    doc.delete()
    return redirect("documents")


def edit_document(request, id):
    doc = Document.objects.get(id=id)

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

def create_chat(request):
    from .models import ChatSession
    chat = ChatSession.objects.create(title="New Chat")
    return redirect(f"/?chat_id={chat.id}")

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
        chat = ChatSession.objects.create(title=title)
        return JsonResponse({"chat_id": chat.id, "title": title})
    return JsonResponse({"error": "POST required"}, status=405)

def delete_chat(request, chat_id):
    from .models import ChatSession

    chat = ChatSession.objects.get(id=chat_id)
    chat.delete()

    return redirect("/")

def cache_page(request):
    from .redis_client import redis_client
    
    cache_entries = []
    if redis_client is not None:
        try:
            # Fetch all keys matching chat:*
            keys = redis_client.keys("chat:*")
            for k in keys:
                val = redis_client.get(k)
                ttl = redis_client.ttl(k)
                
                # Try to abbreviate long answers for the preview
                val_preview = val if val and len(val) < 200 else val[:200] + "..."
                
                cache_entries.append({
                    "key": k,
                    "value": val_preview,
                    "ttl": ttl if ttl > 0 else "Expired"
                })
        except Exception as e:
            print("Redis read error on cache page:", e)
            
    return render(request, "cache.html", {
        "cache_entries": cache_entries,
        "redis_available": redis_client is not None
    })

def clear_cache(request):
    from .redis_client import redis_client
    if redis_client is not None:
        try:
            keys = redis_client.keys("chat:*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            print("Redis delete error:", e)
    return redirect("cache")