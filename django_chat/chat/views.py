from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils.timezone import localtime
import json

from .models import Document, ChatHistory, ChatSession
from chat.services.fastapi_client import FastAPIClient
from chat.services.document_service import DocumentExtractionService
from .semantic_cache import invalidate_by_document
from .redis_client import redis_client

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

        # Step 1: Extract content on Django side
        content = DocumentExtractionService.extract_text(file, file.name.lower())

        # Step 2: Send to FastAPI for Pinecone embedding
        pinecone_id, fastapi_text = FastAPIClient.upload_document(file)
        
        if fastapi_text and fastapi_text.strip():
            content = fastapi_text
        elif not content:
            content = fastapi_text  # last-resort fallback

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
    doc = get_object_or_404(Document, id=id, user=request.user)

    if doc.pinecone_id:
        FastAPIClient.delete_document(doc.pinecone_id)

    try:
        invalidate_by_document(str(doc.pinecone_id))
    except Exception as e:
        print("Cache invalidation error:", e)

    doc.delete()
    return redirect("documents")


@login_required
def edit_document(request, id):
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
    chat = ChatSession.objects.create(user=request.user, title="New Chat")
    return redirect(f"/?chat_id={chat.id}")


@login_required
def create_chat_ajax(request):
    if request.method == "POST":
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
    chat = get_object_or_404(ChatSession, id=chat_id, user=request.user)
    chat.delete()
    return redirect("/")


@login_required
def cache_page(request):
    cache_entries = []
    if redis_client is not None:
        try:
            user_prefix = f"{request.user.id}:" if request.user.is_authenticated else "public:"
            keys = redis_client.keys(f"chat:emb:{user_prefix}*")
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
                    answer_preview = answer if len(answer) < 300 else answer[:300] + "..."
                    cache_entries.append({
                        "key":     k,
                        "query":   query,
                        "value":   answer_preview,
                        "doc_id":  doc_id,
                        "ttl":     ttl if ttl > 0 else "Expired",
                    })
                except Exception:
                    pass  
        except Exception as e:
            print("Redis read error on cache page:", e)

    return render(request, "cache.html", {
        "cache_entries": cache_entries,
        "redis_available": redis_client is not None
    })


@login_required
def clear_cache(request):
    if redis_client is not None:
        try:
            user_prefix = f"{request.user.id}:" if request.user.is_authenticated else "public:"
            keys = redis_client.keys(f"chat:emb:{user_prefix}*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            print("Redis delete error:", e)
    return redirect("cache")