from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils.timezone import localtime
import json

from .models import Document, ChatHistory, ChatSession
from chat.services.fastapi_client import FastAPIClient
from chat.services.document_service import DocumentExtractionService
from .semantic_cache import invalidate_by_document, get_user_cache_entries, clear_user_cache
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
        title = request.POST.get("title", "").strip() or "Untitled"
        file = request.FILES.get("file")

        if not file:
            return render(request, "upload.html", {"error": "No file selected"})

        # Extract text locally first (fast, no network)
        file_bytes = file.read()
        file_name  = file.name.lower()
        content = DocumentExtractionService.extract_text_from_bytes(file_bytes, file_name) or ""

        # Upload to FastAPI / Pinecone (synchronous — this is reliable on Render)
        import io
        file_like = io.BytesIO(file_bytes)
        try:
            pinecone_id, fastapi_text = FastAPIClient.upload_document(
                file_like, filename=file.name
            )
        except Exception as e:
            print(f"[upload] FastAPI call failed: {e}")
            pinecone_id, fastapi_text = "", ""

        if fastapi_text and fastapi_text.strip():
            content = fastapi_text

        embedding_status = Document.EMBEDDING_DONE if pinecone_id else Document.EMBEDDING_FAILED

        doc = Document.objects.create(
            user=request.user,
            title=title,
            content=content,
            pinecone_id=pinecone_id or None,
            embedding_status=embedding_status,
        )
        print(f"[upload] Doc #{doc.id} saved — pinecone_id={pinecone_id!r}  status={embedding_status}")

        if not pinecone_id:
            # Embedding failed (FastAPI cold start or error) — let user know and retry
            return render(request, "upload.html", {
                "error": (
                    "The AI embedding service is starting up. "
                    "Please wait 30 seconds and try uploading again."
                ),
                "prefill_title": title,
            })

        return redirect("documents")

    return render(request, "upload.html")


@login_required
def document_list(request):
    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {"docs": docs})


@login_required
def delete_document(request, id):
    doc = get_object_or_404(Document, id=id, user=request.user)

    # try to remove from Pinecone — if FastAPI is down, still delete from DB
    if doc.pinecone_id:
        try:
            FastAPIClient.delete_document(doc.pinecone_id)
        except Exception as e:
            print("Pinecone delete failed (continuing with DB delete):", e)

    try:
        invalidate_by_document(str(doc.pinecone_id), user_id=request.user.id)
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
    # only pull cache entries for the currently logged in user, not everyone
    cache_entries = get_user_cache_entries(user_id=request.user.id)

    return render(request, "cache.html", {
        "cache_entries": cache_entries,
        "redis_available": redis_client is not None
    })


@login_required
def clear_cache(request):
    # wipe only this user's cache - don't touch other users' data
    clear_user_cache(user_id=request.user.id)
    return redirect("cache")