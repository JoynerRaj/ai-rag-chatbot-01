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

        # Read file bytes once — we need them both for local extraction and FastAPI
        file_bytes = file.read()
        file_name  = file.name.lower()

        # Extract text locally (fast — no network call)
        content = DocumentExtractionService.extract_text_from_bytes(file_bytes, file_name)
        if not content:
            content = ""

        # Save document to DB immediately so the user can continue right away
        doc = Document.objects.create(
            user=request.user,
            title=title,
            content=content,
            pinecone_id=None,
            embedding_status=Document.EMBEDDING_PENDING,
        )
        print(f"[upload] Doc #{doc.id} saved instantly, starting background embedding…")

        # --- background embedding thread ---
        def _embed_in_background(doc_id: int, raw_bytes: bytes, fname: str):
            """Runs in a daemon thread — embeds the document in Pinecone after save."""
            import io
            from django.db import connection
            try:
                from chat.models import Document as Doc
                from chat.services.fastapi_client import FastAPIClient

                file_like = io.BytesIO(raw_bytes)
                file_like.name = fname

                pinecone_id, fastapi_text = FastAPIClient.upload_document(file_like)

                doc_obj = Doc.objects.get(id=doc_id)
                if pinecone_id:
                    if fastapi_text and fastapi_text.strip():
                        doc_obj.content = fastapi_text
                    doc_obj.pinecone_id      = pinecone_id
                    doc_obj.embedding_status = Doc.EMBEDDING_DONE
                    print(f"[embed] Doc #{doc_id} embedded OK — pinecone_id={pinecone_id}")
                else:
                    doc_obj.embedding_status = Doc.EMBEDDING_FAILED
                    print(f"[embed] Doc #{doc_id} embedding FAILED (FastAPI returned empty)")
                doc_obj.save()
            except Exception as exc:
                import traceback, logging
                logging.error(f"[embed] Doc #{doc_id} background error: {exc}")
                traceback.print_exc()
                try:
                    from chat.models import Document as Doc
                    Doc.objects.filter(id=doc_id).update(embedding_status=Doc.EMBEDDING_FAILED)
                except Exception:
                    pass
            finally:
                connection.close()  # release DB connection from thread

        import threading
        t = threading.Thread(
            target=_embed_in_background,
            args=(doc.id, file_bytes, file.name),
            daemon=True,
        )
        t.start()

        # Redirect immediately — embedding happens in background
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