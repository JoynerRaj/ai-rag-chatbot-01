"""
views.py

HTTP view functions for the chat application.

Responsibilities:
  - Serving the chat, upload, documents, and cache pages
  - Handling file uploads (including large files via chunked upload)
  - Tracking upload progress while background embedding runs
  - Document CRUD (edit, delete)
  - Chat session management (create, delete)
  - Cache inspection and clearing
  - A diagnostic endpoint for debugging the embedding pipeline
"""

import os
import json
import shutil
import tempfile
import threading

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils.timezone import localtime

from .models import Document, ChatHistory, ChatSession
from chat.services import embedding_service
from chat.services.audio_service import transcribe_audio
from chat.services.document_service import DocumentExtractionService
from .semantic_cache import invalidate_by_document, get_user_cache_entries, clear_user_cache
from .redis_client import redis_client


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(request):
    return HttpResponse("OK")


# ---------------------------------------------------------------------------
# Diagnostic endpoint
# ---------------------------------------------------------------------------

def debug_embed_test(request):
    """
    Visit /debug/embed/ to test the full embedding pipeline step by step.
    Each step is independent so you can see exactly which one fails.
    """
    import traceback
    results = {}

    # Make sure all required environment variables are present
    results["env_vars"] = {
        "GEMINI_API_KEY":    bool(os.environ.get("GEMINI_API_KEY")),
        "PINECONE_API_KEY":  bool(os.environ.get("PINECONE_API_KEY")),
        "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME", "(not set — default: rag-index)"),
        "PINECONE_CLOUD":    os.environ.get("PINECONE_CLOUD",    "(not set — default: aws)"),
        "PINECONE_REGION":   os.environ.get("PINECONE_REGION",   "(not set — default: us-east-1)"),
    }

    # Test that the key libraries can be imported
    for lib_name, import_str in [
        ("import_fitz",     "import fitz"),
        ("import_docx",     "import docx"),
        ("import_pinecone", "from pinecone import Pinecone"),
        ("import_genai",    "from google import genai"),
    ]:
        try:
            exec(import_str)
            results[lib_name] = "OK"
        except Exception as e:
            results[lib_name] = f"FAILED: {e}"

    # Try a live embedding call with a short test string
    try:
        from chat.services.embedding_service import _embed_text
        vec = _embed_text("hello world test")
        results["google_embed"] = f"OK — got {len(vec)} dimensions"
    except Exception as e:
        results["google_embed"] = f"FAILED: {e}\n{traceback.format_exc()}"

    # Try connecting to Pinecone and reading index statistics
    try:
        from chat.services.embedding_service import _get_pinecone_index
        idx   = _get_pinecone_index()
        stats = idx.describe_index_stats()
        results["pinecone_connect"] = f"OK — {stats.total_vector_count} vectors in index"
    except Exception as e:
        results["pinecone_connect"] = f"FAILED: {e}\n{traceback.format_exc()}"

    # Show the most recent failed document so we can read its error trace
    try:
        failed_doc = Document.objects.filter(embedding_status="failed").order_by("-uploaded_at").first()
        if failed_doc:
            results["last_failed_document"] = {
                "id":          failed_doc.id,
                "title":       failed_doc.title,
                "error_trace": failed_doc.content,
                "time":        (
                    failed_doc.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")
                    if failed_doc.uploaded_at else "unknown"
                ),
            }
        else:
            results["last_failed_document"] = "No failed documents found"
    except Exception as e:
        results["last_failed_document"] = f"FAILED TO READ DB: {e}"

    return JsonResponse(results, json_dumps_params={"indent": 2})


# ---------------------------------------------------------------------------
# Background embedding — shared by upload_page and upload_chunk
# ---------------------------------------------------------------------------

def _embed_in_background(doc_id: int, filepath: str, filename: str):
    """
    Runs in a daemon thread after a file lands on disk.

    For audio files: transcribe first with Gemini, then embed the transcript.
    For regular documents: embed the raw file bytes directly.

    Either way, the Document record is updated so the progress page knows
    whether embedding succeeded or failed.
    """
    try:
        doc = Document.objects.get(id=doc_id)

        if filename.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
            transcript = transcribe_audio(filepath, filename)

            if transcript and transcript.strip():
                doc.content   = transcript
                pinecone_id   = embedding_service.embed_and_store(
                    transcript.encode("utf-8"),
                    filename + ".txt",
                )
                doc.pinecone_id      = pinecone_id
                doc.embedding_status = Document.EMBEDDING_DONE
                print(f"[embed] audio doc #{doc_id} transcribed and stored (id={pinecone_id!r})")
            else:
                doc.embedding_status = Document.EMBEDDING_FAILED
                doc.content          = "EMBEDDING FAILED: audio transcription returned no text"
                print(f"[embed] audio doc #{doc_id} — transcription produced no text")

        else:
            with open(filepath, "rb") as f:
                file_bytes = f.read()

            pinecone_id          = embedding_service.embed_and_store(file_bytes, filename)
            doc.pinecone_id      = pinecone_id
            doc.embedding_status = Document.EMBEDDING_DONE
            print(f"[embed] doc #{doc_id} stored in Pinecone (id={pinecone_id!r})")

        doc.save()

    except Exception:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[embed] background thread crashed for doc #{doc_id}:\n{error_trace}")
        try:
            doc = Document.objects.get(id=doc_id)
            doc.embedding_status = Document.EMBEDDING_FAILED
            doc.content          = f"EMBEDDING FAILED:\n\n{error_trace}"
            doc.save()
        except Exception:
            pass

    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

@login_required
def chat_page(request):
    documents = Document.objects.filter(user=request.user)
    chats     = ChatSession.objects.filter(user=request.user).order_by("-created_at")
    chat_id   = request.GET.get("chat_id")

    chat_data = []
    for chat in chats:
        last = ChatHistory.objects.filter(session=chat).order_by("-created_at").first()
        if last:
            last_message = last.question[:40]
            time_str     = localtime(last.created_at).strftime("%H:%M")
        else:
            last_message = "No messages yet"
            time_str     = ""

        chat_data.append({
            "id":           chat.id,
            "title":        chat.title,
            "last_message": last_message,
            "time":         time_str,
        })

    return render(request, "chat.html", {
        "documents":       documents,
        "chats":           chat_data,
        "current_chat_id": str(chat_id) if chat_id else "",
    })


# ---------------------------------------------------------------------------
# File upload (standard — for files up to ~60 MB)
# ---------------------------------------------------------------------------

@login_required
def upload_page(request):
    if request.method != "POST":
        return render(request, "upload.html")

    title   = request.POST.get("title", "").strip() or "Untitled"
    file    = request.FILES.get("file")
    is_ajax = request.headers.get("X-Upload-Ajax") == "true"

    if not file:
        if is_ajax:
            return JsonResponse({"error": "No file selected"}, status=400)
        return render(request, "upload.html", {"error": "No file selected"})

    file_name = file.name.lower()
    is_audio  = file_name.endswith((".mp3", ".wav", ".ogg", ".m4a"))

    # Write the incoming file straight to disk so we never hold it all in RAM
    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file_name)
    with os.fdopen(temp_fd, "wb") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    if is_audio:
        content = "Audio file — transcript pending."
    else:
        with open(temp_path, "rb") as f:
            content = DocumentExtractionService.extract_text_from_bytes(f.read(), file_name) or ""

    if not content.strip():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        msg = "Could not extract any text from this file. Please upload a valid PDF, DOCX, or TXT."
        if is_ajax:
            return JsonResponse({"error": msg}, status=400)
        return render(request, "upload.html", {"error": msg})

    # Save the record immediately with PENDING status so the progress page can render
    doc = Document.objects.create(
        user=request.user,
        title=title,
        content=content,
        pinecone_id=None,
        embedding_status=Document.EMBEDDING_PENDING,
    )
    print(f"[upload] doc #{doc.id} created — starting background embedding...")

    threading.Thread(
        target=_embed_in_background,
        args=(doc.id, temp_path, file.name),
        daemon=True,
    ).start()

    if is_ajax:
        return JsonResponse({"doc_id": doc.id})

    return redirect("upload_progress", doc_id=doc.id)


# ---------------------------------------------------------------------------
# Upload progress polling
# ---------------------------------------------------------------------------

@login_required
def upload_progress(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    return render(request, "upload_progress.html", {"doc": doc})


@login_required
def upload_status(request, doc_id):
    """Called by the progress page every few seconds to check embedding status."""
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    return JsonResponse({
        "status":      doc.embedding_status,
        "pinecone_id": doc.pinecone_id or "",
    })


# ---------------------------------------------------------------------------
# Chunked upload (for files larger than ~60 MB)
# ---------------------------------------------------------------------------

@login_required
def upload_chunk(request):
    """
    Receives one slice of a large file at a time.

    The browser cuts the file into ~4 MB pieces and POSTs them here in order.
    Once all pieces have arrived they are assembled into a single file on disk,
    a Document record is created, and background embedding starts — just like
    the regular upload view.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    upload_id     = request.POST.get("upload_id", "").strip()
    chunk_index   = int(request.POST.get("chunk_index", 0))
    total_chunks  = int(request.POST.get("total_chunks", 1))
    original_name = request.POST.get("filename", "upload.bin")
    title         = request.POST.get("title", "Untitled").strip() or "Untitled"
    chunk_file    = request.FILES.get("chunk")

    if not upload_id or not chunk_file:
        return JsonResponse({"error": "Missing upload_id or chunk data"}, status=400)

    # Each upload gets its own temp directory so concurrent uploads don't collide
    chunk_dir  = os.path.join(tempfile.gettempdir(), f"ragupload_{upload_id}")
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}")
    os.makedirs(chunk_dir, exist_ok=True)

    with open(chunk_path, "wb") as out:
        for data in chunk_file.chunks():
            out.write(data)

    received = len([f for f in os.listdir(chunk_dir) if f.startswith("chunk_")])
    print(f"[chunk] upload_id={upload_id}  {chunk_index + 1}/{total_chunks}  received={received}")

    if received < total_chunks:
        return JsonResponse({"status": "chunk_received", "received": received, "total": total_chunks})

    # All chunks are on disk — assemble them into one file
    file_name = original_name.lower()
    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file_name)
    with os.fdopen(temp_fd, "wb") as assembled:
        for i in range(total_chunks):
            part = os.path.join(chunk_dir, f"chunk_{i:05d}")
            with open(part, "rb") as pf:
                assembled.write(pf.read())

    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"[chunk] assembly done — {temp_path!r}  file={original_name!r}")

    is_audio = file_name.endswith((".mp3", ".wav", ".ogg", ".m4a"))

    if is_audio:
        content = "Audio file — transcript pending."
    else:
        with open(temp_path, "rb") as f:
            content = DocumentExtractionService.extract_text_from_bytes(f.read(), file_name) or ""

    if not content.strip():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JsonResponse({"error": "Could not extract any text from this file."}, status=400)

    doc = Document.objects.create(
        user=request.user,
        title=title,
        content=content,
        pinecone_id=None,
        embedding_status=Document.EMBEDDING_PENDING,
    )
    print(f"[chunk] doc #{doc.id} created — starting background embedding...")

    threading.Thread(
        target=_embed_in_background,
        args=(doc.id, temp_path, original_name),
        daemon=True,
    ).start()

    return JsonResponse({"status": "complete", "doc_id": doc.id})


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@login_required
def document_list(request):
    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {"docs": docs})


@login_required
def delete_document(request, id):
    doc = get_object_or_404(Document, id=id, user=request.user)

    # Remove from Pinecone first (skip audio docs — they use a different storage path)
    if doc.pinecone_id and not doc.pinecone_id.startswith("audio_"):
        try:
            embedding_service.delete_document(doc.pinecone_id)
        except Exception as e:
            print(f"[delete] Pinecone delete failed (continuing anyway): {e}")

    # Wipe any cache entries that were based on this document
    try:
        invalidate_by_document(str(doc.pinecone_id), user_id=request.user.id)
    except Exception as e:
        print(f"[delete] cache invalidation error: {e}")

    doc.delete()
    return redirect("documents")


@login_required
def edit_document(request, id):
    doc = get_object_or_404(Document, id=id, user=request.user)

    if request.method == "POST":
        doc.title   = request.POST.get("title")
        doc.content = request.POST.get("content")
        doc.save()
        return redirect("documents")

    return render(request, "edit.html", {"doc": doc})


# ---------------------------------------------------------------------------
# Chat session management
# ---------------------------------------------------------------------------

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
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body      = json.loads(request.body)
        first_msg = body.get("message", "New Chat")
    except Exception:
        first_msg = "New Chat"

    title = (first_msg[:50] if first_msg else "New Chat")
    chat  = ChatSession.objects.create(user=request.user, title=title)
    return JsonResponse({"chat_id": chat.id, "title": title})


@login_required
def delete_chat(request, chat_id):
    chat = get_object_or_404(ChatSession, id=chat_id, user=request.user)
    chat.delete()
    return redirect("/")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@login_required
def cache_page(request):
    cache_entries = get_user_cache_entries(user_id=request.user.id)
    return render(request, "cache.html", {
        "cache_entries":   cache_entries,
        "redis_available": redis_client is not None,
    })


@login_required
def clear_cache(request):
    clear_user_cache(user_id=request.user.id)
    return redirect("cache")


# ---------------------------------------------------------------------------
# Memory management  (MemPalace-inspired)
# ---------------------------------------------------------------------------

@login_required
def memory_page(request):
    """Display all stored memories for the current user."""
    from chat.services.memory_service import get_user_memory_entries
    entries = get_user_memory_entries(user_id=request.user.id)
    return render(request, "memories.html", {
        "memory_entries": entries,
        "total": len(entries),
    })


@login_required
def delete_memory(request, memory_id):
    """Delete a single memory entry (DB row + Pinecone vector)."""
    from chat.services.memory_service import delete_memory_entry
    delete_memory_entry(memory_id=memory_id, user_id=request.user.id)
    return redirect("memories")


@login_required
def clear_memories(request):
    """Wipe ALL memories for the current user."""
    from chat.services.memory_service import clear_user_memories
    clear_user_memories(user_id=request.user.id)
    return redirect("memories")