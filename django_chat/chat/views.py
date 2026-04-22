from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils.timezone import localtime
import json
import io
import threading

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

        # Detect whether this is an AJAX upload from our JS fetch() call
        is_ajax = request.headers.get("X-Upload-Ajax") == "true"

        if not file:
            if is_ajax:
                return JsonResponse({"error": "No file selected"}, status=400)
            return render(request, "upload.html", {"error": "No file selected"})

        import os
        import tempfile

        file_name = file.name.lower()
        is_audio = file_name.endswith((".mp3", ".wav", ".ogg", ".m4a"))

        # Stream the incoming file straight to a temp file on disk.
        # We never hold the whole thing in RAM, so there is no memory spike
        # regardless of how large the file is.
        temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file_name)
        with os.fdopen(temp_fd, 'wb') as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        if is_audio:
            content = "Audio file events mapped."
        else:
            with open(temp_path, 'rb') as f:
                content = DocumentExtractionService.extract_text_from_bytes(f.read(), file_name) or ""

        if not content.strip():
            if os.path.exists(temp_path):
                os.remove(temp_path)
            msg = "Could not read any text from this file. Make sure it is a valid PDF, DOCX, or TXT."
            if is_ajax:
                return JsonResponse({"error": msg}, status=400)
            return render(request, "upload.html", {"error": msg})

        # Create the DB record immediately with PENDING status.
        # The user is sent to the progress page right away; embedding happens in a thread.
        doc = Document.objects.create(
            user=request.user,
            title=title,
            content=content,
            pinecone_id=None,
            embedding_status=Document.EMBEDDING_PENDING,
        )
        print(f"[upload] Doc #{doc.id} saved, starting background embedding...")

        def embed_in_background(doc_id, filepath, original_filename):
            try:
                doc_obj = Document.objects.get(id=doc_id)

                with open(filepath, 'rb') as file_like:
                    if original_filename.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
                        # upload_audio now returns the Gemini transcript (or "" on failure)
                        transcript = FastAPIClient.upload_audio(
                            file_like, original_filename, filepath=filepath
                        )
                        doc_obj.pinecone_id = f"audio_{doc_id}"

                        if transcript and transcript.strip():
                            # Store the real transcript so questions can be answered from it
                            doc_obj.content = transcript
                            doc_obj.embedding_status = Document.EMBEDDING_DONE
                            print(f"[upload] Audio #{doc_id} transcribed: {len(transcript)} chars")
                        else:
                            # Transcription failed but mark done so the user can still try
                            # asking questions (ai_agent will fall back to general knowledge)
                            doc_obj.embedding_status = Document.EMBEDDING_DONE
                            print(f"[upload] Audio #{doc_id} processed (no transcript returned)")

                        doc_obj.save()
                    else:
                        pinecone_id, fastapi_text = FastAPIClient.upload_document(
                            file_like, filename=original_filename, filepath=filepath
                        )
                        if pinecone_id:
                            doc_obj.pinecone_id = pinecone_id
                            doc_obj.embedding_status = Document.EMBEDDING_DONE
                            if fastapi_text and fastapi_text.strip():
                                doc_obj.content = fastapi_text
                            print(f"[upload] Doc #{doc_id} embedded - pinecone_id={pinecone_id!r}")
                        else:
                            doc_obj.embedding_status = Document.EMBEDDING_FAILED
                            print(f"[upload] Doc #{doc_id} embedding failed after retries")
                        doc_obj.save()
            except Exception as e:
                print(f"[upload] background embed crashed for doc #{doc_id}: {e}")
            finally:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass


        t = threading.Thread(
            target=embed_in_background,
            args=(doc.id, temp_path, file.name),
            daemon=True
        )
        t.start()

        # For AJAX uploads: return JSON so the browser can navigate itself.
        # This avoids the server-side redirect which requires the full 60MB
        # upload to finish before Django can respond.
        if is_ajax:
            return JsonResponse({"doc_id": doc.id})

        return redirect("upload_progress", doc_id=doc.id)

    return render(request, "upload.html")


@login_required
def upload_progress(request, doc_id):
    # show the user a loading screen while embedding runs in the background
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    return render(request, "upload_progress.html", {"doc": doc})


@login_required
def upload_status(request, doc_id):
    # the progress page polls this every few seconds to check if embedding is done
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    return JsonResponse({
        "status": doc.embedding_status,
        "pinecone_id": doc.pinecone_id or "",
    })


@login_required
def upload_chunk(request):
    """
    Receives one chunk of a large file upload at a time.
    The browser slices the file into ~4 MB pieces and POSTs them here
    sequentially, so no single request is large enough to hit Render's
    30-second response timeout.

    Once all chunks have arrived this view assembles them into a single
    temp file on disk, creates the Document record, and starts the
    background embedding thread — identical to what upload_page does.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    import os
    import shutil
    import tempfile

    upload_id    = request.POST.get("upload_id", "").strip()
    chunk_index  = int(request.POST.get("chunk_index", 0))
    total_chunks = int(request.POST.get("total_chunks", 1))
    original_name = request.POST.get("filename", "upload.bin")
    title        = request.POST.get("title", "Untitled").strip() or "Untitled"
    chunk_file   = request.FILES.get("chunk")

    if not upload_id or not chunk_file:
        return JsonResponse({"error": "Missing upload_id or chunk data"}, status=400)

    # Each upload gets its own temp directory keyed by the upload_id
    chunk_dir  = os.path.join(tempfile.gettempdir(), f"ragupload_{upload_id}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Write this chunk to disk
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}")
    with open(chunk_path, "wb") as out:
        for data in chunk_file.chunks():
            out.write(data)

    # Check how many chunks we have so far
    received = len([f for f in os.listdir(chunk_dir) if f.startswith("chunk_")])
    print(f"[chunk] upload_id={upload_id}  chunk {chunk_index + 1}/{total_chunks}  received={received}")

    if received < total_chunks:
        # Not all chunks have arrived yet — tell the browser to keep going
        return JsonResponse({"status": "chunk_received", "received": received, "total": total_chunks})

    # All chunks are on disk — assemble them into a single file
    file_name = original_name.lower()
    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file_name)
    with os.fdopen(temp_fd, "wb") as assembled:
        for i in range(total_chunks):
            part = os.path.join(chunk_dir, f"chunk_{i:05d}")
            with open(part, "rb") as pf:
                assembled.write(pf.read())

    # The individual chunks are no longer needed
    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"[chunk] assembly complete: {temp_path}  file={original_name!r}")

    is_audio = file_name.endswith((".mp3", ".wav", ".ogg", ".m4a"))

    if is_audio:
        content = "Audio file events mapped."
    else:
        with open(temp_path, "rb") as f:
            content = DocumentExtractionService.extract_text_from_bytes(f.read(), file_name) or ""

    if not content.strip():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JsonResponse({"error": "Could not read any text from this file."}, status=400)

    doc = Document.objects.create(
        user=request.user,
        title=title,
        content=content,
        pinecone_id=None,
        embedding_status=Document.EMBEDDING_PENDING,
    )
    print(f"[chunk] Doc #{doc.id} created, starting background embedding...")

    def embed_in_background(doc_id, filepath, orig_filename):
        try:
            doc_obj = Document.objects.get(id=doc_id)

            with open(filepath, "rb") as file_like:
                if orig_filename.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
                    success = FastAPIClient.upload_audio(file_like, orig_filename, filepath=filepath)
                    if success:
                        doc_obj.pinecone_id = f"audio_{doc_id}"
                        doc_obj.embedding_status = Document.EMBEDDING_DONE
                        print(f"[chunk] Audio #{doc_id} processed successfully.")
                    else:
                        doc_obj.embedding_status = Document.EMBEDDING_FAILED
                        print(f"[chunk] Audio #{doc_id} processing failed.")
                    doc_obj.save()
                else:
                    pinecone_id, fastapi_text = FastAPIClient.upload_document(
                        file_like, filename=orig_filename, filepath=filepath
                    )
                    if pinecone_id:
                        doc_obj.pinecone_id = pinecone_id
                        doc_obj.embedding_status = Document.EMBEDDING_DONE
                        if fastapi_text and fastapi_text.strip():
                            doc_obj.content = fastapi_text
                        print(f"[chunk] Doc #{doc_id} embedded - pinecone_id={pinecone_id!r}")
                    else:
                        doc_obj.embedding_status = Document.EMBEDDING_FAILED
                        print(f"[chunk] Doc #{doc_id} embedding failed after retries")
                    doc_obj.save()
        except Exception as e:
            print(f"[chunk] background embed crashed for doc #{doc_id}: {e}")
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

    threading.Thread(target=embed_in_background, args=(doc.id, temp_path, original_name), daemon=True).start()

    return JsonResponse({"status": "complete", "doc_id": doc.id})


@login_required
def document_list(request):
    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {"docs": docs})


@login_required
def delete_document(request, id):
    doc = get_object_or_404(Document, id=id, user=request.user)

    # try to remove from Pinecone - if FastAPI is down, still delete from DB
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