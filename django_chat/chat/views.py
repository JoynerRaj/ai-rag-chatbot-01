from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils.timezone import localtime
import json
import io
import threading

from .models import Document, ChatHistory, ChatSession
from chat.services.fastapi_client import FastAPIClient
from chat.services import embedding_service
from chat.services.document_service import DocumentExtractionService
from .semantic_cache import invalidate_by_document, get_user_cache_entries, clear_user_cache
from .redis_client import redis_client

def health_check(request):
    return HttpResponse("OK")


def debug_embed_test(request):
    """
    Diagnostic endpoint - visit /debug/embed/ to see exactly what's failing.
    Tests each step and returns JSON with pass/fail and the exact error.
    """
    import os, traceback
    results = {}

    # step 1: check env vars
    results["env_vars"] = {
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "PINECONE_API_KEY": bool(os.environ.get("PINECONE_API_KEY")),
        "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME", "(not set - using default rag-index)"),
        "PINECONE_CLOUD": os.environ.get("PINECONE_CLOUD", "(not set - using default aws)"),
        "PINECONE_REGION": os.environ.get("PINECONE_REGION", "(not set - using default us-east-1)"),
    }

    # step 2: test imports
    try:
        import fitz
        results["import_fitz"] = "OK"
    except Exception as e:
        results["import_fitz"] = f"FAILED: {e}"

    try:
        import docx
        results["import_docx"] = "OK"
    except Exception as e:
        results["import_docx"] = f"FAILED: {e}"

    try:
        from pinecone import Pinecone
        results["import_pinecone"] = "OK"
    except Exception as e:
        results["import_pinecone"] = f"FAILED: {e}"

    try:
        from google import genai
        results["import_genai"] = "OK"
    except Exception as e:
        results["import_genai"] = f"FAILED: {e}"

    # step 3: test Google embedding
    try:
        from chat.services.embedding_service import _embed_text
        vec = _embed_text("hello world test")
        results["google_embed"] = f"OK - got {len(vec)} dimensions"
    except Exception as e:
        results["google_embed"] = f"FAILED: {e}\n{traceback.format_exc()}"

    # step 4: test Pinecone connection
    try:
        from chat.services.embedding_service import _get_pinecone_index
        idx = _get_pinecone_index()
        stats = idx.describe_index_stats()
        results["pinecone_connect"] = f"OK - {stats.total_vector_count} vectors in index"
    except Exception as e:
        results["pinecone_connect"] = f"FAILED: {e}\n{traceback.format_exc()}"

    # step 5: get last database error
    try:
        from chat.models import Document
        failed_doc = Document.objects.filter(embedding_status="failed").order_by("-uploaded_at").first()
        if failed_doc:
            results["last_failed_document"] = {
                "id": failed_doc.id,
                "title": failed_doc.title,
                "error_trace": failed_doc.content,
                "time": failed_doc.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if failed_doc.uploaded_at else "Unknown"
            }
        else:
            results["last_failed_document"] = "No failed documents found"
    except Exception as e:
        results["last_failed_document"] = f"FAILED TO READ DB: {e}"

    return JsonResponse(results, json_dumps_params={"indent": 2})


def _transcribe_audio_with_gemini(filepath, filename):
    """Upload audio to Gemini Files API and return a verbatim transcript."""
    import time
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""

    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "m4a": "audio/mp4"}
    ext = filename.lower().rsplit(".", 1)[-1]
    mime_type = mime_map.get(ext, "audio/mpeg")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    file_obj = None
    try:
        print(f"[transcribe] uploading {filename!r} to Gemini...")
        with open(filepath, "rb") as f:
            file_obj = client.files.upload(
                file=f,
                config=types.UploadFileConfig(mime_type=mime_type, display_name=filename),
            )

        waited = 0
        while hasattr(file_obj, "state") and file_obj.state.name == "PROCESSING":
            if waited >= 120:
                print("[transcribe] timed out")
                return ""
            time.sleep(3)
            waited += 3
            file_obj = client.files.get(name=file_obj.name)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_uri(file_uri=file_obj.uri, mime_type=mime_type),
                types.Part.from_text(
                    "Provide a complete, verbatim transcription of all speech in this audio. "
                    "Include every word — do not summarize."
                ),
            ],
        )

        transcript = response.text.strip() if response.text else ""
        print(f"[transcribe] done: {len(transcript)} chars")
        return transcript

    except Exception as e:
        print(f"[transcribe] error: {e}")
        return ""
    finally:
        if file_obj:
            try:
                client.files.delete(name=file_obj.name)
            except Exception:
                pass

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

                with open(filepath, "rb") as f:
                    file_bytes = f.read()

                if original_filename.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
                    transcript = _transcribe_audio_with_gemini(filepath, original_filename)
                    if transcript and transcript.strip():
                        doc_obj.content = transcript
                        # Trick the extractor into reading the transcript as plain text
                        pinecone_id = embedding_service.embed_and_store(
                            transcript.encode("utf-8"), 
                            original_filename + ".txt"
                        )
                        doc_obj.pinecone_id = pinecone_id
                        doc_obj.embedding_status = Document.EMBEDDING_DONE
                        print(f"[upload] Audio #{doc_id} transcribed and embedded locally.")
                    else:
                        doc_obj.embedding_status = Document.EMBEDDING_FAILED
                        doc_obj.content = "EMBEDDING FAILED:\n\nAudio transcription returned no text."
                        print(f"[upload] Audio #{doc_id} transcription failed.")
                    doc_obj.save()
                else:
                    # embed directly in Django - no FastAPI cold-start issues
                    pinecone_id = embedding_service.embed_and_store(file_bytes, original_filename)
                    doc_obj.pinecone_id = pinecone_id
                    doc_obj.embedding_status = Document.EMBEDDING_DONE
                    print(f"[upload] doc #{doc_id} embedded locally, pinecone_id={pinecone_id!r}")
                    doc_obj.save()

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"[upload] background embed failed for doc #{doc_id}:\n{error_trace}")
                try:
                    doc_obj = Document.objects.get(id=doc_id)
                    doc_obj.embedding_status = Document.EMBEDDING_FAILED
                    # Save the error to the document content so we can see what broke
                    doc_obj.content = f"EMBEDDING FAILED:\n\n{error_trace}"
                    doc_obj.save()
                except Exception:
                    pass
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
                file_bytes = file_like.read()
                file_like.seek(0)
                
                if orig_filename.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
                    transcript = _transcribe_audio_with_gemini(filepath, orig_filename)
                    if transcript and transcript.strip():
                        doc_obj.content = transcript
                        pinecone_id = embedding_service.embed_and_store(
                            transcript.encode("utf-8"), 
                            orig_filename + ".txt"
                        )
                        doc_obj.pinecone_id = pinecone_id
                        doc_obj.embedding_status = Document.EMBEDDING_DONE
                        print(f"[chunk] Audio #{doc_id} transcribed and embedded locally.")
                    else:
                        doc_obj.embedding_status = Document.EMBEDDING_FAILED
                        doc_obj.content = "EMBEDDING FAILED:\n\nAudio transcription returned no text."
                        print(f"[chunk] Audio #{doc_id} processing failed.")
                    doc_obj.save()
                else:
                    pinecone_id = embedding_service.embed_and_store(file_bytes, orig_filename)
                    doc_obj.pinecone_id = pinecone_id
                    doc_obj.embedding_status = Document.EMBEDDING_DONE
                    print(f"[chunk] Doc #{doc_id} embedded locally - pinecone_id={pinecone_id!r}")
                    doc_obj.save()
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[chunk] background embed crashed for doc #{doc_id}:\n{error_trace}")
            try:
                doc_obj = Document.objects.get(id=doc_id)
                doc_obj.embedding_status = Document.EMBEDDING_FAILED
                doc_obj.content = f"EMBEDDING FAILED:\n\n{error_trace}"
                doc_obj.save()
            except Exception:
                pass
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

    # try to remove from Pinecone
    if doc.pinecone_id and not doc.pinecone_id.startswith("audio_"):
        try:
            embedding_service.delete_document(doc.pinecone_id)
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