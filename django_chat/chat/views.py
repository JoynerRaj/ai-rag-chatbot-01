from django.shortcuts import render, redirect
from .models import Document, ChatHistory, ChatSession
import requests
from .pinecone_utils import delete_document_vectors


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

        import os
        fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
        try:
            files = {"file": file}
            res = requests.post(
                fastapi_url,
                files=files,
                timeout=60
            )

            print("FastAPI status:", res.status_code)
            print("FastAPI response:", res.text)

            if res.status_code == 200:
                pinecone_id = res.json().get("document_id", "")

                if file.name.endswith(".txt"):
                    file.seek(0)
                    content = file.read().decode("utf-8")

        except Exception as e:
            print("FastAPI upload failed:", e)

        doc = Document.objects.create(
            title=title,
            content=content,
            pinecone_id=pinecone_id
        )

        print("Saved to Django DB, id:", doc.id, "pinecone_id:", doc.pinecone_id)

        return redirect("documents")

    return render(request, "upload.html")


def document_list(request):
    docs = Document.objects.all()
    return render(request, "documents.html", {"docs": docs})


def delete_document(request, id):
    doc = Document.objects.get(id=id)

    try:
        if doc.pinecone_id:
            delete_document_vectors(doc.pinecone_id)
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

def delete_chat(request, chat_id):
    from .models import ChatSession

    chat = ChatSession.objects.get(id=chat_id)
    chat.delete()

    return redirect("/")