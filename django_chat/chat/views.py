from django.shortcuts import render, redirect
from .models import Document, ChatHistory
import requests


def chat_page(request):
    documents = Document.objects.all()
    return render(request, 'chat.html', {'documents': documents})


def upload_page(request):
    if request.method == "POST":
        title = request.POST.get("title")
        file = request.FILES.get("file")

        if not file:
            return render(request, "upload.html", {"error": "No file selected"})

        try:
            content = file.read().decode("utf-8")
        except:
            return render(request, "upload.html", {"error": "File must be text (.txt)"})

        # Reset file pointer before sending to FastAPI
        file.seek(0)

        # Send to FastAPI first to get pinecone_id
        pinecone_id = ""
        try:
            files = {"file": file}
            res = requests.post(
                "https://ai-rag-chatbot-01.onrender.com/upload",
                files=files,
                timeout=60  # ADD THIS — wait up to 60s for Render to wake up
            )
            print("FastAPI status:", res.status_code)
            print("FastAPI response:", res.text)

            if res.status_code == 200:
                pinecone_id = res.json().get("document_id", "")
                print("Got pinecone_id:", pinecone_id)

        except Exception as e:
            print("FastAPI upload failed:", e)
            # Save to Django even if FastAPI fails
            pinecone_id = ""

        # Save in Django DB with pinecone_id
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
        res = requests.delete("https://ai-rag-chatbot-01.onrender.com/delete-all")
        print("FastAPI delete response:", res.status_code, res.text)
    except Exception as e:
        print("Delete API failed:", e)

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