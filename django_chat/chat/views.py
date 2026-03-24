from django.shortcuts import render, redirect
from .models import Document
import requests


def chat_page(request):
    return render(request, "chat.html")


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

        # Save in Django DB
        Document.objects.create(title=title, content=content)

        # Reset file pointer (IMPORTANT)
        file.seek(0)

        # Send to FastAPI
        try:
            files = {"file": file}
            res = requests.post(
                "https://ai-rag-chatbot-01.onrender.com/upload",
                files=files
            )

            print("FastAPI status:", res.status_code)
            print("FastAPI response:", res.text)

        except Exception as e:
            print("FastAPI upload failed:", e)

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