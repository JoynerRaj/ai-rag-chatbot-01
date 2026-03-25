from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import uuid
import random

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


class QueryRequest(BaseModel):
    query: str


def fake_embedding():
    return [random.random() for _ in range(384)]


@app.get("/")
def home():
    return {"message": "FastAPI Service Running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    chunks = text.split("\n\n")

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": fake_embedding(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)

    return {"message": "Stored in Pinecone successfully"}


@app.post("/query")
def query(data: dict):
    question = data.get("question")
    document_id = data.get("document_id")

    if document_id:
        filter = {"document_id": document_id}
    else:
        filter = None

    results = index.query(
        vector=embed(question),
        top_k=3,
        include_metadata=True,
        filter=filter
    )

    context = ""
    sources = []

    for match in results["matches"]:
        text = match["metadata"]["text"]
        context += text + "\n"
        sources.append(text)

    answer = get_gemini_response(question, context)

    return {
        "answer": answer,
        "sources": sources[:2]
    }

@app.delete("/delete-all")
def delete_all():
    index.delete(delete_all=True)
    return {"message": "All vectors deleted from Pinecone"}