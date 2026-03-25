from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import uuid

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

# Real embedding model (384 dimensions — matches your Pinecone index)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    return embed_model.encode(text).tolist()


@app.get("/")
def home():
    return {"message": "FastAPI Service Running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    document_id = str(uuid.uuid4())

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embed(chunk),
            "metadata": {
                "text": chunk,
                "document_id": document_id,
                "file_name": file.filename
            }
        })

    index.upsert(vectors)

    return {
        "message": "Stored in Pinecone successfully",
        "document_id": document_id
    }


@app.post("/query")
def query_api(data: dict):
    query = data.get("query")
    document_id = data.get("document_id")

    if not query:
        return {"results": []}

    # Apply filter only when a specific document is selected
    filter = None
    if document_id and document_id.strip():
        filter = {"document_id": {"$eq": document_id}}

    results = index.query(
        vector=embed(query),
        top_k=3,
        include_metadata=True,
        filter=filter
    )

    texts = []
    for match in results["matches"]:
        texts.append({
            "text": match["metadata"].get("text", ""),
            "file_name": match["metadata"].get("file_name", "Unknown")
        })

    return {"results": texts}


@app.delete("/delete-all")
def delete_all():
    index.delete(delete_all=True)
    return {"message": "All vectors deleted from Pinecone"}