from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
import hashlib
import numpy as np

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


def embed(text):
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(384)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


@app.get("/")
def home():
    return {"message": "FastAPI Upload Service Running"}


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


@app.delete("/delete-all")
def delete_all():
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print("Delete error:", e)
    return {"message": "All vectors deleted from Pinecone"}