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
def query_api(data: QueryRequest):
    query_embedding = fake_embedding()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    texts = []

    for match in results["matches"]:
        if "metadata" in match and "text" in match["metadata"]:
            texts.append(match["metadata"]["text"])

    return {"results": texts}

@app.delete("/delete-all")
def delete_all():
    index.delete(delete_all=True)
    return {"message": "All vectors deleted from Pinecone"}