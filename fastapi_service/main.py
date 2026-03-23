from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
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

model = SentenceTransformer("all-MiniLM-L6-v2")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "FastAPI Service Running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    chunks = text.split("\n\n")

    embeddings = model.encode(chunks)

    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {"text": chunks[i]}
        })

    index.upsert(vectors)

    return {"message": "Stored in Pinecone successfully"}


@app.post("/query")
def query_api(data: QueryRequest):
    query = data.query

    query_embedding = model.encode([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
    )

    texts = []

    for match in results["matches"]:
        if "metadata" in match and "text" in match["metadata"]:
            texts.append(match["metadata"]["text"])

    return {"results": texts}