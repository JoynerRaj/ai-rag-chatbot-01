from operator import index

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
import fitz
import docx
import io

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

def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "rag-index"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)

# Safely import PyTorch on Main Thread to prevent C++ Extension native crashes in Background threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from sentence_transformers import SentenceTransformer

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@app.on_event("startup")
async def startup_event():
    import threading
    def preload_model():
        print("Pre-loading HuggingFace weights in background thread...")
        try:
            get_model()
            print("PyTorch Model Initialized Successfully!")
        except Exception as e:
            print(f"Failed to preload model: {e}")
            
    thread = threading.Thread(target=preload_model, daemon=True)
    thread.start()

def embed(text):
    model_instance = get_model()
    return model_instance.encode(text).tolist()

def split_text(text, chunk_size=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


def extract_text(file: UploadFile, content: bytes):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return content.decode("utf-8")

    elif filename.endswith(".pdf"):
        text = ""
        pdf = fitz.open(stream=content, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return None


@app.get("/")
def home():
    return {"message": "FastAPI Upload Service Running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    text = extract_text(file, content)

    if not text:
        return {"error": "Unsupported file type"}

    chunks = split_text(text)

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

    index = get_index()
    index.upsert(vectors)

    return {
        "message": "Stored in Pinecone successfully",
        "document_id": document_id
    }

from pydantic import BaseModel
class SearchQuery(BaseModel):
    query: str
    document_id: str = None
    top_k: int = 5

@app.post("/search")
def search_pinecone(req: SearchQuery):
    try:
        query_embedding = embed(req.query)
        filter_ = None
        if req.document_id and req.document_id.strip():
            filter_ = {"document_id": {"$eq": req.document_id}}

        index = get_index()
        results = index.query(
            vector=query_embedding,
            top_k=req.top_k,
            include_metadata=True,
            filter=filter_ if filter_ else None
        )
        
        # Pinecone objects must be converted to native dicts for JSON serialization
        results_dict = results.to_dict()
        
        matches = []
        for match in results_dict.get("matches", []):
            matches.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "text": match.get("metadata", {}).get("text", "")
            })
        return matches
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))