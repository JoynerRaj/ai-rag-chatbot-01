from operator import index

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
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
app  = FastAPI()

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

model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

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