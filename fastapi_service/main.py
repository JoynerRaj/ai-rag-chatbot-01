from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


model = None


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

@app.get("/")
def home():
    return {"message": "RAG API Running "}


def load_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")


def process_text(text):
    load_model()

    chunks = text.split("\n")
    chunks = [c for c in chunks if c.strip() != ""]

    embeddings = model.encode(chunks)
    return chunks, embeddings


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        chunks, embeddings = process_text(text)

        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": str(i),
                "values": emb.tolist(),
                "metadata": {"text": chunks[i]}
            })

        index.upsert(vectors)

        return {"message": "Stored in Pinecone successfully"}

    except Exception as e:
        return {"error": str(e)}


@app.post("/query")
async def query(q: str):
    try:
        load_model()

        
        query_embedding = model.encode([q])[0]

    
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_metadata=True
        )

    
        context = " ".join(
            [match["metadata"]["text"] for match in results["matches"]]
        )

        prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{q}
"""

    
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Answer only from given context."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = completion.choices[0].message.content

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}