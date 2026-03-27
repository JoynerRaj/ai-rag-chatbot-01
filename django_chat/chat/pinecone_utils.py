import os
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "rag-index"
index = pc.Index(INDEX_NAME)

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


def embed(text):
    return model.encode(text).tolist()


def query_pinecone(query, document_id=None, top_k=5):
    filter_ = None
    if document_id and document_id.strip():
        filter_ = {"document_id": {"$eq": document_id}}

    query_embedding = embed(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_
    )

    texts = []

    SIMILARITY_THRESHOLD = 0.0

    for match in results["matches"]:
        score = match.get("score", 0)
        print("DEBUG:", score, match["metadata"].get("text"))

        if score < SIMILARITY_THRESHOLD:
            continue

        texts.append({
            "text": match["metadata"].get("text", ""),
            "file_name": match["metadata"].get("file_name", "Unknown")
        })

    return texts


def delete_document_vectors(document_id):
    if document_id:
        index.delete(
            delete_all=False,
            filter={"document_id": {"$eq": document_id}}
        )