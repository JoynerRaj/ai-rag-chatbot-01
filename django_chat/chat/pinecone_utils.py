import os
from dotenv import load_dotenv

load_dotenv()

model = None
index = None


def get_model():
    global model
    if model is None:
        print(">> [Model] Importing SentenceTransformer...")
        from sentence_transformers import SentenceTransformer
        print(">> [Model] Downloading/Loading 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(">> [Model] Successfully loaded 'all-MiniLM-L6-v2'.")
    return model


def get_index():
    global index
    if index is None:
        print(">> [Pinecone] Initializing Pinecone Client...")
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print(">> [Pinecone] Fetching 'rag-index'...")
        index = pc.Index("rag-index")
        print(">> [Pinecone] Successfully connected to 'rag-index'.")
    return index


def embed(text):
    model_instance = get_model()
    return model_instance.encode(text).tolist()


def query_pinecone(query, document_id=None, top_k=5):
    index = get_index()

    filter_ = None
    if document_id and str(document_id).strip():
        filter_ = {"document_id": {"$eq": document_id}}

    query_embedding = embed(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_
    )

    texts = []

    for match in results["matches"]:
        texts.append({
            "text": match["metadata"].get("text", ""),
            "file_name": match["metadata"].get("file_name", "Unknown")
        })

    return texts


def delete_document_vectors(document_id):
    index = get_index()

    if document_id:
        index.delete(
            delete_all=False,
            filter={"document_id": {"$eq": document_id}}
        )