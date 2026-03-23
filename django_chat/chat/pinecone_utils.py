from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("rag-index")

model = SentenceTransformer("all-MiniLM-L6-v2")


def query_text(query, top_k=3):
    query_embedding = model.encode([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    texts = []

    for match in results["matches"]:
        texts.append(match["metadata"]["text"])

    return texts