"""
vector_store.py

Wraps the Pinecone client so the rest of the app doesn't have to worry
about connection management or index creation.  The index object is cached
at the class level so we only connect once per process.
"""

from pinecone import Pinecone, ServerlessSpec
from core.config import settings


class VectorStoreService:
    _index = None

    @classmethod
    def get_index(cls):
        """Return the shared Pinecone index, creating it on the first call if needed."""
        if cls._index is not None:
            return cls._index

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"[pinecone] creating index '{settings.PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
        else:
            print(f"[pinecone] using existing index '{settings.PINECONE_INDEX_NAME}'")

        cls._index = pc.Index(settings.PINECONE_INDEX_NAME)
        return cls._index

    @classmethod
    def upsert_vectors(cls, vectors: list[dict]):
        cls.get_index().upsert(vectors)

    @classmethod
    def search(cls, query_embedding: list[float], top_k: int = 5, document_id: str = None):
        """Run a similarity search, optionally scoped to one document."""
        filter_ = None
        if document_id and document_id.strip():
            filter_ = {"document_id": {"$eq": document_id}}

        return cls.get_index().query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_,
        )

    @classmethod
    def delete_document(cls, document_id: str):
        """Delete all vectors that belong to a single document."""
        cls.get_index().delete(filter={"document_id": {"$eq": document_id}})
