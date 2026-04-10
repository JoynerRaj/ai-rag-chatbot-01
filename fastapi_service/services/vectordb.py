from pinecone import Pinecone, ServerlessSpec
from core.config import settings

class VectorDBService:
    _index = None

    @classmethod
    def get_index(cls):
        if cls._index is None:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION)
                )
            
            cls._index = pc.Index(settings.PINECONE_INDEX_NAME)
            
        return cls._index

    @classmethod
    def upsert_vectors(cls, vectors: list[dict]):
        index = cls.get_index()
        index.upsert(vectors)
        
    @classmethod
    def search(cls, query_embedding: list[float], top_k: int = 5, document_id: str = None):
        filter_ = None
        if document_id and document_id.strip():
            filter_ = {"document_id": {"$eq": document_id}}
            
        index = cls.get_index()
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_ if filter_ else None
        )
        
    @classmethod
    def delete_document(cls, document_id: str):
        index = cls.get_index()
        index.delete(filter={"document_id": {"$eq": document_id}})
