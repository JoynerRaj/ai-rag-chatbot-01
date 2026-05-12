"""
routes.py

Document-related API endpoints:
  POST   /upload        — extract text from a file and store it in Pinecone
  POST   /search        — similarity search against stored vectors
  DELETE /delete/{id}   — remove a document's vectors from Pinecone
  POST   /embed         — get the embedding vector for a query string
"""

import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.document import DocumentService
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from schemas.document import SearchQuery, EmbedQuery


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Read a document, chunk it, embed each chunk, and upsert to Pinecone."""
    content = await file.read()

    text = DocumentService.extract_text(file, content)
    if not text:
        print(f"[upload] no text extracted from {file.filename!r}")
        return {"error": "Unsupported file type or empty document"}

    chunks      = DocumentService.split_text(text)
    document_id = str(uuid.uuid4())
    print(f"[upload] {file.filename!r}  chunks={len(chunks)}  doc_id={document_id}")

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id":     str(uuid.uuid4()),
            "values": EmbeddingService.embed_text(chunk),
            "metadata": {
                "text":        chunk,
                "document_id": document_id,
                "file_name":   file.filename,
            },
        })

    try:
        VectorStoreService.upsert_vectors(vectors)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[upload] Pinecone upsert failed: {e}")
        return {"error": f"Vector DB error: {str(e)}"}

    return {
        "message":     "Stored in Pinecone successfully",
        "document_id": document_id,
        "text":        text,
    }


@router.post("/search")
def search_documents(req: SearchQuery):
    """Find the most relevant document chunks for a given query."""
    try:
        print(f"[search] query={req.query!r}  doc_id={req.document_id!r}  top_k={req.top_k}")

        query_embedding = EmbeddingService.embed_text(req.query)
        results = VectorStoreService.search(
            query_embedding=query_embedding,
            top_k=req.top_k,
            document_id=req.document_id,
        )

        matches = []
        for match in results.matches:
            text_context = match.metadata.get("text", "") if match.metadata else ""
            print(f"[search] score={match.score:.3f}  text={text_context[:60]!r}")
            matches.append({
                "id":    match.id,
                "score": match.score,
                "text":  text_context,
            })

        print(f"[search] returned {len(matches)} result(s)")
        return matches

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{document_id}")
def delete_document(document_id: str):
    """Remove all Pinecone vectors that belong to a single document."""
    try:
        VectorStoreService.delete_document(document_id)
        return {"message": "Deleted successfully"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
def get_embedding(req: EmbedQuery):
    """Return the embedding vector for a text string — used by the semantic cache."""
    try:
        return {"embedding": EmbeddingService.embed_text(req.text)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
