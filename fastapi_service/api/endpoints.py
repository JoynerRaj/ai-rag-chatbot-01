import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.document import DocumentService
from services.embedding import EmbeddingService
from services.vectordb import VectorDBService
from schemas.document import SearchQuery, EmbedQuery

router = APIRouter()

@router.get("/")
def home():
    return {"message": "FastAPI Upload Service Running"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    text = DocumentService.extract_text(file, content)

    if not text:
        print(f"[upload] no text from file: {file.filename!r}")
        return {"error": "Unsupported file type or empty document"}

    chunks = DocumentService.split_text(text)
    document_id = str(uuid.uuid4())
    print(f"[upload] {file.filename!r}  chunks={len(chunks)}  id={document_id}")

    # build one vector per chunk with the text and doc ID in the metadata
    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": EmbeddingService.embed_text(chunk),
            "metadata": {
                "text": chunk,
                "document_id": document_id,
                "file_name": file.filename
            }
        })

    try:
        VectorDBService.upsert_vectors(vectors)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[upload] Pinecone upsert failed: {e}")
        return {"error": f"Vector DB error: {str(e)}"}

    return {
        "message": "Stored in Pinecone successfully",
        "document_id": document_id,
        "text": text
    }

@router.post("/search")
def search_pinecone(req: SearchQuery):
    try:
        print(f"[search] query={req.query!r}  doc_id={req.document_id!r}  top_k={req.top_k}")

        query_embedding = EmbeddingService.embed_text(req.query)

        results = VectorDBService.search(
            query_embedding=query_embedding,
            top_k=req.top_k,
            document_id=req.document_id
        )

        matches = []
        for match in results.matches:
            text_context = ""
            if match.metadata:
                text_context = match.metadata.get("text", "")

            print(f"[search] score={match.score:.3f}  id={match.id}  text={text_context[:60]!r}")

            matches.append({
                "id": match.id,
                "score": match.score,
                "text": text_context
            })

        print(f"[search] {len(matches)} result(s) returned")
        return matches
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{document_id}")
def delete_document(document_id: str):
    try:
        VectorDBService.delete_document(document_id)
        return {"message": "Deleted successfully"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
def get_embedding(req: EmbedQuery):
    """Used by the semantic cache to get an embedding for a query string."""
    try:
        return {"embedding": EmbeddingService.embed_text(req.text)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
