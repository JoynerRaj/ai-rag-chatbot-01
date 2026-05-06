import os
import uuid
import fitz       # PyMuPDF for PDF
import docx       # python-docx for Word files
import io

from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone, ServerlessSpec

# use 384 dims so we stay compatible with the existing Pinecone index
# Google gemini-embedding-2 supports output_dimensionality from 1 to 768
EMBEDDING_MODEL    = "gemini-embedding-2"
EMBEDDING_DIM      = 384
CHUNK_WORD_SIZE    = 200
CHUNK_OVERLAP      = 30

_pinecone_index = None   # module-level cache so we don't reconnect every call


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key    = os.environ.get("PINECONE_API_KEY", "")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-index")
    cloud      = os.environ.get("PINECONE_CLOUD", "aws")
    region     = os.environ.get("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set - check Render environment variables")

    pc = Pinecone(api_key=api_key)

    # list_indexes() returns different types across pinecone versions, handle both
    try:
        existing_names = pc.list_indexes().names()
    except AttributeError:
        existing_names = [idx.get("name", idx) if isinstance(idx, dict) else idx.name
                         for idx in pc.list_indexes()]

    print(f"[pinecone] existing indexes: {existing_names}")

    if index_name not in existing_names:
        print(f"[pinecone] creating index '{index_name}' dim={EMBEDDING_DIM}")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        print(f"[pinecone] using existing index '{index_name}'")

    _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def _embed_text(text: str) -> list[float]:
    """Embed text using Google text-embedding-004 at 384 dims."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set - check Render environment variables")

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return result.embeddings[0].values


def _split_text(text: str) -> list[str]:
    """Break text into overlapping word chunks for embedding."""
    words  = text.split()
    chunks = []
    step   = CHUNK_WORD_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_WORD_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Pull raw text from PDF, TXT, or DOCX."""
    fname = filename.lower()
    if fname.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    if fname.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    if fname.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def embed_and_store(file_bytes: bytes, filename: str) -> str:
    """
    Full pipeline: extract text → chunk → embed → upsert to Pinecone.
    Returns the document_id on success, raises an exception on failure.
    """
    print(f"[embed] starting for '{filename}' ({len(file_bytes)} bytes)")

    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{filename}' - is the file empty or password-protected?")

    print(f"[embed] extracted {len(text)} chars from '{filename}'")

    chunks = _split_text(text)
    print(f"[embed] split into {len(chunks)} chunks")

    document_id = str(uuid.uuid4())
    index = _get_pinecone_index()

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = _embed_text(chunk)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "document_id": document_id,
                "file_name": filename,
            },
        })
        # batch upsert every 50 vectors to stay within Pinecone request limits
        if len(vectors) >= 50:
            index.upsert(vectors)
            print(f"[embed] upserted batch at chunk {i + 1}/{len(chunks)}")
            vectors = []

    if vectors:
        index.upsert(vectors)

    print(f"[embed] done — doc_id={document_id!r}  chunks={len(chunks)}")
    return document_id


def search_documents(query: str, document_id: str = None, top_k: int = 8) -> str:
    """Find matching document chunks in Pinecone for the given query."""
    try:
        print(f"[search] query={query!r}  doc_id={document_id!r}")
        query_vec = _embed_text(query)
        index     = _get_pinecone_index()

        filter_ = {"document_id": {"$eq": document_id}} if document_id and str(document_id).strip() else None

        results = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter_,
        )

        matches = results.matches
        if not matches:
            print(f"[search] no results")
            return "No relevant information found in the uploaded documents."

        for m in matches:
            snippet = str(m.metadata.get("text", ""))[:60] if m.metadata else ""
            print(f"[search] score={m.score:.3f}  text={snippet!r}")

        THRESHOLD = 0.30
        relevant  = [m for m in matches if m.score >= THRESHOLD]

        if not relevant:
            scores = [round(m.score, 3) for m in matches]
            print(f"[search] all below threshold {THRESHOLD}: {scores}")
            return "No sufficiently relevant content found in the uploaded documents."

        chunks = "\n---\n".join(m.metadata.get("text", "") for m in relevant)
        return f"Relevant document excerpts:\n{chunks}"

    except Exception as e:
        print(f"[search] error: {e}")
        import traceback
        traceback.print_exc()
        return f"[Search error: {e}]"


def delete_document(document_id: str):
    """Remove all Pinecone vectors belonging to one document."""
    try:
        index = _get_pinecone_index()
        index.delete(filter={"document_id": {"$eq": document_id}})
        print(f"[pinecone] deleted doc_id={document_id!r}")
    except Exception as e:
        print(f"[pinecone] delete error: {e}")
