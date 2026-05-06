import os
import uuid
import fitz       # PyMuPDF for PDF
import docx       # python-docx for Word files
import io

from google import genai
from pinecone import Pinecone, ServerlessSpec

# we embed directly in Django using Google's text-embedding-004 model
# this removes the FastAPI cold-start problem entirely
EMBEDDING_MODEL    = "text-embedding-004"
EMBEDDING_DIM      = 768    # Google text-embedding-004 output size
CHUNK_WORD_SIZE    = 200    # words per chunk
CHUNK_OVERLAP      = 30     # overlap to keep context across boundaries


def _get_pinecone_index():
    """Connect to Pinecone and return the index, creating it if needed."""
    api_key    = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-index")
    cloud      = os.environ.get("PINECONE_CLOUD", "aws")
    region     = os.environ.get("PINECONE_REGION", "us-east-1")

    pc = Pinecone(api_key=api_key)

    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[pinecone] created new index '{index_name}' dim={EMBEDDING_DIM}")

    return pc.Index(index_name)


def _embed_text(text: str) -> list[float]:
    """Ask Google Gemini to embed a piece of text and return the vector."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def _split_text(text: str) -> list[str]:
    """Split text into overlapping word chunks for better semantic search coverage."""
    words  = text.split()
    chunks = []
    step   = CHUNK_WORD_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_WORD_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Pull raw text from a PDF, TXT, or DOCX file."""
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
    Extract text → chunk → embed with Google → upsert into Pinecone.
    Returns the document_id string on success, or raises on error.
    """
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError(f"Could not extract text from '{filename}'")

    chunks = _split_text(text)
    print(f"[embed] '{filename}' → {len(chunks)} chunks")

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
        # upsert in batches of 50 to stay within Pinecone request limits
        if len(vectors) >= 50:
            index.upsert(vectors)
            print(f"[embed] upserted batch up to chunk {i + 1}")
            vectors = []

    if vectors:
        index.upsert(vectors)

    print(f"[embed] done — document_id={document_id!r}  total chunks={len(chunks)}")
    return document_id


def search_documents(query: str, document_id: str = None, top_k: int = 8) -> str:
    """
    Embed the query and find matching chunks in Pinecone.
    Returns the best chunks as a formatted string for the AI prompt.
    """
    try:
        query_vec = _embed_text(query)
        index     = _get_pinecone_index()

        filter_ = {"document_id": {"$eq": document_id}} if document_id and document_id.strip() else None

        results = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter_,
        )

        matches = results.matches
        if not matches:
            print(f"[search] no results for: {query!r}")
            return "No relevant information found in the uploaded documents."

        for m in matches:
            print(f"[search] score={m.score:.3f}  text={str(m.metadata.get('text',''))[:60]!r}")

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
        return f"[Search error: {e}]"


def delete_document(document_id: str):
    """Remove all Pinecone vectors that belong to a specific document."""
    try:
        index = _get_pinecone_index()
        index.delete(filter={"document_id": {"$eq": document_id}})
        print(f"[pinecone] deleted vectors for document_id={document_id!r}")
    except Exception as e:
        print(f"[pinecone] delete error: {e}")
