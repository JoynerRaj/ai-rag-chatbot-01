import os
import uuid
import io
import re
import traceback

import fitz
import docx
from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone, ServerlessSpec


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM   = 384

# Semantic chunking settings
MAX_CHUNK_WORDS   = 250   # hard cap per chunk
MIN_CHUNK_WORDS   = 20    # ignore chunks shorter than this
CHUNK_OVERLAP_WORDS = 40  # words to carry over from the previous chunk

_pinecone_index = None


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key    = os.environ.get("PINECONE_API_KEY", "")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-index")
    cloud      = os.environ.get("PINECONE_CLOUD", "aws")
    region     = os.environ.get("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)

    try:
        existing_names = pc.list_indexes().names()
    except AttributeError:
        existing_names = [
            idx.get("name", idx) if isinstance(idx, dict) else idx.name
            for idx in pc.list_indexes()
        ]

    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def _embed_text(text):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return result.embeddings[0].values


def _split_into_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences on '. ', '! ', '? ' boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [p for p in parts if p.strip()]


def _semantic_chunk(text: str) -> list[str]:
    """
    Paragraph-aware semantic chunker.

    Strategy:
      1. Split on blank lines (paragraph boundaries) — these are the natural
         semantic units in most documents.
      2. If a paragraph fits within MAX_CHUNK_WORDS, keep it as one chunk.
      3. If it is too long, split it on sentence boundaries.
      4. Accumulate sentences into chunks up to MAX_CHUNK_WORDS.
      5. Carry CHUNK_OVERLAP_WORDS of context from the previous chunk so no
         information is lost at chunk boundaries.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    carry_words: list[str] = []  # overlap words from previous chunk

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = _split_into_sentences(para)
        current_words: list[str] = list(carry_words)

        for sentence in sentences:
            s_words = sentence.split()

            # If adding this sentence would overflow, flush first
            if len(current_words) + len(s_words) > MAX_CHUNK_WORDS and len(current_words) >= MIN_CHUNK_WORDS:
                chunk_text = " ".join(current_words)
                chunks.append(chunk_text)
                # carry the last N words as overlap into the next chunk
                carry_words = current_words[-CHUNK_OVERLAP_WORDS:] if len(current_words) > CHUNK_OVERLAP_WORDS else list(current_words)
                current_words = list(carry_words)

            current_words.extend(s_words)

        # End of paragraph — flush what we have if it's big enough
        if len(current_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(current_words))
            carry_words = current_words[-CHUNK_OVERLAP_WORDS:] if len(current_words) > CHUNK_OVERLAP_WORDS else list(current_words)
        elif current_words:
            # Short paragraph — merge it into the carry for the next paragraph
            carry_words = current_words

    # Flush any remaining words
    if len(carry_words) >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(carry_words))

    return chunks


def _extract_text_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) tuples — 1-indexed pages."""
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((i, text))
    return pages


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Return plain text from a file. Used by legacy callers."""
    fname = filename.lower()
    if fname.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    if fname.endswith(".pdf"):
        pages = _extract_text_pdf(file_bytes)
        return "\n\n".join(text for _, text in pages)
    if fname.endswith(".docx"):
        doc_obj = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc_obj.paragraphs)
    if fname.endswith(".csv"):
        import csv
        raw = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(raw))
        rows = []
        for row in reader:
            rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)
    return ""


def _describe_image_with_gemini(file_bytes: bytes, filename: str) -> str:
    """
    Ask Gemini to describe an image so we can embed the description.
    Returns a plain-text description of the image content.
    """
    import base64
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        ext = filename.lower().rsplit(".", 1)[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

        client = genai.Client(api_key=api_key)
        b64 = base64.b64encode(file_bytes).decode()
        contents = [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {"text": "Describe this image in detail. Extract all visible text, data, charts, diagrams, and key information. Be comprehensive."},
                ]
            }
        ]
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        return response.text or ""
    except Exception as e:
        print(f"[embed] image description failed: {e}")
        return ""


def embed_and_store(file_bytes: bytes, filename: str, doc_title: str = "") -> str:
    """
    Main entry point: extract text → semantic chunk → embed → upsert to Pinecone.

    Returns the document_id (a UUID string) that links all Pinecone vectors
    for this document together.
    """
    print(f"[embed] starting '{filename}' ({len(file_bytes)} bytes)")
    fname = filename.lower()
    document_id = str(uuid.uuid4())
    index = _get_pinecone_index()
    vectors = []

    # ── Image: describe via Gemini, embed description ────────────────────────
    if fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
        description = _describe_image_with_gemini(file_bytes, filename)
        if not description.strip():
            raise ValueError(f"Could not extract description from image '{filename}'")
        chunks_with_meta = [(description, 1)]  # treat whole description as page 1

    # ── PDF: page-aware chunking ──────────────────────────────────────────────
    elif fname.endswith(".pdf"):
        pages = _extract_text_pdf(file_bytes)
        chunks_with_meta = []  # (chunk_text, page_num)
        for page_num, page_text in pages:
            for chunk in _semantic_chunk(page_text):
                chunks_with_meta.append((chunk, page_num))

    # ── All other formats: page 1 ─────────────────────────────────────────────
    else:
        raw_text = extract_text(file_bytes, filename)
        if not raw_text.strip():
            raise ValueError(f"No text extracted from '{filename}'")
        chunks_with_meta = [(chunk, 1) for chunk in _semantic_chunk(raw_text)]

    if not chunks_with_meta:
        raise ValueError(f"No chunks generated from '{filename}'")

    print(f"[embed] {len(chunks_with_meta)} semantic chunks from '{filename}'")

    for i, (chunk_text, page_num) in enumerate(chunks_with_meta):
        embedding = _embed_text(chunk_text)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text":        chunk_text,
                "document_id": document_id,
                "file_name":   filename,
                "doc_title":   doc_title or filename,
                "page_num":    page_num,
                "chunk_index": i,
            },
        })
        # Upsert in batches of 50 to stay within Pinecone request limits
        if len(vectors) >= 50:
            index.upsert(vectors=vectors)
            print(f"[embed] upserted batch ending at chunk {i + 1}/{len(chunks_with_meta)}")
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

    print(f"[embed] done doc_id={document_id} total_chunks={len(chunks_with_meta)}")
    return document_id


def search_documents(query: str, document_id: str = None, top_k: int = 8) -> str:
    """
    Dense vector search in Pinecone, then BM25 reranking for hybrid retrieval.

    Returns a formatted context string that includes source citations so the
    LLM can reference them in its answer.
    """
    try:
        print(f"[search] query={query!r} doc_id={document_id!r}")
        query_vec = _embed_text(query)
        index = _get_pinecone_index()
        filter_ = {"document_id": {"$eq": document_id}} if document_id and str(document_id).strip() else None

        # Fetch more candidates than needed so BM25 can rerank effectively
        results = index.query(
            vector=query_vec,
            top_k=min(top_k * 3, 25),
            include_metadata=True,
            filter=filter_,
        )

        matches = results.matches
        if not matches:
            return "No relevant information found in the uploaded documents."

        # Apply a relevance threshold to filter out weak matches
        THRESHOLD = 0.55
        relevant = [m for m in matches if m.score >= THRESHOLD]

        if not relevant:
            return "No sufficiently relevant content found in the uploaded documents."

        # BM25 reranking: combine dense score + keyword overlap score
        relevant = _bm25_rerank(query, relevant, top_n=top_k)

        for m in relevant:
            snippet = str(m.metadata.get("text", ""))[:60] if m.metadata else ""
            print(f"[search] score={m.score:.3f} text={snippet!r}")

        # Build context string with source references for the LLM
        parts = []
        for m in relevant:
            meta = m.metadata or {}
            text      = meta.get("text", "")
            file_name = meta.get("file_name", "unknown")
            page_num  = meta.get("page_num", 1)
            parts.append(f"[Source: {file_name}, Page {page_num}]\n{text}")

        return "Relevant document excerpts:\n\n" + "\n\n---\n\n".join(parts)

    except Exception as e:
        print(f"[search] error: {e}")
        traceback.print_exc()
        return ""


def _bm25_rerank(query: str, matches: list, top_n: int = 8) -> list:
    """
    Simple BM25-style keyword reranking on top of dense results.

    We compute a keyword overlap score and blend it with the dense cosine score
    so documents that match both semantically AND lexically rank highest.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # rank-bm25 not installed — just return dense results trimmed to top_n
        return matches[:top_n]

    query_tokens = query.lower().split()
    corpus = [m.metadata.get("text", "") for m in matches]
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalise BM25 scores to [0, 1]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    norm_bm25 = [s / max_bm25 for s in bm25_scores]

    # Blend: 70% dense similarity + 30% BM25 keyword score
    combined = [(0.7 * m.score + 0.3 * norm_bm25[i], m) for i, m in enumerate(matches)]
    combined.sort(key=lambda x: x[0], reverse=True)

    return [m for _, m in combined[:top_n]]


def search_with_sources(query: str, document_id: str = None, top_k: int = 8) -> tuple[str, list[dict]]:
    """
    Like search_documents() but also returns structured source metadata
    so the AI agent can pass citations through to the frontend.

    Returns: (context_string, [{"file_name": ..., "page_num": ...}, ...])
    """
    try:
        query_vec = _embed_text(query)
        index = _get_pinecone_index()
        filter_ = {"document_id": {"$eq": document_id}} if document_id and str(document_id).strip() else None

        results = index.query(
            vector=query_vec,
            top_k=min(top_k * 3, 25),
            include_metadata=True,
            filter=filter_,
        )

        matches = results.matches
        if not matches:
            return ("", [])

        THRESHOLD = 0.55
        relevant = [m for m in matches if m.score >= THRESHOLD]
        if not relevant:
            return ("", [])

        relevant = _bm25_rerank(query, relevant, top_n=top_k)

        parts = []
        sources = []
        seen_sources = set()  # avoid duplicate citations for the same file+page

        for m in relevant:
            meta = m.metadata or {}
            text      = meta.get("text", "")
            file_name = meta.get("file_name", "unknown")
            page_num  = meta.get("page_num", 1)
            parts.append(f"[Source: {file_name}, Page {page_num}]\n{text}")

            key = (file_name, page_num)
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"file_name": file_name, "page_num": page_num})

        context = "Relevant document excerpts:\n\n" + "\n\n---\n\n".join(parts)
        return (context, sources)

    except Exception as e:
        print(f"[search_with_sources] error: {e}")
        traceback.print_exc()
        return ("", [])


def delete_document(document_id: str):
    try:
        _get_pinecone_index().delete(filter={"document_id": {"$eq": document_id}})
        print(f"[pinecone] deleted doc_id={document_id}")
    except Exception as e:
        print(f"[pinecone] delete error: {e}")
