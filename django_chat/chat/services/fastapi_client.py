import os
import io
import requests
import time

class FastAPIClient:
    @staticmethod
    def get_base_url():
        # FASTAPI_URL must be set in Render env vars for Django service
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def _base(cls):
        # strip /upload to get the root URL
        return cls.get_base_url().replace("/upload", "")

    @classmethod
    def wake_up(cls):
        """Ping FastAPI root to see if it's alive. Used for health checks only."""
        url = cls._base() + "/"
        for attempt in range(6):
            try:
                res = requests.get(url, timeout=8)
                if res.ok:
                    print(f"FastAPI awake after {attempt + 1} attempt(s)")
                    return True
            except Exception as e:
                print(f"FastAPI wake attempt {attempt + 1} failed: {e}")
            time.sleep(3)
        print("FastAPI did not wake in time.")
        return False

    @classmethod
    def upload_document(cls, file, filename: str = None) -> tuple[str, str]:
        """Send a document to FastAPI to be embedded into Pinecone.

        This is always called from a background thread in views.py so it is
        fine to block and retry here without worrying about Render timeouts.

        file     : file-like object (BytesIO)
        filename : original filename so FastAPI knows the file type
        """
        url = cls.get_base_url()
        fname = filename or getattr(file, 'name', None) or "upload.bin"

        # try up to 3 times - FastAPI on Render free tier can be slow to cold start
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                wait = 30
                print(f"[upload_document] waiting {wait}s before retry (attempt {attempt + 1})...")
                time.sleep(wait)

            try:
                file.seek(0)

                res = requests.post(
                    url,
                    files={"file": (fname, file, "application/octet-stream")},
                    timeout=90,
                )

                print(f"[upload_document] attempt {attempt + 1} - status={res.status_code}  fname={fname!r}")

                if res.status_code == 200:
                    res_json = res.json()
                    if "error" in res_json:
                        print(f"[upload_document] FastAPI error: {res_json['error']}")
                        return "", ""
                    return res_json.get("document_id", ""), res_json.get("text", "")

                # 502/503/504 usually means FastAPI is still waking up, retry
                if res.status_code in (502, 503, 504):
                    print(f"[upload_document] got {res.status_code}, will retry...")
                    continue

                print(f"[upload_document] unexpected {res.status_code}: {res.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"[upload_document] attempt {attempt + 1} timed out, retrying...")
            except requests.exceptions.ConnectionError as e:
                print(f"[upload_document] attempt {attempt + 1} connection error: {e}")
            except Exception as e:
                print(f"[upload_document] attempt {attempt + 1} error: {e}")

        print(f"[upload_document] all {MAX_ATTEMPTS} attempts failed for {fname!r}")
        return "", ""

    @classmethod
    def search_documents(cls, query: str, document_id: str = None) -> str:
        """Search the Pinecone vector DB via FastAPI."""
        try:
            url = cls._base() + "/search"
            payload = {"query": query, "top_k": 8}
            if document_id and str(document_id).strip():
                payload["document_id"] = document_id

            res = requests.post(url, json=payload, timeout=30)
            if not res.ok:
                return f"[RAG search failed: {res.text}]"

            results = res.json()
            if not results:
                print(f"[search_documents] Pinecone returned 0 results for: {query!r}")
                return "No relevant information found in the uploaded documents."

            # log the scores so we can see what pinecone actually returned
            for r in results:
                print(f"[search_documents] score={r.get('score', 0):.3f}  text={r.get('text', '')[:60]!r}")

            # 0.30 threshold - lower than before to handle short queries like 'what is ai'
            # if scores are still too low, check that the document embedded correctly
            SCORE_THRESHOLD = 0.30
            relevant = [item for item in results if item.get("score", 0) >= SCORE_THRESHOLD]

            if not relevant:
                scores = [round(r.get('score', 0), 3) for r in results]
                print(f"[search_documents] All scores below threshold {SCORE_THRESHOLD}: {scores}")
                return "No sufficiently relevant content found in the uploaded documents for this query."

            chunks = "\n---\n".join([item["text"] for item in relevant])
            return f"Relevant document excerpts:\n{chunks}"
        except Exception as e:
            return f"[RAG search error: {str(e)}]"

    @classmethod
    def delete_document(cls, document_id: str):
        """Delete a document from Pinecone via FastAPI."""
        try:
            url = cls._base() + f"/delete/{document_id}"
            res = requests.delete(url, timeout=30)
            if not res.ok:
                print("Failed calling delete API on FastAPI:", res.text)
        except Exception as e:
            print("Pinecone delete error:", e)
