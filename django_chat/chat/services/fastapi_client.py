import os
import requests
import time

class FastAPIClient:
    @staticmethod
    def get_base_url():
        # FASTAPI_URL must be set in Render env vars for Django service:
        #   https://ai-rag-chatbot-01.onrender.com/upload
        # For local Docker Compose the internal name is used automatically.
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def _base(cls):
        """Root URL without the /upload suffix."""
        return cls.get_base_url().replace("/upload", "")

    @classmethod
    def wake_up(cls):
        """Ping the FastAPI health endpoint to wake it from Render cold start.
        Makes up to 6 quick attempts over ~55 seconds.
        """
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
        """Uploads document to FastAPI for Pinecone embedding.

        Retries a few times to handle Render free-tier cold starts gracefully.
        file     : file-like object (Django upload OR io.BytesIO)
        filename : explicit filename — required when file is a BytesIO
        """
        url = cls.get_base_url()
        fname = filename or getattr(file, 'name', None) or "upload.bin"

        # read the bytes once so we can seek back before each retry attempt
        raw_bytes = file.read()

        MAX_ATTEMPTS = 4
        WAIT_SECONDS = [0, 20, 30, 40]  # how long to wait before each attempt

        for attempt in range(MAX_ATTEMPTS):
            wait = WAIT_SECONDS[attempt]
            if wait > 0:
                print(f"[upload_document] waiting {wait}s before retry (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
                time.sleep(wait)

            try:
                import io
                file_like = io.BytesIO(raw_bytes)

                res = requests.post(
                    url,
                    files={"file": (fname, file_like, "application/octet-stream")},
                    timeout=60,
                )

                print(f"[upload_document] attempt {attempt + 1} — status={res.status_code}  fname={fname!r}")

                if res.status_code == 200:
                    res_json = res.json()
                    if "error" in res_json:
                        print(f"[upload_document] FastAPI returned error: {res_json['error']}")
                        # document issue (bad file), no point retrying
                        return "", ""
                    return res_json.get("document_id", ""), res_json.get("text", "")

                # 502/503 usually means FastAPI is still waking up, retry
                if res.status_code in (502, 503, 504):
                    print(f"[upload_document] got {res.status_code}, will retry...")
                    continue

                print(f"[upload_document] unexpected status {res.status_code}: {res.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"[upload_document] attempt {attempt + 1} timed out, will retry...")
            except requests.exceptions.ConnectionError as e:
                print(f"[upload_document] attempt {attempt + 1} connection error: {e}")
            except Exception as e:
                print(f"[upload_document] attempt {attempt + 1} unexpected error: {e}")

        print(f"[upload_document] all {MAX_ATTEMPTS} attempts failed for {fname!r}")
        return "", ""

    @classmethod
    def search_documents(cls, query: str, document_id: str = None) -> str:
        """Searches documents via FastAPI Pinecone vector DB."""
        try:
            url = cls.get_base_url().replace("/upload", "") + "/search"
            payload = {"query": query, "top_k": 8}  # fetch more, we'll filter by score
            if document_id and str(document_id).strip():
                payload["document_id"] = document_id

            res = requests.post(url, json=payload, timeout=30)
            if not res.ok:
                return f"[RAG search failed: {res.text}]"

            results = res.json()
            if not results:
                return "No relevant information found in the uploaded documents."

            # only keep chunks with a meaningful similarity score (0.45+)
            # lower than this usually means the document doesn't actually cover the topic
            SCORE_THRESHOLD = 0.45
            relevant = [item for item in results if item.get("score", 0) >= SCORE_THRESHOLD]

            if not relevant:
                return "No sufficiently relevant content found in the uploaded documents for this query."

            chunks = "\n---\n".join([item["text"] for item in relevant])
            return f"Relevant document excerpts:\n{chunks}"
        except Exception as e:
            return f"[RAG search error: {str(e)}]"

    @classmethod
    def delete_document(cls, document_id: str):
        """Deletes a document from FastAPI Pinecone DB."""
        try:
            url = cls._base() + f"/delete/{document_id}"
            res = requests.delete(url, timeout=30)
            if not res.ok:
                print("Failed calling delete API on FastAPI:", res.text)
        except Exception as e:
            print("Pinecone delete error:", e)
