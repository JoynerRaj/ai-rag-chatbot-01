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

        file     : file-like object (Django upload OR io.BytesIO)
        filename : explicit filename — required when file is a BytesIO
        
        NOTE: No wake_up() call here — that added 60+ seconds to the Django
        HTTP request and triggered Render's load-balancer 502. FastAPI wakes
        naturally within the 90s upload timeout.
        """
        url = cls.get_base_url()
        try:
            fname = filename or getattr(file, 'name', None) or "upload.bin"
            file.seek(0)

            res = requests.post(
                url,
                files={"file": (fname, file, "application/octet-stream")},
                timeout=90,   # Render free tier wakes in ~30-60s; 90s gives a safe margin
            )

            print(f"[upload_document] status={res.status_code}  fname={fname!r}")

            if res.status_code == 200:
                res_json = res.json()
                if "error" in res_json:
                    print(f"[upload_document] FastAPI error: {res_json['error']}")
                    return "", ""
                return res_json.get("document_id", ""), res_json.get("text", "")
            else:
                print(f"[upload_document] non-200 response: {res.text[:200]}")
        except Exception as e:
            print(f"[upload_document] exception: {e}")
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
