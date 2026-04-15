import os
import requests
import time

class FastAPIClient:
    @staticmethod
    def get_base_url():
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def _base(cls):
        """Root URL without the /upload suffix."""
        return cls.get_base_url().replace("/upload", "")

    @classmethod
    def wake_up(cls):
        """Ping the FastAPI health endpoint to wake it from Render cold start.
        Waits up to 40 seconds for the service to come alive.
        """
        url = cls._base() + "/"
        for attempt in range(4):
            try:
                res = requests.get(url, timeout=15)
                if res.ok:
                    print(f"FastAPI awake after {attempt + 1} attempt(s)")
                    return True
            except Exception as e:
                print(f"FastAPI wake attempt {attempt + 1} failed: {e}")
            time.sleep(5)
        print("FastAPI did not wake in time.")
        return False

    @classmethod
    def upload_document(cls, file) -> tuple[str, str]:
        """Uploads document to FastAPI for Pinecone embedding.
        Wakes up the Render service first if it's cold-starting.
        """
        cls.wake_up()  # ensure FastAPI is alive before the real upload
        url = cls.get_base_url()
        try:
            file.seek(0)
            res = requests.post(url, files={"file": file}, timeout=120)
            if res.status_code == 200:
                res_json = res.json()
                return res_json.get("document_id", ""), res_json.get("text", "")
        except Exception as e:
            print("FastAPI upload failed:", e)
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
