import os
import requests

class FastAPIClient:
    @staticmethod
    def get_base_url():
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def upload_document(cls, file) -> tuple[str, str]:
        """Uploads document to FastAPI for Pinecone embedding."""
        url = cls.get_base_url()
        try:
            file.seek(0)
            res = requests.post(url, files={"file": file}, timeout=60)
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
            payload = {"query": query, "top_k": 5}
            if document_id and str(document_id).strip():
                payload["document_id"] = document_id

            res = requests.post(url, json=payload, timeout=30)
            if not res.ok:
                return f"[RAG search failed: {res.text}]"

            results = res.json()
            if not results:
                return "No relevant information found in the uploaded documents."

            chunks = "\n---\n".join([item["text"] for item in results])
            return f"Relevant document excerpts:\n{chunks}"
        except Exception as e:
            return f"[RAG search error: {str(e)}]"

    @classmethod
    def delete_document(cls, document_id: str):
        """Deletes a document from FastAPI Pinecone DB."""
        try:
            url = cls.get_base_url().replace("/upload", "") + f"/delete/{document_id}"
            res = requests.delete(url, timeout=30)
            if not res.ok:
                print("Failed calling delete API on FastAPI:", res.text)
        except Exception as e:
            print("Pinecone delete error:", e)
