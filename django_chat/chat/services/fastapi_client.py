import os
import io
import uuid
import requests
import time


class FastAPIClient:

    @staticmethod
    def get_base_url():
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def _base(cls):
        return cls.get_base_url().replace("/upload", "")

    @classmethod
    def wake_up(cls):
        """Ping FastAPI until it responds. Waits up to ~3 minutes for cold start."""
        url = cls._base() + "/"
        print(f"[wake_up] pinging {url} ...")
        for attempt in range(20):  # 20 × 10s = up to 3.3 minutes
            try:
                res = requests.get(url, timeout=12)
                if res.ok:
                    print(f"[wake_up] FastAPI is up (attempt {attempt + 1})")
                    return True
            except Exception as e:
                print(f"[wake_up] attempt {attempt + 1}/20 failed: {e}")
            time.sleep(10)
        print("[wake_up] FastAPI never responded - giving up")
        return False

    @classmethod
    def upload_document(cls, file, filename=None, filepath=None):
        """Embed a document via FastAPI and return (pinecone_id, extracted_text)."""
        url = cls.get_base_url()
        fname = filename or getattr(file, "name", None) or "upload.bin"

        # wait for FastAPI to wake up before trying to upload
        # on Render free tier it can take 60-90 seconds to cold start
        if not cls.wake_up():
            print(f"[upload_document] FastAPI never came up - aborting {fname!r}")
            return "", ""

        # FastAPI is alive, now attempt the upload with a generous timeout
        for attempt in range(3):
            if attempt > 0:
                print(f"[upload_document] retry {attempt + 1}/3 in 15s...")
                time.sleep(15)
            try:
                if filepath and os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        res = requests.post(
                            url,
                            files={"file": (fname, f, "application/octet-stream")},
                            timeout=120,
                        )
                else:
                    file.seek(0)
                    res = requests.post(
                        url,
                        files={"file": (fname, file, "application/octet-stream")},
                        timeout=120,
                    )

                print(f"[upload_document] attempt {attempt + 1} → {res.status_code}")

                if res.status_code == 200:
                    data = res.json()
                    if "error" in data:
                        print(f"[upload_document] FastAPI error: {data['error']}")
                        return "", ""
                    return data.get("document_id", ""), data.get("text", "")

                if res.status_code in (502, 503, 504):
                    print(f"[upload_document] got {res.status_code}, retrying...")
                    continue

                print(f"[upload_document] unexpected {res.status_code}: {res.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"[upload_document] attempt {attempt + 1} timed out (120s)")
            except Exception as e:
                print(f"[upload_document] attempt {attempt + 1} error: {e}")

        print(f"[upload_document] all retries exhausted for {fname!r}")
        return "", ""

    @classmethod
    def search_documents(cls, query, document_id=None):
        """Return the best matching text chunks from Pinecone for the given query."""
        try:
            payload = {"query": query, "top_k": 8}
            if document_id and str(document_id).strip():
                payload["document_id"] = document_id

            res = requests.post(cls._base() + "/search", json=payload, timeout=30)
            if not res.ok:
                return f"[RAG search failed: {res.text}]"

            results = res.json()
            if not results:
                return "No relevant information found in the uploaded documents."

            for r in results:
                print(f"[search] score={r.get('score', 0):.3f}  {r.get('text', '')[:60]!r}")

            THRESHOLD = 0.30
            relevant = [r for r in results if r.get("score", 0) >= THRESHOLD]
            if not relevant:
                return "No sufficiently relevant content found in the uploaded documents for this query."

            chunks = "\n---\n".join(r["text"] for r in relevant)
            return f"Relevant document excerpts:\n{chunks}"

        except Exception as e:
            return f"[RAG search error: {e}]"

    @classmethod
    def delete_document(cls, document_id):
        try:
            res = requests.delete(cls._base() + f"/delete/{document_id}", timeout=30)
            if not res.ok:
                print("Pinecone delete failed:", res.text)
        except Exception as e:
            print("Pinecone delete error:", e)

    @classmethod
    def upload_audio(cls, file_like, filename, filepath=None):
        """Send audio to FastAPI for sound-event detection. Returns True on success."""
        url = cls._base() + "/audio/upload-audio/"

        def _multipart_gen(path, file_name, boundary):
            yield f"--{boundary}\r\n".encode()
            yield f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode()
            yield b"Content-Type: audio/mpeg\r\n\r\n"
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
            yield b"\r\n"
            yield f"--{boundary}--\r\n".encode()

        try:
            if filepath and os.path.exists(filepath):
                boundary = uuid.uuid4().hex
                headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
                res = requests.post(
                    url,
                    data=_multipart_gen(filepath, filename, boundary),
                    headers=headers,
                    timeout=120,
                )
            else:
                file_like.seek(0)
                res = requests.post(
                    url,
                    files={"file": (filename, file_like, "audio/mpeg")},
                    timeout=120,
                )

            if res.ok:
                return True
            print(f"[upload_audio] {res.status_code}: {res.text}")
            return False

        except Exception as e:
            print(f"[upload_audio] error: {e}")
            return False

    @classmethod
    def ask_audio(cls, question):
        """Query the audio event RAG endpoint. Returns the answer or empty string."""
        try:
            res = requests.post(
                cls._base() + "/audio/ask/",
                json={"question": question},
                timeout=30,
            )
            if res.ok:
                return res.json().get("answer", "")
            print(f"[ask_audio] {res.status_code}: {res.text}")
            return ""
        except Exception as e:
            print(f"[ask_audio] error: {e}")
            return ""
