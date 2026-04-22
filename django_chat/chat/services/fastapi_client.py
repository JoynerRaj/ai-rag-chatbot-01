import os
import io
import requests
import time

class FastAPIClient:
    @staticmethod
    def get_base_url():
        # set FASTAPI_URL in Render env vars, e.g. https://ai-rag-chatbot-01.onrender.com/upload
        return os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")

    @classmethod
    def _base(cls):
        # strip /upload to get the root URL for other endpoints
        return cls.get_base_url().replace("/upload", "")

    @classmethod
    def wake_up(cls):
        """Ping FastAPI to see if it's awake. Mainly used in health checks."""
        url = cls._base() + "/"
        for attempt in range(6):
            try:
                res = requests.get(url, timeout=8)
                if res.ok:
                    print(f"FastAPI is up after {attempt + 1} attempt(s)")
                    return True
            except Exception as e:
                print(f"Wake attempt {attempt + 1} failed: {e}")
            time.sleep(3)
        print("FastAPI didn't wake in time")
        return False

    @classmethod
    def upload_document(cls, file, filename: str = None, filepath: str = None) -> tuple[str, str]:
        """
        Send the document bytes to FastAPI so it can embed and store in Pinecone.
        Called from a background thread so we can safely wait/retry here.
        """
        url = cls.get_base_url()
        fname = filename or getattr(file, 'name', None) or "upload.bin"

        # try 3 times - FastAPI on Render free tier can take a minute to cold start
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                # give FastAPI some time to finish waking up before we retry
                print(f"[upload_document] waiting 30s before retry (attempt {attempt + 1})...")
                time.sleep(30)

            try:
                if filepath and os.path.exists(filepath):
                    import subprocess
                    import json
                    print(f"[upload_document] Streaming via curl: {filepath}")
                    result = subprocess.run([
                        "curl", "-s", "-X", "POST", url,
                        "-F", f"file=@{filepath};filename={fname};type=application/octet-stream"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        try:
                            res_json = json.loads(result.stdout)
                            if "error" in res_json:
                                print(f"[upload_document] FastAPI returned an error: {res_json['error']}")
                                return "", ""
                            return res_json.get("document_id", ""), res_json.get("text", "")
                        except Exception as e:
                            print(f"[upload_document] Failed parsing JSON: {result.stdout}")
                    else:
                        print(f"[upload_document] Curl failed: {result.stderr}")
                else:
                    file.seek(0)
                    res = requests.post(
                        url,
                        files={"file": (fname, file, "application/octet-stream")},
                        timeout=90,
                    )
                    print(f"[upload_document] attempt {attempt + 1} - status={res.status_code}  file={fname!r}")

                    if res.status_code == 200:
                        res_json = res.json()
                        if "error" in res_json:
                            print(f"[upload_document] FastAPI returned an error: {res_json['error']}")
                            return "", ""
                        return res_json.get("document_id", ""), res_json.get("text", "")

                    # 502/503/504 usually just means FastAPI is still starting, try again
                    if res.status_code in (502, 503, 504):
                        print(f"[upload_document] got {res.status_code}, retrying...")
                        continue

                    print(f"[upload_document] unexpected status {res.status_code}: {res.text[:200]}")

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
        """Search Pinecone through FastAPI and return the best matching text chunks."""
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

            # log all scores so we can see what Pinecone actually matched
            for r in results:
                print(f"[search_documents] score={r.get('score', 0):.3f}  text={r.get('text', '')[:60]!r}")

            # 0.30 is low enough to catch short queries like "what is ai"
            # if everything still scores below this, the document probably doesn't cover the topic
            SCORE_THRESHOLD = 0.30
            relevant = [item for item in results if item.get("score", 0) >= SCORE_THRESHOLD]

            if not relevant:
                scores = [round(r.get('score', 0), 3) for r in results]
                print(f"[search_documents] nothing above threshold {SCORE_THRESHOLD}: {scores}")
                return "No sufficiently relevant content found in the uploaded documents for this query."

            chunks = "\n---\n".join([item["text"] for item in relevant])
            return f"Relevant document excerpts:\n{chunks}"
        except Exception as e:
            return f"[RAG search error: {str(e)}]"

    @classmethod
    def delete_document(cls, document_id: str):
        """Tell FastAPI to remove a document from Pinecone."""
        try:
            url = cls._base() + f"/delete/{document_id}"
            res = requests.delete(url, timeout=30)
            if not res.ok:
                print("Pinecone delete failed:", res.text)
        except Exception as e:
            print("Pinecone delete error:", e)

    @classmethod
    def upload_audio(cls, file_like, filename: str, filepath: str = None) -> bool:
        """Upload an audio file to the audio event extractor."""
        url = cls._base() + "/audio/upload-audio/"
        try:
            if filepath and os.path.exists(filepath):
                # Use curl to stream the file from disk with ZERO memory overhead
                import subprocess
                import json
                
                print(f"[upload_audio] Streaming via curl: {filepath}")
                result = subprocess.run([
                    "curl", "-s", "-X", "POST", url,
                    "-F", f"file=@{filepath};filename={filename};type=audio/mpeg"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout)
                        print(f"[upload_audio] Success: {data}")
                        return True
                    except:
                        print(f"[upload_audio] Failed parsing JSON: {result.stdout}")
                else:
                    print(f"[upload_audio] Curl failed: {result.stderr}")
                return False
            else:
                # Fallback to requests (loads into memory, okay for small files)
                file_like.seek(0)
                res = requests.post(url, files={"file": (filename, file_like, "audio/mpeg")}, timeout=90)
                if res.ok:
                    print(f"[upload_audio] Success: {res.json()}")
                    return True
                print(f"[upload_audio] Failed {res.status_code}: {res.text}")
                return False
        except Exception as e:
            print(f"[upload_audio] Error: {e}")
            return False

    @classmethod
    def ask_audio(cls, query: str) -> str:
        """Ask a question about audio events."""
        url = cls._base() + "/audio/ask/"
        try:
            res = requests.post(url, json={"question": query}, timeout=30)
            if res.ok:
                data = res.json()
                return data.get("answer", "")
            return ""
        except Exception as e:
            print(f"[ask_audio] Error: {e}")
            return ""
