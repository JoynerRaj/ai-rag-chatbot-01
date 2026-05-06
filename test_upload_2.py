import asyncio
from fastapi import FastAPI, UploadFile, File
import uvicorn
import threading
import requests
import uuid

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8005)

t = threading.Thread(target=run_server, daemon=True)
t.start()
import time
time.sleep(2)

def upload_document(filepath, fname):
    url = "http://127.0.0.1:8005/upload"
    
    def _multipart_gen(path, file_name, boundary):
        yield f"--{boundary}\r\n".encode()
        yield f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode()
        yield b"Content-Type: application/octet-stream\r\n\r\n"
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk
        yield b"\r\n"
        yield f"--{boundary}--\r\n".encode()

    boundary = uuid.uuid4().hex
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    res = requests.post(
        url,
        data=_multipart_gen(filepath, fname, boundary),
        headers=headers,
        timeout=90,
    )
    print(res.status_code)
    print(res.text)

with open("test.txt", "w") as f:
    f.write("hello world")

upload_document("test.txt", "test.txt")
