import requests
import uuid

def upload_document(filepath, fname):
    url = "http://localhost:8000/upload"
    
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
