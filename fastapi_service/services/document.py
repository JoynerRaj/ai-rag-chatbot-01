from fastapi import UploadFile
import fitz
import docx
import io

class DocumentService:
    @staticmethod
    def extract_text(file: UploadFile, content: bytes) -> str | None:
        filename = file.filename.lower()

        if filename.endswith(".txt"):
            return content.decode("utf-8")

        elif filename.endswith(".pdf"):
            text = ""
            pdf = fitz.open(stream=content, filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text

        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([para.text for para in doc.paragraphs])

        return None

    @staticmethod
    def split_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
        """Split text into overlapping chunks for better semantic search coverage.
        
        chunk_size: number of words per chunk (200 gives ~300 tokens, good for embeddings)
        overlap: words shared between consecutive chunks to avoid losing context at boundaries
        """
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
