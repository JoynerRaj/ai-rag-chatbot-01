from fastapi import UploadFile
import fitz
import docx
import io

class DocumentService:
    @staticmethod
    def extract_text(file: UploadFile, content: bytes) -> str | None:
        """Pull raw text out of a .txt, .pdf, or .docx file."""
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

        # unsupported type
        return None

    @staticmethod
    def split_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
        """
        Break the text into overlapping word chunks before embedding.

        chunk_size=200 gives roughly 300 tokens per chunk, which works well
        with the sentence-transformer model. The overlap stops context from
        being cut off at chunk boundaries.
        """
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
