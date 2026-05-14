import io
import re
import csv
import zipfile


class DocumentExtractionService:

    # Image formats are handled differently: Gemini describes them during embedding.
    # We just return a placeholder so the upload view knows the file is valid.
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    @staticmethod
    def extract_text(file, filename: str) -> str:
        content = ""
        try:
            if filename.endswith(".txt"):
                file.seek(0)
                content = file.read().decode("utf-8", errors="ignore")

            elif filename.endswith(".pdf"):
                from pypdf import PdfReader
                file.seek(0)
                reader = PdfReader(io.BytesIO(file.read()))
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                content = "\n".join(pages_text)

            elif filename.endswith(".docx"):
                file.seek(0)
                with zipfile.ZipFile(io.BytesIO(file.read())) as z:
                    if "word/document.xml" in z.namelist():
                        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                        content = re.sub(r"<[^>]+>", " ", xml)
                        content = re.sub(r"\s+", " ", content).strip()

            elif filename.endswith(".csv"):
                file.seek(0)
                raw = file.read().decode("utf-8", errors="ignore")
                reader = csv.DictReader(io.StringIO(raw))
                rows = []
                for row in reader:
                    rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
                content = "\n".join(rows)

        except Exception as e:
            print(f"Django-side content extraction failed for {filename}:", e)

        if not filename.endswith(DocumentExtractionService.IMAGE_EXTENSIONS):
            file.seek(0)
        return content

    @staticmethod
    def extract_text_from_bytes(raw_bytes: bytes, filename: str) -> str:
        """Extract text from raw bytes — used when the file has already been read."""
        content = ""
        try:
            if filename.endswith(".txt"):
                content = raw_bytes.decode("utf-8", errors="ignore")

            elif filename.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_bytes))
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                content = "\n".join(pages_text)

            elif filename.endswith(".docx"):
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                    if "word/document.xml" in z.namelist():
                        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                        content = re.sub(r"<[^>]+>", " ", xml)
                        content = re.sub(r"\s+", " ", content).strip()

            elif filename.endswith(".csv"):
                raw = raw_bytes.decode("utf-8", errors="ignore")
                reader = csv.DictReader(io.StringIO(raw))
                rows = []
                for row in reader:
                    rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
                content = "\n".join(rows)

            elif filename.endswith(DocumentExtractionService.IMAGE_EXTENSIONS):
                # Images are described by Gemini during the embedding phase.
                # Return a non-empty placeholder so upload validation passes.
                content = f"Image file: {filename} — will be described by AI during embedding."

        except Exception as e:
            print(f"Django-side bytes extraction failed for {filename}:", e)

        return content
