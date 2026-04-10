import io
import re
import zipfile

class DocumentExtractionService:
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

        except Exception as e:
            print(f"Django-side content extraction failed for {filename}:", e)
            
        file.seek(0)
        return content
