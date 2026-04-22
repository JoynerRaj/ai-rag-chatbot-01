from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from audio.db import insert_events, execute_query
from audio.event_detector import process_audio
from audio.query_parser import generate_sql_from_intent
from audio.response_generator import generate_natural_language_answer

router = APIRouter(prefix="/audio", tags=["Audio"])


class AskRequest(BaseModel):
    question: str


@router.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Receive an audio file, run event detection, and store results in SQLite."""
    import os
    import tempfile

    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file.filename)
    try:
        with os.fdopen(temp_fd, "wb") as tmp:
            while chunk := await file.read(65536):
                tmp.write(chunk)

        events = process_audio(temp_path)
        if events:
            insert_events(events)

        return {"message": f"Processed {file.filename}", "events_detected": len(events)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/ask/")
async def ask_audio_question(req: AskRequest) -> Dict[str, Any]:
    """Answer a question about stored audio events using SQL + Gemini."""
    try:
        sql = generate_sql_from_intent(req.question)
        results = execute_query(sql)
        answer = generate_natural_language_answer(req.question, results)
        return {"answer": answer, "executed_sql": sql, "raw_results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
