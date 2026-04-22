from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from audio.db import insert_events, execute_query
from audio.event_detector import process_audio
from audio.query_parser import generate_sql_from_intent
from audio.response_generator import generate_natural_language_answer

router = APIRouter(prefix="/audio", tags=["Audio Event RAG"])


class AskRequest(BaseModel):
    question: str


@router.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, streams it to disk to avoid loading it into RAM,
    runs the (mock) sound-event detector, and saves any detected events to SQLite.

    NOTE: Speech transcription is handled separately by the Django service
    using Gemini's Files API directly inside a background thread, so this
    endpoint stays fast and never blocks the FastAPI event loop.
    """
    import os
    import tempfile

    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Stream the incoming bytes to a temp file in 64 KB chunks so we never
    # hold the entire audio in memory at once.
    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file.filename)
    try:
        with os.fdopen(temp_fd, "wb") as tmp:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)

        print(f"[audio] received {file.filename!r} → temp file {temp_path}")

        # Sound-event detection (currently mock; swap in YAMNet / CLAP here).
        # This runs fast because it just returns hardcoded data for now.
        events = process_audio(temp_path)
        if events:
            insert_events(events)

        return {
            "message": f"Processed {file.filename} — {len(events)} events detected.",
            "events_detected": len(events),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up regardless of success or failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/ask/")
async def ask_audio_question(req: AskRequest) -> Dict[str, Any]:
    """
    Answer a natural language question about the stored audio events
    by converting the question to SQL, executing it, and formatting
    the result into a plain English response.
    """
    try:
        sql_query = generate_sql_from_intent(req.question)
        print(f"[Audio RAG] Generated SQL: {sql_query}")

        db_results = execute_query(sql_query)
        print(f"[Audio RAG] DB Results: {db_results}")

        # Returns "" when results are empty so the Django fallback chain triggers
        final_answer = generate_natural_language_answer(req.question, db_results)

        return {
            "answer": final_answer,
            "executed_sql": sql_query,
            "raw_results": db_results,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
