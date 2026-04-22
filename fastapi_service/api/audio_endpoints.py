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
    Endpoint to receive an audio file, process it into events, and store them in SQLite.
    The file is streamed directly to disk to avoid loading the whole thing into RAM.
    """
    import os
    import tempfile

    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # Write the upload to a temp file in 64 KB chunks so we never hold
    # the entire audio file in memory at once.
    temp_fd, temp_path = tempfile.mkstemp(suffix="_" + file.filename)
    try:
        with os.fdopen(temp_fd, "wb") as tmp:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)

        print(f"[audio] received {file.filename!r} → temp file {temp_path}")

        # event_detector currently uses mock data; in production it would
        # open temp_path with a real ML model (YAMNet / CLAP).
        events = process_audio(temp_path)

        if events:
            insert_events(events)

        return {
            "message": f"Successfully extracted and saved {len(events)} events from {file.filename}.",
            "events_detected": len(events)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temp file regardless of success or failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/ask/")
async def ask_audio_question(req: AskRequest) -> Dict[str, Any]:
    """
    Endpoint to answer natural language questions about the stored audio events.
    """
    try:
        # 1. Convert question to SQL
        sql_query = generate_sql_from_intent(req.question)
        print(f"[Audio RAG] Generated SQL: {sql_query}")
        
        # 2. Execute SQL
        db_results = execute_query(sql_query)
        print(f"[Audio RAG] DB Results: {db_results}")
        
        # 3. Generate conversational answer
        final_answer = generate_natural_language_answer(req.question, db_results)
        
        return {
            "answer": final_answer,
            "executed_sql": sql_query,
            "raw_results": db_results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
