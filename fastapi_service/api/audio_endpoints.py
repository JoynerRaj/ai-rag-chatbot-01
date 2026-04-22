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
    """
    try:
        content = await file.read()
        
        # In a real application, check file extension or mime type here.
        if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
            raise HTTPException(status_code=400, detail="Unsupported audio format.")

        # Simulate event extraction
        events = process_audio(content)
        
        # Save to SQLite
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
