"""
main.py

FastAPI application entry point.

On startup:
  - The SQLite database is initialised (table created if it doesn't exist yet).
  - The HuggingFace sentence-transformer model is preloaded in the background
    so the first real request doesn't have to wait for it to download.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router as document_router
from api.audio_routes import router as audio_router
from services.embedding import EmbeddingService
from audio.db import init_db

# Load .env from the project root (one level above this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI(
    title="RAG Chatbot — Audio Service",
    description="Handles audio event detection and the document embedding pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router)
app.include_router(audio_router)


@app.on_event("startup")
async def on_startup():
    init_db()
    EmbeddingService.preload_model_background()


@app.head("/")
@app.get("/")
def health_check():
    return {"status": "healthy"}