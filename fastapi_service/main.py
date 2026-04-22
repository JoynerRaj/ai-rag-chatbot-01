from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as main_router
from api.audio_endpoints import router as audio_router
from services.embedding import EmbeddingService
from audio.db import init_db
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI(title="RAG Chatbot Service", description="FastAPI microservice for embedding and retrieval")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
app.include_router(audio_router)

@app.on_event("startup")
async def startup_event():
    init_db()
    EmbeddingService.preload_model_background()