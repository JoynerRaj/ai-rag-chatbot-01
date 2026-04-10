from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as main_router
from services.embedding import EmbeddingService

app = FastAPI(title="RAG Chatbot Service", description="FastAPI microservice for embedding and retrieval")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)

@app.on_event("startup")
async def startup_event():
    EmbeddingService.preload_model_background()