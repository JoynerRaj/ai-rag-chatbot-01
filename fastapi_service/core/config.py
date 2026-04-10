import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "rag-index"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    EMBEDDING_DIMENSION = 384
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

settings = Settings()
