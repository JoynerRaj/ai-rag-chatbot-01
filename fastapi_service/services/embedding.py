import os
import torch
from sentence_transformers import SentenceTransformer
from core.config import settings

# Safely import PyTorch on Main Thread to prevent C++ Extension native crashes in Background threads
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._model

    @classmethod
    def embed_text(cls, text: str) -> list[float]:
        model = cls.get_model()
        return model.encode(text).tolist()

    @classmethod
    def preload_model_background(cls):
        import threading
        def preload():
            print("Pre-loading HuggingFace weights in background thread...")
            try:
                cls.get_model()
                print("PyTorch Model Initialized Successfully!")
            except Exception as e:
                print(f"Failed to preload model: {e}")
                
        thread = threading.Thread(target=preload, daemon=True)
        thread.start()
