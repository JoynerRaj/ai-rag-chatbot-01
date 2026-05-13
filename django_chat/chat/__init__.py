# This file marks the chat app as a Python package.
# Each module here handles one specific concern:
#
#   models.py        — database models (ChatSession, ChatHistory, Document, MemoryEntry)
#   views.py         — HTTP request handlers
#   consumers.py     — WebSocket handler for real-time chat
#   urls.py          — URL routing
#   admin.py         — Django admin registration
#   semantic_cache.py — Pinecone-based semantic response caching
#   redis_client.py  — Redis connection helper
