import json
import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import google.generativeai as genai
from pinecone import Pinecone
from .models import ChatHistory

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pinecone setup directly in Django
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")


def embed(text):
    """Same hash-based embedding as FastAPI — must match exactly."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(384)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def query_pinecone(query, document_id=None):
    """Query Pinecone directly — no FastAPI call needed."""
    filter = None
    if document_id and document_id.strip():
        filter = {"document_id": {"$eq": document_id}}

    results = index.query(
        vector=embed(query),
        top_k=3,
        include_metadata=True,
        filter=filter
    )

    texts = []
    for match in results["matches"]:
        texts.append({
            "text": match["metadata"].get("text", ""),
            "file_name": match["metadata"].get("file_name", "Unknown")
        })

    return texts


class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        session = self.scope["session"]

        if not session.session_key:
            await sync_to_async(session.save)()

        self.session_key = session.session_key
        await self.accept()

        # Send past chat history on connect
        history = await sync_to_async(list)(
            ChatHistory.objects.filter(
                session_key=self.session_key
            ).order_by("asked_at")
        )

        for item in history:
            await self.send(text_data=json.dumps({
                "type": "history",
                "question": item.question,
                "response": item.answer,
            }))

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)

            # Ignore ping
            if data.get("type") == "ping":
                return

            query = data.get("message")
            document_id = data.get("document_id")

            # Query Pinecone directly — no FastAPI call
            results = await sync_to_async(query_pinecone)(query, document_id)

            if not results:
                await self.send(text_data=json.dumps({
                    "response": "No relevant data found in the selected document.",
                }))
                return

            context = "\n".join([item["text"] for item in results])

            prompt = f"""
Answer ONLY from the context below.
If the answer is not found in the context, say "This information is not available in the selected document."

Context:
{context}

Question:
{query}
"""

            model = genai.GenerativeModel("gemini-2.5-flash")
            gemini_response = await sync_to_async(model.generate_content)(prompt)
            answer = gemini_response.text

            # Save to Django DB
            await sync_to_async(ChatHistory.objects.create)(
                session_key=self.session_key,
                question=query,
                answer=answer,
                sources=""
            )

            await self.send(text_data=json.dumps({
                "response": answer,
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                "response": "Error: " + str(e),
            }))