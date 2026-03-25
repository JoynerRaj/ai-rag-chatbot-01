import json
import requests
import os
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import google.generativeai as genai
from .models import ChatHistory

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        session = self.scope["session"]

        if not session.session_key:
            await sync_to_async(session.save)()

        self.session_key = session.session_key
        await self.accept()

        # Load and send past chat history
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

            response = await sync_to_async(requests.post)(
                "https://ai-rag-chatbot-01.onrender.com/query",
                json={"query": query, "document_id": document_id},
                timeout=30
            )

            if response.status_code != 200:
                await self.send(text_data=json.dumps({
                    "response": "FastAPI error: " + response.text,
                }))
                return

            try:
                data_json = response.json()
            except Exception:
                await self.send(text_data=json.dumps({
                    "response": "Invalid response from FastAPI",
                }))
                return

            results = data_json.get("results", [])

            if not results:
                await self.send(text_data=json.dumps({
                    "response": "No relevant data found in the selected document.",
                }))
                return

            top_chunks = results[:3]
            context = "\n".join([item["text"] for item in top_chunks])

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

            # Save to DB
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