import json
import requests
import os
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)

            query = data.get("message")
            document_id = data.get("document_id")

            response = await sync_to_async(requests.post)(
                "https://ai-rag-chatbot-01.onrender.com/query",
                json={
                    "query": query,
                    "document_id": document_id
                },
                timeout=10
            )

            if response.status_code != 200:
                await self.send(text_data=json.dumps({
                    "response": "FastAPI error",
                    "sources": []
                }))
                return

            try:
                data_json = response.json()
            except:
                await self.send(text_data=json.dumps({
                    "response": "Invalid response from FastAPI",
                    "sources": []
                }))
                return

            results = data_json.get("results", [])

            if not results:
                await self.send(text_data=json.dumps({
                    "response": "No relevant data found. Please upload a document first.",
                    "sources": []
                }))
                return

            top_chunks = results[:3]

            context = "\n".join(top_chunks)

            prompt = f"""
Answer ONLY from the context below.
If answer is not in context, say "Not found in document".

Context:
{context}

Question:
{query}
"""

            model = genai.GenerativeModel("gemini-2.5-flash")

            gemini_response = await sync_to_async(
                model.generate_content
            )(prompt)

            await self.send(text_data=json.dumps({
                "response": gemini_response.text,
                "sources": top_chunks[:2]
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                "response": "Error: " + str(e),
                "sources": []
            }))