import json
import requests
import os
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            query = data['message']

            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"query": query}
            )

            results = response.json().get("results", [])

            if not results:
                await self.send(text_data=json.dumps({
                    "response": "No relevant data found. Please upload a document first."
                }))
                return

            context = "\n".join(results[:2])

            prompt = f"""
Answer ONLY from the context below.
If answer is not in context, say "Not found in document".

Context:
{context}

Question:
{query}
"""

            gemini_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            await self.send(text_data=json.dumps({
                "response": gemini_response.text
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                "response": "Error: " + str(e)
            }))