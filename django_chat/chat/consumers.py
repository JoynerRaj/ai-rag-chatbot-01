import json
import os
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import google.generativeai as genai
from .models import ChatHistory, ChatSession
from .pinecone_utils import query_pinecone

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

        query_string = self.scope["query_string"].decode()

        chat_id = None
        if "chat_id=" in query_string:
            value = query_string.split("chat_id=")[-1]
            if value and value != "null":
                chat_id = value

        self.chat_id = int(chat_id) if chat_id else None

        if self.chat_id:
            history = await sync_to_async(list)(
                ChatHistory.objects.filter(session_id=int(self.chat_id)).order_by("created_at")
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

            if data.get("type") == "ping":
                return

            query = data.get("message", "").strip()
            document_id = data.get("document_id")

            if not query:
                await self.send(text_data=json.dumps({
                    "response": "Please enter a valid question.",
                }))
                return

            results = await sync_to_async(query_pinecone)(query, document_id)

            if not results:
                await self.send(text_data=json.dumps({
                    "response": "This information is not available in the selected document.",
                }))
                return

            context = "\n".join([item["text"] for item in results])

            prompt = f"""
You MUST answer ONLY from the given context.
DO NOT use any outside knowledge.

If the answer is not clearly present in the context,
reply EXACTLY with:
"This information is not available in the selected document."

Context:
{context}

Question:
{query}
"""

            model = genai.GenerativeModel("gemini-2.5-flash")

            gemini_response = await sync_to_async(
                model.generate_content
            )(prompt)

            answer = gemini_response.text.strip()

            if "not available" in answer.lower():
                answer = "This information is not available in the selected document."

            if self.chat_id:
                await sync_to_async(ChatHistory.objects.create)(
                    session_id=int(self.chat_id),
                    question=query,
                     answer=answer
                )
                session = await sync_to_async(ChatSession.objects.get)(id=self.chat_id)

                if session.title == "New Chat":
                    new_title = query[:30] + ("..." if len(query) > 30 else "")
                    session.title = new_title
                    await sync_to_async(session.save)()

            await self.send(text_data=json.dumps({
                "response": answer,
                "question": query,
                "chat_id": self.chat_id
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                "response": "Error: " + str(e),
            }))