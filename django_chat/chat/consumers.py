import json
import os
import asyncio
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import google.generativeai as genai
from .models import ChatHistory, ChatSession

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        print("✅ WebSocket CONNECTED")
        await self.accept()

        query_string = self.scope["query_string"].decode()

        chat_id = None
        if "chat_id=" in query_string:
            value = query_string.split("chat_id=")[-1]
            if value and value != "null":
                chat_id = value

        self.chat_id = int(chat_id) if chat_id else None

        # Load chat history
        if self.chat_id:
            history = await sync_to_async(list)(
                ChatHistory.objects.filter(session_id=self.chat_id).order_by("created_at")
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
                await self.send(json.dumps({
                    "response": "Please enter a valid question."
                }))
                return

            # 🔥 IMPORTANT: Run heavy AI work in background thread using sync_to_async
            from asgiref.sync import sync_to_async
            answer = await sync_to_async(self.process_query, thread_sensitive=False)(query, document_id)

            # Save history
            if self.chat_id:
                await sync_to_async(ChatHistory.objects.create)(
                    session_id=self.chat_id,
                    question=query,
                    answer=answer
                )

                session = await sync_to_async(ChatSession.objects.get)(id=self.chat_id)

                if session.title == "New Chat":
                    new_title = query[:30] + ("..." if len(query) > 30 else "")
                    session.title = new_title
                    await sync_to_async(session.save)()

            # Send response
            await self.send(json.dumps({
                "response": answer,
                "question": query,
                "chat_id": self.chat_id
            }))

        except Exception as e:
            await self.send(json.dumps({
                "response": "Error: " + str(e),
            }))

    # 🔥 BACKGROUND FUNCTION (NO ASYNC HERE)
    def process_query(self, query, document_id):
        import traceback
        try:
            print(f"[{self.chat_id}] Starting process_query for: {query}")
            
            # 🔍 Ask FastAPI to handle Heavy Pinecone/PyTorch Embedding
            print(f"[{self.chat_id}] Asking FastAPI for context embeddings...")
            import requests
            fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
            search_api = fastapi_url.replace("/upload", "") + "/search"
            
            res = requests.post(
                search_api,
                json={"query": query, "document_id": document_id},
                timeout=30  # Don't block forever if FastAPI is still loading up
            )
            res.raise_for_status()
            results = res.json()
            
            print(f"[{self.chat_id}] FastAPI returned {len(results) if results else 0} results.")

            if not results:
                return "This information is not available in the selected document."

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

            # 🤖 Gemini
            print(f"[{self.chat_id}] Calling Gemini API...")
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            print(f"[{self.chat_id}] Gemini responded successfully.")

            answer = response.text.strip()

            if "not available" in answer.lower():
                return "This information is not available in the selected document."

            return answer

        except Exception as e:
            print(f"[{self.chat_id}] ERROR in process_query: {e}")
            traceback.print_exc()
            return "Error: " + str(e)