import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import ChatHistory, ChatSession
from chat.services.ai_agent import AIAgentService

class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        print("WebSocket connected")
        await self.accept()

        self.user = self.scope.get("user", None)

        # pull chat_id from the query string so we know which session this is
        query_string = self.scope["query_string"].decode()
        chat_id = None
        if "chat_id=" in query_string:
            value = query_string.split("chat_id=")[-1]
            if value and value != "null":
                chat_id = value

        self.chat_id = int(chat_id) if chat_id else None

        # if opening an existing chat, send the message history down so the UI can render it
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

            # just a keepalive ping from the frontend, nothing to do
            if data.get("type") == "ping":
                return

            query = data.get("message", "").strip()
            document_id = data.get("document_id")

            if not query:
                await self.send(json.dumps({"response": "Please enter a valid question."}))
                return

            # grab the last 10 messages for context - newest first, then flip back to oldest-first
            chat_history = []
            if self.chat_id:
                history_qs = await sync_to_async(list)(
                    ChatHistory.objects.filter(session_id=self.chat_id).order_by("-created_at")[:10]
                )
                history_qs = list(reversed(history_qs))
                chat_history = [{"question": h.question, "answer": h.answer} for h in history_qs]

            answer = await sync_to_async(AIAgentService.process_query, thread_sensitive=False)(
                query=query,
                document_id=document_id,
                user=self.user,
                chat_id=self.chat_id,
                chat_history=chat_history
            )

            # save the exchange so history loads correctly when the chat is reopened
            if self.chat_id:
                await sync_to_async(ChatHistory.objects.create)(
                    session_id=self.chat_id,
                    question=query,
                    answer=answer
                )
                session = await sync_to_async(ChatSession.objects.get)(id=self.chat_id)
                if session.title == "New Chat":
                    session.title = query[:30] + ("..." if len(query) > 30 else "")
                    await sync_to_async(session.save)()

            await self.send(json.dumps({
                "response": answer,
                "question": query,
                "chat_id": self.chat_id
            }))

        except Exception as e:
            print(f"[WebSocket] Unhandled error: {e}")
            await self.send(json.dumps({
                "response": "Something went wrong. Please try again."
            }))