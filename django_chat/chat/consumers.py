import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import ChatHistory, ChatSession
from chat.services.ai_agent import AIAgentService

class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        print("✅ WebSocket CONNECTED")
        await self.accept()

        self.user = self.scope.get("user", None)

        query_string = self.scope["query_string"].decode()
        chat_id = None
        if "chat_id=" in query_string:
            value = query_string.split("chat_id=")[-1]
            if value and value != "null":
                chat_id = value

        self.chat_id = int(chat_id) if chat_id else None

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
                await self.send(json.dumps({"response": "Please enter a valid question."}))
                return

            # fetch the last 10 messages from this session so the model has recent context
            # we order descending first to get the most recent ones, then flip them back to
            # chronological order before passing them to the AI
            chat_history = []
            if self.chat_id:
                history_qs = await sync_to_async(list)(
                    ChatHistory.objects.filter(session_id=self.chat_id).order_by("-created_at")[:10]
                )
                # reverse so the oldest message in the window comes first (chronological)
                history_qs = list(reversed(history_qs))
                chat_history = [{"question": h.question, "answer": h.answer} for h in history_qs]

            # Run AI in background thread
            answer = await sync_to_async(AIAgentService.process_query, thread_sensitive=False)(
                query=query,
                document_id=document_id,
                user=self.user,
                chat_id=self.chat_id,
                chat_history=chat_history
            )

            # Save to DB
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
            await self.send(json.dumps({"response": "Error: " + str(e)}))