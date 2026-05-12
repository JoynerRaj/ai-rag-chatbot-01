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

            def get_generator():
                return AIAgentService.process_query(
                    query=query,
                    document_id=document_id,
                    user=self.user,
                    chat_id=self.chat_id,
                    chat_history=chat_history,
                    stream=True
                )

            generator = await sync_to_async(get_generator, thread_sensitive=False)()

            full_answer = ""
            if hasattr(generator, "__iter__") and not isinstance(generator, str):
                def get_next():
                    try:
                        return next(generator)
                    except StopIteration:
                        return None
                        
                while True:
                    chunk = await sync_to_async(get_next, thread_sensitive=False)()
                    if chunk is None:
                        break
                    full_answer += chunk
                    await self.send(json.dumps({
                        "type": "stream",
                        "chunk": chunk
                    }))
                
                await self.send(json.dumps({
                    "type": "stream_end",
                    "question": query,
                    "chat_id": self.chat_id
                }))
            else:
                full_answer = generator
                await self.send(json.dumps({
                    "type": "response",
                    "response": full_answer,
                    "question": query,
                    "chat_id": self.chat_id
                }))

            # save the exchange so history loads correctly when the chat is reopened
            if self.chat_id and full_answer:
                await sync_to_async(ChatHistory.objects.create)(
                    session_id=self.chat_id,
                    question=query,
                    answer=full_answer
                )
                session = await sync_to_async(ChatSession.objects.get)(id=self.chat_id)
                if session.title == "New Chat":
                    session.title = query[:30] + ("..." if len(query) > 30 else "")
                    await sync_to_async(session.save)()

                # ── MemPalace-inspired: extract & store memories async ──────
                # Extract key facts from the exchange in a background thread so
                # they are available to future queries via semantic memory search.
                if self.user and getattr(self.user, "is_authenticated", False):
                    # Snapshot values so the closure is safe even if self changes
                    _uid  = self.user.id
                    _sid  = self.chat_id
                    _q    = query
                    _ans  = full_answer

                    def _store_memory_inline():
                        """
                        Run Gemini extraction + Pinecone upsert directly in the
                        sync thread pool (via sync_to_async below) so we are
                        never killed as a daemon thread.  stream_end was already
                        sent to the browser, so this never blocks the UX.
                        """
                        try:
                            from chat.services.memory_service import (
                                _is_worth_remembering,
                                extract_memories,
                                _store_single_memory,
                            )
                            from chat.models import MemoryEntry

                            if not _is_worth_remembering(_q, _ans):
                                print(f"[memory] skipped trivial exchange user={_uid}")
                                return

                            facts = extract_memories(_q, _ans)
                            print(f"[memory] extracted {len(facts)} fact(s) for user={_uid}")

                            for fact in facts:
                                vid = _store_single_memory(
                                    user_id=_uid,
                                    session_id=_sid,
                                    fact=fact,
                                    question=_q,
                                )
                                if vid:
                                    MemoryEntry.objects.create(
                                        user_id=_uid,
                                        session_id=_sid,
                                        content=fact,
                                        source_question=_q[:500],
                                        pinecone_id=vid,
                                    )
                                    print(f"[memory] saved: {fact[:60]!r}")
                        except Exception as _e:
                            import traceback
                            print(f"[memory] storage error (non-critical): {_e}")
                            traceback.print_exc()

                    await sync_to_async(_store_memory_inline, thread_sensitive=False)()

        except Exception as e:
            print(f"[WebSocket] Unhandled error: {e}")
            await self.send(json.dumps({
                "response": "Something went wrong. Please try again."
            }))