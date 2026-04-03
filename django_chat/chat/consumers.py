import json
import os
import requests
from dotenv import load_dotenv
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from google import genai
from google.genai import types
from .models import ChatHistory, ChatSession
from .redis_client import redis_client  # Cache-Aside layer

load_dotenv()

# ── RAG Tool Definition ──────────────────────────────────────────────────────
# Gemini will autonomously decide WHEN to call this tool based on the user's message.
# For greetings/casual chat → it won't call it.
# For document-specific questions → it will call it to fetch context.
search_doc_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the uploaded knowledge base documents for information relevant "
                "to the user's question. Use this when the user asks a specific factual "
                "question that may be answered by the uploaded documents. "
                "Do NOT use this for greetings, thanks, or casual conversation."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query to look up in the documents"
                    ),
                    "document_id": types.Schema(
                        type=types.Type.STRING,
                        description="Optional UUID of a specific document to search within. Leave empty to search all."
                    )
                },
                required=["query"]
            )
        )
    ]
)


def call_rag_search(query: str, document_id: str = None) -> str:
    """Calls FastAPI /search and returns relevant context text."""
    try:
        fastapi_url = os.environ.get("FASTAPI_URL", "http://fastapi:8000/upload")
        search_api = fastapi_url.replace("/upload", "") + "/search"

        payload = {"query": query, "top_k": 5}
        if document_id and str(document_id).strip():
            payload["document_id"] = document_id

        res = requests.post(search_api, json=payload, timeout=30)
        if not res.ok:
            return f"[RAG search failed: {res.text}]"

        results = res.json()
        if not results:
            return "No relevant information found in the uploaded documents."

        chunks = "\n---\n".join([item["text"] for item in results])
        return f"Relevant document excerpts:\n{chunks}"
    except Exception as e:
        return f"[RAG search error: {str(e)}]"


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

            # Run AI in background thread
            answer = await sync_to_async(self.process_query, thread_sensitive=False)(query, document_id)

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

    # ── Agentic RAG Loop (background thread, NO async) ────────────────────────
    def process_query(self, query: str, document_id: str):
        import traceback
        try:
            print(f"[{self.chat_id}] process_query: {query!r}")

            # ── Cache-Aside: READ ────────────────────────────────────────────
            normalized_query = query.lower().strip()
            cache_key = f"chat:{normalized_query}:{document_id}"

            if redis_client is not None:
                try:
                    cached = redis_client.get(cache_key)
                    if cached:
                        print(f"[{self.chat_id}] ✅ Cache HIT for key: {cache_key!r}")
                        return cached
                except Exception as redis_err:
                    print(f"[{self.chat_id}] ⚠️  Redis read error (skipping cache): {redis_err}")
            # ── End Cache READ ───────────────────────────────────────────────

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            if document_id and str(document_id).strip():
                system_instruction = (
                    "You are a helpful AI assistant. The user has selected a SPECIFIC document to search. "
                    "You MUST call the 'search_documents' tool for EVERY question to retrieve context from that document. "
                    "Answer ONLY based on the retrieved document context. "
                    "If the answer is not found in the document, say: 'This information is not available in the selected document.' "
                    "Do NOT use your general knowledge — only use what the tool returns."
                )
            else:
                system_instruction = (
                    "You are a helpful, friendly AI assistant. "
                    "You have access to a tool called 'search_documents' that lets you search uploaded documents. "
                    "When the user asks a specific factual question that might be in the documents, "
                    "call the tool to retrieve relevant context, then answer based on that context. "
                    "For greetings, casual chat, or general knowledge questions, respond naturally without using the tool."
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_doc_tool],
            )

            contents = [types.Content(role="user", parts=[types.Part(text=query)])]

            # ── Agentic loop: Gemini decides when to call the RAG tool ────────
            MAX_ROUNDS = 3
            for round_num in range(MAX_ROUNDS):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )

                # Append model response to conversation
                contents.append(response.candidates[0].content)

                # Check for function calls
                fn_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "function_call") and part.function_call
                ]

                if not fn_calls:
                    break  # No tool calls → final answer ready

                # Execute each tool call
                tool_response_parts = []
                for fn_call in fn_calls:
                    if fn_call.name == "search_documents":
                        args = dict(fn_call.args)
                        rag_query = args.get("query", query)
                        doc_id = args.get("document_id", document_id) or document_id

                        print(f"[{self.chat_id}] 🔍 RAG tool called: query={rag_query!r}")
                        rag_result = call_rag_search(rag_query, doc_id)
                        print(f"[{self.chat_id}] 📄 RAG result: {len(rag_result)} chars")

                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name="search_documents",
                                response={"result": rag_result}
                            )
                        )

                # Feed tool results back
                contents.append(types.Content(role="tool", parts=tool_response_parts))

            # Extract final text
            answer = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    answer += part.text

            answer = answer.strip()
            print(f"[{self.chat_id}] ✅ Answer: {answer[:80]!r}")

            # ── Cache-Aside: WRITE ───────────────────────────────────────────
            if answer and redis_client is not None:
                try:
                    redis_client.set(cache_key, answer, ex=3600)  # TTL: 1 hour
                    print(f"[{self.chat_id}] 💾 Stored in Redis: key={cache_key!r}")
                except Exception as redis_err:
                    print(f"[{self.chat_id}] ⚠️  Redis write error (skipping store): {redis_err}")
            # ── End Cache WRITE ──────────────────────────────────────────────

            return answer if answer else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            print(f"[{self.chat_id}] ERROR: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"