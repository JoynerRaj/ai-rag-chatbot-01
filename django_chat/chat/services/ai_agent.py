import os
import traceback
from google import genai
from google.genai import types
from chat.services.fastapi_client import FastAPIClient
from chat.semantic_cache import semantic_cache_get, semantic_cache_set
from chat.models import Document

# this is the tool definition that tells gemini it can search our documents
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

# words and phrases we treat as small talk - these don't need to be cached
CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
    "good afternoon", "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you",
    "great", "nice", "cool", "awesome", "got it", "understood"
}

def _is_casual_message(text: str) -> bool:
    """returns True if this looks like a greeting or small talk that shouldn't be cached"""
    cleaned = text.strip().lower().rstrip("!?.,:;")
    return cleaned in CASUAL_PHRASES or len(cleaned.split()) <= 2 and cleaned in CASUAL_PHRASES


class AIAgentService:
    @staticmethod
    def process_query(query: str, document_id: str, user, chat_id: int, chat_history: list = None) -> str:
        try:
            user_id = user.id if (user and user.is_authenticated) else None
            print(f"[{chat_id}] process_query: {query!r}  user_id={user_id}")

            # no documents = nothing to search, tell the user to upload something first
            if user and user.is_authenticated:
                has_docs = Document.objects.filter(user=user).exists()
            else:
                has_docs = Document.objects.exists()

            if not has_docs:
                return (
                    "No documents uploaded yet.\n\n"
                    "Please go to **Documents -> Upload** and add a document first. "
                    "I can only answer questions based on your uploaded knowledge base."
                )

            # skip cache check for greetings and small talk, not worth looking up
            if not _is_casual_message(query):
                cached = semantic_cache_get(query, document_id, user_id=user_id)
                if cached:
                    print(f"[{chat_id}] Semantic Cache HIT")
                    return cached

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options={"api_version": "v1alpha"}
            )

            if document_id and str(document_id).strip():
                system_instruction = (
                    "You are a helpful AI assistant. The user has selected a specific document to focus on. "
                    "You have access to the full conversation history of this session — use it to remember "
                    "anything the user has told you, such as their name, preferences, or prior questions. "
                    "When answering factual queries, you MUST call the 'search_documents' tool to retrieve context from that document. "
                    "If the answer is found in the document, answer based on the document. "
                    "If the user is just saying hello, asking about you, or making casual conversation, you can answer naturally without using the tool. "
                    "If the user asks a specific question not covered in the document, kindly inform them that it's not in the selected document, "
                    "but you may provide general helpful information if appropriate."
                )
            else:
                system_instruction = (
                    "You are a helpful AI assistant with access to an uploaded knowledge base. "
                    "You have access to the full conversation history of this session. "
                    "Always use the conversation history to remember personal details the user has shared, "
                    "such as their name, what they are working on, or anything they mentioned before. "
                    "If the user previously said their name is X and then asks 'what is my name', you MUST answer using that prior information. "
                    "When the user asks factual questions about documents, you MUST call the 'search_documents' tool to search the uploaded documents. "
                    "Answer based on what the tool returns when possible. "
                    "If the user is just greeting you, asking about your capabilities, or making casual conversation, respond naturally without searching. "
                    "If the documents do not contain the answer to a factual query, say: "
                    "'I couldn't find specific information about this in the uploaded documents, but here's what I know generally:' and provide a helpful answer."
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_doc_tool],
            )

            # put the previous messages in first so gemini knows what was discussed before
            contents = []
            if chat_history:
                for past in chat_history:
                    contents.append(types.Content(role="user", parts=[types.Part(text=past["question"])]))
                    contents.append(types.Content(role="model", parts=[types.Part(text=past["answer"])]))

            # then add what the user just asked
            contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

            MAX_ROUNDS = 3
            tool_was_used = False  # will flip to True if gemini actually calls search_documents

            for _ in range(MAX_ROUNDS):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )

                contents.append(response.candidates[0].content)

                fn_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "function_call") and part.function_call
                ]

                if not fn_calls:
                    break

                tool_response_parts = []
                for fn_call in fn_calls:
                    if fn_call.name == "search_documents":
                        args = dict(fn_call.args)
                        rag_query = args.get("query", query)
                        doc_id = args.get("document_id", document_id) or document_id

                        print(f"[{chat_id}] RAG tool called: query={rag_query!r}")
                        rag_result = FastAPIClient.search_documents(rag_query, doc_id)
                        print(f"[{chat_id}] RAG result: {len(rag_result)} chars")

                        tool_was_used = True  # gemini searched the docs, this answer is worth caching

                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name="search_documents",
                                response={"result": rag_result}
                            )
                        )

                contents.append(types.Content(role="tool", parts=tool_response_parts))

            answer = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, "text") and part.text])
            answer = answer.strip()

            print(f"[{chat_id}] Answer: {answer[:80]!r}")

            # only cache if we actually searched documents - no caching for hello/thanks type replies
            if tool_was_used and answer and user_id is not None:
                semantic_cache_set(query, answer, document_id, user_id=user_id)

            return answer if answer else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            print(f"[{chat_id}] ERROR: {e}")
            traceback.print_exc()
            error_msg = str(e)

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                return "The AI model is currently experiencing high demand. Spikes in demand are usually temporary. Please wait a few moments and try again."
            elif "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                return "The AI service quota has been exceeded. Please try again later."
            elif "400" in error_msg or "INVALID_ARGUMENT" in error_msg:
                return "There was an issue processing your request. Please try rephrasing your question."
            else:
                return "An unexpected error occurred while generating a response. Please try again later."
