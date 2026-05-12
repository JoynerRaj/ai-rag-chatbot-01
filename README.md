# AI RAG Chatbot

A full-stack chatbot that lets users upload documents and ask questions about them.
Answers are grounded in uploaded content using a Retrieval-Augmented Generation (RAG) pipeline.

## How it works

1. A user uploads a PDF, DOCX, or TXT file (or an audio file).
2. The Django backend extracts the text, chunks it, and stores vector embeddings in Pinecone.
3. When the user asks a question, the most relevant chunks are retrieved from Pinecone
   and sent to Gemini along with the question.
4. Frequently-asked questions are cached in Redis so repeated queries skip the embedding step.
5. Audio files are transcribed by Gemini before being embedded, so you can ask questions
   about spoken content too.

## Project layout

```
ai-rag-chatbot/
├── django_chat/        Main web application (Django + Channels + Daphne)
│   ├── chat/           App — models, views, WebSocket consumer, services
│   └── django_chat/    Project settings, URL config, ASGI entry point
│
└── fastapi_service/    Audio event microservice (FastAPI)
    ├── api/            HTTP route handlers
    ├── audio/          Event detection, SQL generation, answer formatting
    ├── core/           App config (env vars)
    ├── schemas/        Pydantic request/response models
    └── services/       Embedding, document parsing, Pinecone wrapper
```

## Quickstart (local)

### Prerequisites
- Python 3.11+
- Redis running on `localhost:6379`
- A `.env` file at the project root (copy `.env.example` and fill in the blanks)

### Run the Django app

```bash
cd django_chat
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

For WebSocket support (needed for streaming chat):

```bash
daphne -b 0.0.0.0 -p 8000 django_chat.asgi:application
```

### Run the FastAPI service (optional — for audio event queries)

```bash
cd fastapi_service
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### Run everything with Docker Compose

```bash
docker-compose up --build
```

## Environment variables

| Variable             | Required | Description                                        |
|----------------------|----------|----------------------------------------------------|
| `GEMINI_API_KEY`     | Yes      | Google AI Studio API key                           |
| `PINECONE_API_KEY`   | Yes      | Pinecone API key                                   |
| `PINECONE_INDEX_NAME`| No       | Defaults to `rag-index`                            |
| `PINECONE_CLOUD`     | No       | Defaults to `aws`                                  |
| `PINECONE_REGION`    | No       | Defaults to `us-east-1`                            |
| `SECRET_KEY`         | Yes      | Django secret key (keep this private)              |
| `DATABASE_URL`       | No       | Postgres URL — falls back to SQLite if not set     |
| `REDIS_URL`          | No       | Full Redis URL — falls back to `localhost:6379`    |

## Deployment (Render)

The `render.yaml` file configures the Django service.  
Set all required environment variables in the Render dashboard before deploying.
