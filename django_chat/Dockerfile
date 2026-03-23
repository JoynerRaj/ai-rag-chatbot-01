FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install django channels daphne pinecone sentence-transformers google-generativeai

CMD ["daphne", "-b", "0.0.0.0", "-p", "8001", "django_chat.asgi:application"]