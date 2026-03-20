FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir django channels daphne requests

CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "django_chat.asgi:application"]