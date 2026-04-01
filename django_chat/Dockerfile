FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "python manage.py migrate --run-syncdb && daphne -b 0.0.0.0 -p 8001 django_chat.asgi:application"]