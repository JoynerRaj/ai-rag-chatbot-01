from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    pinecone_id = models.CharField(max_length=100, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class ChatHistory(models.Model):
    session_key = models.CharField(max_length=100)
    question = models.TextField()
    answer = models.TextField()
    sources = models.CharField(max_length=500, blank=True)
    asked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_key} - {self.asked_at}"