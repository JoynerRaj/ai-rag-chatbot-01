from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    EMBEDDING_PENDING = "pending"
    EMBEDDING_DONE    = "done"
    EMBEDDING_FAILED  = "failed"
    EMBEDDING_STATUS_CHOICES = [
        (EMBEDDING_PENDING, "Pending"),
        (EMBEDDING_DONE,    "Done"),
        (EMBEDDING_FAILED,  "Failed"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=255)
    content = models.TextField()
    pinecone_id = models.CharField(max_length=100, blank=True, null=True)
    embedding_status = models.CharField(
        max_length=10,
        choices=EMBEDDING_STATUS_CHOICES,
        default=EMBEDDING_PENDING,
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class ChatHistory(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)