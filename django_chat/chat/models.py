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


class MemoryEntry(models.Model):
    """
    Mirrors a single Pinecone memory vector in the relational DB.

    Inspired by MemPalace's "drawer" concept — each row is one verbatim
    fact extracted from a conversation turn and stored for future recall.
    The pinecone_id links back to the actual embedding vector so we can
    delete it on demand.
    """

    MEMORY_TYPE_EPISODIC  = "episodic"   # something that happened
    MEMORY_TYPE_FACT      = "fact"       # a stated fact or preference
    MEMORY_TYPE_CHOICES   = [
        (MEMORY_TYPE_EPISODIC, "Episodic"),
        (MEMORY_TYPE_FACT,     "Fact / Preference"),
    ]

    user            = models.ForeignKey(User, on_delete=models.CASCADE, related_name="memories")
    session         = models.ForeignKey(
        ChatSession, on_delete=models.SET_NULL, null=True, blank=True, related_name="memories"
    )
    content         = models.TextField(help_text="The extracted memory / fact sentence.")
    source_question = models.TextField(blank=True, help_text="Original question that triggered this memory.")
    memory_type     = models.CharField(
        max_length=20,
        choices=MEMORY_TYPE_CHOICES,
        default=MEMORY_TYPE_FACT,
    )
    pinecone_id     = models.CharField(max_length=100, blank=True, null=True)
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Memory Entries"

    def __str__(self):
        return self.content[:60]