from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0006_alter_document_embedding_status"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="MemoryEntry",
            fields=[
                ("id",              models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("content",         models.TextField(help_text="The extracted memory / fact sentence.")),
                ("source_question", models.TextField(blank=True, help_text="Original question that triggered this memory.")),
                ("memory_type",     models.CharField(
                    choices=[("episodic", "Episodic"), ("fact", "Fact / Preference")],
                    default="fact",
                    max_length=20,
                )),
                ("pinecone_id",     models.CharField(blank=True, max_length=100, null=True)),
                ("created_at",      models.DateTimeField(auto_now_add=True)),
                ("session",         models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name="memories",
                    to="chat.chatsession",
                )),
                ("user",            models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name="memories",
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                "verbose_name_plural": "Memory Entries",
                "ordering": ["-created_at"],
            },
        ),
    ]
