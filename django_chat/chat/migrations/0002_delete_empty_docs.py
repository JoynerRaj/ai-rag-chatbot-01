from django.db import migrations


def delete_empty_content_docs(apps, schema_editor):
    """
    One-time cleanup: remove all documents with empty content.
    These were uploaded before the content-extraction fix was deployed.
    Users must re-upload them — they will now capture content correctly.
    """
    Document = apps.get_model('chat', 'Document')
    empty_docs = Document.objects.filter(content='')
    count = empty_docs.count()
    empty_docs.delete()
    print(f"[migration] Deleted {count} document(s) with empty content.")


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(delete_empty_content_docs, migrations.RunPython.noop),
    ]
