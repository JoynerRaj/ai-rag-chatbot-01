from django.db import migrations


def nuke_all_documents(apps, schema_editor):
    """
    Hard reset: delete every Document record.
    Documents uploaded before the content-extraction fix had blank/whitespace
    content that slipped through the previous migration's filter.
    Users must re-upload — new uploads capture content correctly on Django side.
    """
    Document = apps.get_model('chat', 'Document')
    count = Document.objects.count()
    Document.objects.all().delete()
    print(f"[migration 0003] Wiped {count} document(s) for clean slate.")


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0002_delete_empty_docs'),
    ]

    operations = [
        migrations.RunPython(nuke_all_documents, migrations.RunPython.noop),
    ]
