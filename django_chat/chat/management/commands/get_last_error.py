from django.core.management.base import BaseCommand
from chat.models import Document

class Command(BaseCommand):
    help = 'Print the latest embedding error'

    def handle(self, *args, **options):
        failed_docs = Document.objects.filter(embedding_status=Document.EMBEDDING_FAILED).order_by('-created_at')
        if not failed_docs.exists():
            self.stdout.write(self.style.WARNING('No failed documents found.'))
            return

        doc = failed_docs.first()
        self.stdout.write(self.style.SUCCESS(f'Latest failed document: {doc.id} - {doc.title}'))
        self.stdout.write('--- Error Trace ---')
        self.stdout.write(str(doc.content))
        self.stdout.write('-------------------')
