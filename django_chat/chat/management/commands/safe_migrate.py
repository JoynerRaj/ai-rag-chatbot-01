"""
safe_migrate.py

A custom management command that fixes a common Render deployment issue:
Django's django_migrations table records chat app migrations as "applied",
but the actual PostgreSQL tables don't exist (e.g., after switching databases).

This command:
  1. Checks whether the core chat tables actually exist in the DB.
  2. If they are missing, removes the stale records from django_migrations
     so that the next `migrate` call will CREATE the tables from scratch.
  3. Runs `migrate` to bring the schema fully up to date.
"""

from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.db import connection, OperationalError, ProgrammingError


# Chat app tables that must exist for the app to work.
CHAT_TABLES = [
    "chat_chatsession",
    "chat_chathistory",
    "chat_document",
]


class Command(BaseCommand):
    help = "Safely migrate — auto-repairs stale migration state if chat tables are missing."

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("==> safe_migrate: checking database state..."))

        try:
            existing_tables = connection.introspection.table_names()
        except (OperationalError, ProgrammingError) as exc:
            self.stdout.write(self.style.WARNING(f"Could not list tables ({exc}). Proceeding with normal migrate."))
            call_command("migrate", verbosity=1)
            return

        missing = [t for t in CHAT_TABLES if t not in existing_tables]

        if missing:
            self.stdout.write(
                self.style.WARNING(
                    f"Chat tables missing from database: {missing}\n"
                    "Resetting stale migration records so they can be re-created..."
                )
            )
            try:
                with connection.cursor() as cursor:
                    # Only delete if django_migrations itself exists
                    if "django_migrations" in existing_tables:
                        cursor.execute("DELETE FROM django_migrations WHERE app = 'chat'")
                        self.stdout.write(self.style.SUCCESS("Cleared stale chat migration records."))
                    else:
                        self.stdout.write("django_migrations table not yet created — migrate will build it fresh.")
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f"Could not clear migration records: {exc}"))
        else:
            self.stdout.write(self.style.SUCCESS("All chat tables present — no repair needed."))

        self.stdout.write(self.style.MIGRATE_HEADING("==> Running migrate..."))
        call_command("migrate", verbosity=1)
        self.stdout.write(self.style.SUCCESS("==> safe_migrate complete!"))
