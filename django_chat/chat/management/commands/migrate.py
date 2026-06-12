"""
migrate.py  — overrides Django's built-in migrate command

WHY THIS EXISTS
---------------
On Render's free tier there is no shell access, so we cannot manually run
`manage.py migrate chat zero --fake && manage.py migrate` to repair a broken
migration state.

The broken state happens when:
  - Django's django_migrations table already contains records for the chat app
    (i.e. Django thinks migrations are applied), BUT
  - The actual PostgreSQL tables (chat_chatsession, chat_chathistory, etc.)
    do not exist — e.g. after switching from SQLite to a fresh PostgreSQL DB.

In that situation the normal `migrate` command prints:
    Apply all migrations: account, admin, auth, contenttypes, sessions, sites
    No migrations to apply.
…and 'chat' never appears, so tables are never created.

THIS COMMAND detects that inconsistency, removes the stale chat rows from
django_migrations, and then calls the real migrate so all chat tables are
created from scratch.
"""

from django.core.management.commands.migrate import Command as MigrateCommand
from django.db import connection, OperationalError, ProgrammingError


# Tables that must exist for the chat app to work correctly.
_REQUIRED_CHAT_TABLES = [
    "chat_chatsession",
    "chat_chathistory",
    "chat_document",
]


class Command(MigrateCommand):
    """
    A drop-in replacement for `manage.py migrate` that auto-repairs a
    corrupted chat-migration state before delegating to the real command.
    """

    def handle(self, *args, **options):
        self._repair_if_needed()
        super().handle(*args, **options)

    # ------------------------------------------------------------------
    # Repair logic
    # ------------------------------------------------------------------

    def _repair_if_needed(self):
        try:
            existing = set(connection.introspection.table_names())
        except (OperationalError, ProgrammingError) as exc:
            # Can't introspect yet (DB unreachable / no tables) — let the
            # real migrate handle it from scratch.
            self.stdout.write(
                self.style.WARNING(f"[migrate] DB not ready for introspection ({exc}); skipping repair.")
            )
            return

        missing = [t for t in _REQUIRED_CHAT_TABLES if t not in existing]
        if not missing:
            # All tables present — nothing to repair.
            return

        self.stdout.write(
            self.style.WARNING(
                f"[migrate] Chat tables missing: {missing}\n"
                "[migrate] Clearing stale chat migration records so they can be re-created..."
            )
        )

        if "django_migrations" not in existing:
            # Completely fresh database — let the real migrate do everything.
            self.stdout.write("[migrate] django_migrations not yet created; skipping cleanup.")
            return

        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM django_migrations WHERE app = 'chat'")
            self.stdout.write(
                self.style.SUCCESS("[migrate] Stale chat migration records removed. Will now apply them.")
            )
        except Exception as exc:
            self.stdout.write(
                self.style.ERROR(f"[migrate] Could not clear stale records: {exc}")
            )
