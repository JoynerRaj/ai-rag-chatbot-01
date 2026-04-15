import os
import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat'

    def ready(self):
        # Only run the scheduler in the main process (not in migrate, shell, etc.)
        # RUN_MAIN is set by Django's auto-reloader; on Render there's no reloader
        # so we use an env-var guard to avoid double-start locally.
        if os.environ.get("SCHEDULER_STARTED"):
            return
        os.environ["SCHEDULER_STARTED"] = "1"

        self._start_keep_alive_scheduler()

    @staticmethod
    def _start_keep_alive_scheduler():
        """Pings Django health + FastAPI root every 10 min so Render never sleeps them."""
        try:
            import requests as req
            from apscheduler.schedulers.background import BackgroundScheduler

            django_url  = os.environ.get("DJANGO_URL",  "https://django-rag.onrender.com/health/")
            fastapi_url = os.environ.get("FASTAPI_URL", "https://ai-rag-chatbot-01.onrender.com/upload")
            # strip /upload from FASTAPI_URL to get the root ping endpoint
            fastapi_ping = fastapi_url.replace("/upload", "/")

            def ping_services():
                for name, url in [("Django", django_url), ("FastAPI", fastapi_ping)]:
                    try:
                        r = req.get(url, timeout=10)
                        logger.info(f"[keep-alive] {name} → {r.status_code}")
                    except Exception as e:
                        logger.warning(f"[keep-alive] {name} ping failed: {e}")

            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(ping_services, "interval", minutes=10, id="keep_alive")
            scheduler.start()
            logger.info("[keep-alive] scheduler started — pinging every 10 min")

        except Exception as e:
            logger.error(f"[keep-alive] failed to start scheduler: {e}")
