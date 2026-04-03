import os
import redis
from dotenv import load_dotenv

load_dotenv()

_REDIS_URL  = os.environ.get("REDIS_URL", "")
_REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))


def _build_client():
    try:
        if _REDIS_URL:
            client = redis.from_url(_REDIS_URL, decode_responses=True)
        else:
            client = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)

        # Verify the connection is actually alive
        client.ping()
        print(f"[Redis] ✅ Connected – {'URL' if _REDIS_URL else f'{_REDIS_HOST}:{_REDIS_PORT}'}")
        return client

    except Exception as e:
        print(f"[Redis] ⚠️  Could not connect ({e}). Caching disabled – falling back to normal flow.")
        return None


# Module-level singleton; None means Redis is unavailable → callers skip cache.
redis_client = _build_client()
