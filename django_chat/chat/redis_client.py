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
            # on Render we get a full URL from the env
            client = redis.from_url(_REDIS_URL, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
        else:
            # local dev - just use host/port directly
            client = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)

        client.ping()
        print(f"[Redis] Connected - {'URL' if _REDIS_URL else f'{_REDIS_HOST}:{_REDIS_PORT}'}")
        return client

    except Exception as e:
        # if redis isn't running, don't crash the whole app - just skip caching
        print(f"[Redis] Could not connect ({e}). Caching disabled.")
        return None


# one shared connection for the whole app - if None, caching is quietly skipped
redis_client = _build_client()
