import os
import sys
from pathlib import Path
from datetime import timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

# Keep the secret key in an environment variable — never hardcode it here.
# For local dev you can put it in your .env file.
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "change-me-before-deploying-to-production",
)

DEBUG = os.environ.get("DEBUG", "False") == "True"

ALLOWED_HOSTS = ["django-rag.onrender.com", "*"]

CSRF_TRUSTED_ORIGINS = [
    "https://django-rag.onrender.com",
]

# -----------------------------------------------------------------------
# Installed apps
# -----------------------------------------------------------------------
INSTALLED_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",

    # django-allauth (handles login / signup / Google OAuth)
    "allauth",
    "allauth.account",

    # DRF + JWT — used for the /api/auth/ endpoints
    "rest_framework",
    "rest_framework_simplejwt",

    # Django Channels — WebSocket support for streaming chat
    "channels",

    "chat",
]

# -----------------------------------------------------------------------
# ASGI / Channels
# -----------------------------------------------------------------------
ASGI_APPLICATION = "django_chat.asgi.application"

REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL:
    # Production: use Redis as the channel layer backend
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL]},
        }
    }
else:
    # Local dev fallback — no Redis required
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }

# -----------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# -----------------------------------------------------------------------
# URLs and templates
# -----------------------------------------------------------------------
ROOT_URLCONF = "django_chat.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "django_chat.wsgi.application"

# -----------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    DATABASES = {"default": dj_database_url.parse(DATABASE_URL)}
else:
    # SQLite is fine for local development
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME":   BASE_DIR / "db.sqlite3",
        }
    }

# -----------------------------------------------------------------------
# Password validation
# -----------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# -----------------------------------------------------------------------
# Internationalisation
# -----------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE     = "UTC"
USE_I18N      = True
USE_TZ        = True

# -----------------------------------------------------------------------
# Static files
# -----------------------------------------------------------------------
STATIC_URL  = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

# Using the default storage backend avoids the W004 warning on Render
STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"

# -----------------------------------------------------------------------
# Media files
# -----------------------------------------------------------------------
MEDIA_URL  = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# -----------------------------------------------------------------------
# Default primary key type
# -----------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# -----------------------------------------------------------------------
# Authentication (allauth)
# -----------------------------------------------------------------------
SITE_ID = 1

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]

LOGIN_URL           = "/accounts/login/"
LOGIN_REDIRECT_URL  = "/"
LOGOUT_REDIRECT_URL = "/accounts/login/"

ACCOUNT_SIGNUP_FIELDS      = ["email*", "password1*", "password2*"]  # * means required
ACCOUNT_LOGIN_METHODS      = {"email"}   # users log in with their email address
ACCOUNT_EMAIL_VERIFICATION = "none"     # set to 'mandatory' to require email confirmation

ACCOUNT_SIGNUP_TEMPLATE = "account/signup.html"
ACCOUNT_LOGIN_TEMPLATE  = "account/login.html"

# -----------------------------------------------------------------------
# Django REST Framework
# -----------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        # Session auth is kept so the Django admin and allauth pages still work
        "rest_framework.authentication.SessionAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
}

# -----------------------------------------------------------------------
# JWT settings (djangorestframework-simplejwt)
# -----------------------------------------------------------------------
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME":    timedelta(days=7),
    "REFRESH_TOKEN_LIFETIME":   timedelta(days=30),
    "ROTATE_REFRESH_TOKENS":    True,
    "BLACKLIST_AFTER_ROTATION": False,
    "AUTH_HEADER_TYPES":        ("Bearer",),
    "USER_ID_FIELD":            "id",
    "USER_ID_CLAIM":            "user_id",
}