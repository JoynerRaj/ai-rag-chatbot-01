from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken


def _get_tokens(user):
    """Return a fresh access + refresh token pair for the given user."""
    refresh = RefreshToken.for_user(user)
    return {
        "refresh": str(refresh),
        "access":  str(refresh.access_token),
    }


@api_view(["POST"])
@permission_classes([AllowAny])
def register(request):
    """
    POST /api/auth/register/
    Body: { "username": "...", "email": "...", "password": "..." }
    Creates a new account and returns JWT tokens immediately.
    """
    username = request.data.get("username", "").strip()
    email    = request.data.get("email", "").strip().lower()
    password = request.data.get("password", "")

    if not username or not email or not password:
        return Response(
            {"error": "username, email, and password are all required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if User.objects.filter(username=username).exists():
        return Response(
            {"error": "That username is already taken."},
            status=status.HTTP_409_CONFLICT,
        )

    if User.objects.filter(email=email).exists():
        return Response(
            {"error": "An account with that email already exists."},
            status=status.HTTP_409_CONFLICT,
        )

    user = User.objects.create_user(username=username, email=email, password=password)
    return Response(
        {"message": "Account created.", "user_id": user.id, **_get_tokens(user)},
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
@permission_classes([AllowAny])
def login(request):
    """
    POST /api/auth/login/
    Body: { "username": "...", "password": "..." }
    Returns JWT tokens on success.
    """
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")

    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response(
            {"error": "Invalid credentials. Please check your username and password."},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    return Response({"user_id": user.id, "username": user.username, **_get_tokens(user)})


@api_view(["GET"])
def me(request):
    """
    GET /api/auth/me/
    Returns basic profile info for the currently authenticated user.
    Requires a valid Bearer token in the Authorization header.
    """
    user = request.user
    return Response({
        "user_id":  user.id,
        "username": user.username,
        "email":    user.email,
    })
