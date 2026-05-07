from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import TokenRefreshView
from chat import auth_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),  # login, signup, google OAuth

    # JWT auth endpoints
    path("api/auth/register/", auth_views.register,   name="jwt_register"),
    path("api/auth/login/",    auth_views.login,      name="jwt_login"),
    path("api/auth/refresh/",  TokenRefreshView.as_view(), name="jwt_refresh"),
    path("api/auth/me/",       auth_views.me,         name="jwt_me"),

    path("", include("chat.urls")),
]