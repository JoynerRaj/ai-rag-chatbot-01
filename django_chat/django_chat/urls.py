from django.contrib import admin
from django.urls import path
from chat.views import chat_page

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", chat_page),
]