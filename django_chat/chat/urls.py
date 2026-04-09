from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health_check, name="health_check"),
    path("", views.chat_page, name="chat"),
    path("documents/", views.document_list, name="documents"),
    path("upload/", views.upload_page, name="upload"),
    path("delete/<int:id>/", views.delete_document, name="delete"),
    path("edit/<int:id>/", views.edit_document, name="edit"),
    path("clear-history/", views.clear_history, name="clear_history"),
    path("new-chat/", views.create_chat, name="new_chat"),
    path("new-chat-ajax/", views.create_chat_ajax, name="new_chat_ajax"),
    path("delete-chat/<int:chat_id>/", views.delete_chat, name="delete_chat"),
    path("cache/", views.cache_page, name="cache"),
    path("cache/clear/", views.clear_cache, name="clear_cache"),
]