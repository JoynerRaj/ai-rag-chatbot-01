from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat"),
    path("documents/", views.document_list, name="documents"),
    path("upload/", views.upload_page, name="upload"),
    path("delete/<int:id>/", views.delete_document, name="delete"),
    path("edit/<int:id>/", views.edit_document, name="edit"),
    path("clear-history/", views.clear_history, name="clear_history"),
    path("new-chat/", views.create_chat, name="new_chat"),
    path("delete-chat/<int:chat_id>/", views.delete_chat, name="delete_chat"),
]