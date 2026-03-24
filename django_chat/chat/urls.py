from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_page, name='chat'),

    path('upload/', views.upload_page, name='upload'),
    path('documents/', views.document_list, name='documents'),
    path('edit/<int:id>/', views.edit_document, name='edit'),
    path('delete/<int:id>/', views.delete_document, name='delete'),
]