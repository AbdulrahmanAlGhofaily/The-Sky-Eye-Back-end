from django.urls import path
from . import views

urlpatterns = [
    path("coordinates/", views.postCoordinates),
    path('image/', views.postImage),
    path('results/', views.fetchResults)
]