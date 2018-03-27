from django.conf.urls import url
from upload_form import views

urlpatterns = [
    url(r'^$', views.form, name = 'process'),
    url(r'^complete/', views.complete, name = 'complete'),
]
