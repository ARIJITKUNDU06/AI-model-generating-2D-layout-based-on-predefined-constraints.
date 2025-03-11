from django.contrib import admin
from django.urls import path, include
from myapp.views import home  # Import the home view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('myapp.urls')),
    path('', home, name='home'),  
]
