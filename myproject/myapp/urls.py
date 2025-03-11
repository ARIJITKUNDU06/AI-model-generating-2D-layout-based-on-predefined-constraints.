from django.urls import path
from .views import home, user_list, predict, get_layout_image

urlpatterns = [
    path("", home, name="home"),
    path("users/", user_list, name="user_list"),
    path("predict/", predict, name="predict"),
    path("get_layout_image/", get_layout_image, name="get_layout_image"),
]
