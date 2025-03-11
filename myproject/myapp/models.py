from django.db import models

class UserProfile(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    age = models.IntegerField()
    date_joined = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
