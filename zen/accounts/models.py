from django.db import models
from datetime import datetime

# Create your models here.

class authentication(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    profession = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    confirm_password = models.CharField(max_length=255, blank=True)
    created_date = models.DateTimeField(blank=True, default=datetime.now)
     

    def __str(self):
        return self.first_name
