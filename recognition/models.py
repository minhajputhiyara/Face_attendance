from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserfaceData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='face_data')
    face_encoding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Face data for {self.user.username}"

class Present(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recognition_present_set')
    date = models.DateField()
    present = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.date} - {'Present' if self.present else 'Absent'}"

class Time(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recognition_time_set')
    date = models.DateField()
    time = models.DateTimeField()
    out = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.date} - {'Out' if self.out else 'In'} at {self.time.strftime('%H:%M:%S')}"
