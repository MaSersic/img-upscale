from django.http import HttpResponse
from django.shortcuts import render
import os

def upload_view(request):
    if request.method == 'POST':
        photo = request.FILES['photo']
        filepath = os.path.join('uploads/', photo.name)
        with open(filepath, 'wb+') as destination:
            for chunk in photo.chunks():
                destination.write(chunk)
        return HttpResponse('Photo uploaded successfully!')
    else:
        return render(request, 'main.html')