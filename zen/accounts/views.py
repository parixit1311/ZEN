from django.shortcuts import redirect, render
from django.contrib.auth import logout, login, authenticate
from django.contrib import messages, auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import authentication
import os

global u_id

def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = authentication.objects.filter(email__iexact=email)
        print(user)
        for u in user:
            if email == u.email:
                if password == u.password:
                    messages.success(request, 'you are logged in')
                    request.session['email']=email
                    u_id=u.id

                    return redirect('home')
                else:
                    messages.warning(request, 'wrong password')
                    return redirect('login')
            else:
                messages.warning(request, 'Invalid Credentials')
                return redirect('login')
    return render(request, 'accounts/login.html')


def register(request):
    if request.method == 'POST':
            name = request.POST.get('name', False)
            email = request.POST.get('email', False)
            profession = request.POST.get('profession', False)
            password = request.POST.get('password', False)
            confirm_password = request.POST.get('confirm_password', False)
            
            
            users = authentication(name = name, email = email, profession=profession, password=password, confirm_password=confirm_password)

            users.save()

            messages.success(request, 'Account created successfully')
            return redirect('login')

            
    return render(request, 'accounts/register.html')     
  

def logout_user(request):
    logout(request)
    return redirect('login')