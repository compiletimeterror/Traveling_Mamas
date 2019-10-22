from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def blog(request):
    return render(request, 'blog.html')

def contact(request):
    return render(request, 'contact.html')

def hotel_room(request):
    return render(request, 'hotel-room.html')

def hotel(request):
    return render(request, 'hotels.html')

def service(request):
    return render(request, 'services.html')

# def tour_place(request):
#     return render(request, 'tour_place.html')

def tour(request):
    return render(request, 'tours.html')

def srinagar(request):
    return render(request, 'srinagar.html')

def shimla(request):
    return render(request, 'shimla.html')

def nainital(request):
    return render(request, 'nainital.html')

def darjeeling(request):
    return render(request, 'darjeeling.html')

def coorg(request):
    return render(request, 'coorg.html')


def ooty(request):
    return render(request, 'ooty.html')

def alleppey(request):
    return render(request, 'alleppey.html')


def dharmashala(request):
    return render(request, 'dharmashala.html')

def kashmir(request):
    return render(request, 'kashmir.html')

def shimla_main(request):
    return render(request, 'shimla_main.html')

def nainital_main(request):
    return render(request, 'nainital_main.html')

def darjeeling_main(request):
    return render(request, 'darjeeling_main.html')

def coorg_main(request):
    return render(request, 'coorg_main.html')

def ooty_main(request):
    return render(request, 'ooty_main.html')

def alleppey_main(request):
    return render(request, 'alleppey_main.html')

def dharamshala_main(request):
    return render(request, 'dharamshala_main.html')

def three(request):
    return render(request, 'three.html')

def two(request):
    return render(request, 'two.html')


def hotel_booking(request):
    return render(request, 'hotel_booking.html')