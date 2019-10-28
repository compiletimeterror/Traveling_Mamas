from django.contrib import admin
from .models import hotel, tour, user_hotel_details, flight, room_type, rooms, user, consists, contact, airline, hotel, hotel_class, has_room, has_flight, belong_to, book
# Register your models here.
admin.site.register(hotel)
admin.site.register(tour)
admin.site.register(user_hotel_details)
admin.site.register(flight)
admin.site.register(room_type)
admin.site.register(rooms)
admin.site.register(user)
admin.site.register(contact)
admin.site.register(consists)
admin.site.register(airline)
admin.site.register(hotel_class)
admin.site.register(has_flight)
admin.site.register(has_room)
admin.site.register(book)
admin.site.register(belong_to)


