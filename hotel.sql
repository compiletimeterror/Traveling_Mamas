create table hotel(
    hotel_id varchar(100) primary key,
    hotel_name varchar(100),
    description varchar(100), 
    location varchar(100),
    reviews int
);

create table tour(
    tour_id varchar(100) primary key,
    tour_name varchar(100),
    fare integer, 
    place varchar(100),
    location varchar(100),
    reviews integer,
    description varchar(100)
);

create table flight(
    flight_number varchar(100) primary key,
    origin varchar(100),
    departure_time varchar(100), 
    place varchar(100),
    arrival_time varchar(100),
    destination varchar(100)
);

create table book(
    user_id varchar(100),
    flight_number varchar(100),
    hotel_id varchar(100),
    tour_id varchar(100),
    primary key (user_id, flight_number, hotel_id, tour_id),
    foreign key (user_id) references user(user_id),
    foreign key (flight_number) references flight(flight_number),
    foreign key (hotel_id) references hotel(hotel_id),
    foreign key (tour_id) references tour(tour_id)
);



create table class (
    flight_number varchar(100), 
    type varchar(100),
    primary key(flight_number, type)
);


create table belong_to (
    airline_id varchar(100),
    flight_number varchar(100),
    primary key (airline_id, flight_number),
    foreign key (airline_id) references airline(airline_id),
    foreign key (flight_number) references flight(flight_number)
);

create table consists (
    hotel_id varchar(100),
    room_id varchar(100),
    primary key(hotel_id, room_id),
    foreign key (hotel_id) references hotel(hotel_id),
    foreign key (room_id) references rooms(room_id)
);

create table room_type (
    room_number varchar(100),
    room_id varchar(100),
    price integer,
    primary key(room_id, room_number)
);

create table rooms(
    room_id varchar(100) primary key,
    accomadation varchar(100)
);

create table has_flight(
    flight_number varchar(100),
    type varchar(100),
    primary key(flight_number, type)
);

alter table has_flight add constraint fk_has_flight_flight foreign key (flight_number) references flight(flight_number);
alter table has_flight add constraint fk_has_flight_flight foreign key (flight_number, type) references class(flight_number, type);

create table has_room(
    room_id varchar(100),
    room_number varchar(100),
    primary key(room_number, room_id)
);


alter table has_room add constraint fk_has_room_rooms foreign key (room_id) references rooms(room_id);
alter table has_room add constraint fk_has_room_room_type foreign key (room_id, room_number) references room_type(room_id, room_number);

create table airline(
    airline_id varchar(100) primary key,
    name varchar(100),
    head_quarter_city varchar(100)
);

create table contact(
    user_id varchar(100) primary key,
    contact varchar(100)
);

create table user(
    user_id varchar(100),
    email varchar(100),
    first_name varchar(100),
    middle_name varchar(100),
    last_name varchar(100),
    password varchar(100),
    dob varchar(100),
    primary key (user_id, email)
);



insert into travellingmamas_tour(tour_name, fare, place, location, reviews, description) 
    values 
        (
            'Family Tour in Srinagar, Kashmir',
            5000,
            'Kashmir, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Shimla, Himachal Pradesh',
            3500,
            'Himachal Pradesh, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Nainital, Uttrakhand',
            2500,
            'Uttrakhand, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Darjeeling, West Bengal',
            1500,
            'West Bengal, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Coorg, Karnataka',
            2500,
            'Karnataka, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Ooty, Tamil Nadu',
            2500,
            'Ooty, Tamil Nadu, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Alleppey, Kerala',
            2100,
            'Kerela, India',
            'India',
            5,
            'Switzerland of India'    
        ),

        (
            'Family Tour in Dharamshala, Himachal Pradesh',
            4200,
            'Himachal Pradesh, India',
            'India',
            5,
            'Switzerland of India'    
        );
    
insert into travellingmamas_room_type(room_number, price) 
    values
         (1001, 600), (1002, 1200),  
         (1003, 1999), (1004, 2499), 
         (1005, 3499), (1005, 4999); 

   
insert into travellingmamas_rooms(accomadation) 
    values
         (4), (2),  
         (4), (2), 
         (2), (3); 

insert into travellingmamas_hotel(hotel_name, description, location, reviews)
    values 
        ('Hotel Woodpark', 'very good place', 'Srinagar, Kashmir', 5),
        ('Hotel Novacany', 'lovely place', 'Srinagar, Kashmir', 5),
        ('Hotel Madaland', 'best in the city', 'Srinagar, Kashmir', 5),
        ('Hotel Sanskriti', 'despacito wow', 'Srinagar, Kashmir', 5),
        ('Hotel Sriram resort', 'shape on you', 'Srinagar, Kashmir', 5),
        ('Hotel Rishi Mangalam', 'senorita Mi Gente', 'Srinagar, Kashmir', 5), 
        ('Rosewood International Hotel', 'Hym for the weekend', 'Srinagar, Kashmir', 5),
        ('Hotel Basant Residency', 'blank space', 'Srinagar, Kashmir', 5),

        ('Hotel Sharma Brothers', 'Mala Mia', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Havetual', 'Shakira ', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Sankriti', 'Sube mela radio', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Ganga', 'Corazon', 'Shimla, Himachal Pradesh', 5),
        ('Hotel American Resort', 'Party Animals', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Mountain Crest', 'Forever', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Snow Queen', 'Vente Pa Ca', 'Shimla, Himachal Pradesh', 5),
        ('Hotel Radisan', 'Luis Fonsi Demi Lovato', 'Shimla, Himachal Pradesh', 5),

        ('Hotel Pythava Resort', 'Chandni Chowk', 'Nainital, Uttrakhand', 5),
        ('Hotel Wild Ville', 'The Chainsmokers', 'Nainital, Uttrakhand', 5),
        ('Hotel Sansar Apna', 'Closer by Halsey', 'Nainital, Uttrakhand', 5),
        ('Hotel Appleview', 'Girls like you', 'Nainital, Uttrakhand', 5),
        ('Hotel Polish Resort', 'Maroon five', 'Nainital, Uttrakhand', 5),
        ('Hotel Hirest', 'Khantroon ke khiladi', 'Nainital, Uttrakhand', 5),
        ('Hotel King Resort', 'All about tha base', 'Nainital, Uttrakhand', 5),
        ('Hotel Nishu', 'Waka Waka', 'Nainital, Uttrakhand', 5),

        ('Hotel Rista Resort', 'We Dont talk anymore', 'Darjeeling, West Bengal', 5),
        ('Hotel Nature Ville', 'Rockaby', 'Darjeeling, West Bengal', 5),
        ('Hotel Sansar', 'Work from home', 'Darjeeling, West Bengal', 5),
        ('Hotel Pineview', 'Perfect', 'Darjeeling, West Bengal', 5),
        ('Hotel British Resort', 'Justin Bieber', 'Darjeeling, West Bengal', 5),
        ('Hotel Hill Crest', 'Cardi B', 'Darjeeling, West Bengal', 5),
        ('Hotel Snow King Resort', 'Echame La Culpa', 'Darjeeling, West Bengal', 5),
        ('Hotel Nishat', 'Love the way you lie', 'Darjeeling, West Bengal', 5),

        ('Hotel Raj Resort', 'Sorry', 'Coorg, Karnataka', 5),
        ('Hotel Abbydhama Esate Stay', 'Roar Ketty Perry', 'Coorg, Karnataka', 5),
        ('Hotel Woodstack Villa', 'Ed sheeran Thinking Loud', 'Coorg, Karnataka', 5),
        ('Hotel Coorg Heights', 'Taylor Swift Shake it off', 'Coorg, Karnataka', 5),
        ('Hotel Indrastha', 'Major Lazer', 'Coorg, Karnataka', 5),
        ('Hotel IBNI', 'Lean On', 'Coorg, Karnataka', 5),
        ('Hotel OYO Hilldale Resort', 'Dark House', 'Coorg, Karnataka', 5),
        ('Hotel Vashati', 'Charlie Puth', 'Coorg, Karnataka', 5),

        ('Hotel Fabkhems', 'What do you mean', 'Ooty, Tamil Nadu', 5),
        ('Hotel Treebo Trend Whispering', 'New Rules', 'Ooty, Tamil Nadu', 5),
        ('Hotel Novacany', 'Criminal Nti Natasha', 'Ooty, Tamil Nadu', 5),
        ('Hotel Sterling', 'Crazy Frog', 'Ooty, Tamil Nadu', 5),
        ('Hotel LakeView', 'Calvin Harris Rihanna', 'Ooty, Tamil Nadu', 5),
        ('Hotel Sunaark Grand', 'Sean Paul', 'Ooty, Tamil Nadu', 5),
        ('Hotel Tuilips', 'Annie Marrie', 'Ooty, Tamil Nadu', 5),
        ('Hotel Berry Hills', 'Rishi Sharma Ed Sheeran', 'Ooty, Tamil Nadu', 5),

        ('Hotel WhiteShore Beach', 'Har dil do pyaar kaerga', 'Alleppey, Kerala', 5),
        ('Hotel Ramada', 'Umm Umm Ummmmmummm', 'Alleppey, Kerala', 5),
        ('Hotel Baywatch Beach Resort', 'Sapna bante badta he', 'Alleppey, Kerala', 5),
        ('Hotel Paagada Resort', 'Zoor koi chalta he', 'Alleppey, Kerala', 5),
        ('Hotel Royal Park', 'Use Hello baby', 'Alleppey, Kerala', 5),
        ('Hotel Zostel', 'Hello world', 'Alleppey, Kerala', 5),
        ('Hotel Bamboon Lagoon', 'Hello there', 'Alleppey, Kerala', 5),
        ('Hotel Palace Resort', 'Hi there', 'Alleppey, Kerala', 5),

        ('Hotel Adhivaha', 'Machine Learning', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Vaikunth', 'Deep Learning', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel InClover', 'Copmuter Vision', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Pink Resorts', 'Natural Language Processing', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Dauladhar', 'Reinforcement Learning', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Zostel', 'Maa shah allah', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Horizon Villa', 'Inshah allah ', 'Dharmashala, Himachal Pradesh', 5),
        ('Hotel Imperial Resorts', 'Khuda Hafeez', 'Dharmashala, Himachal Pradesh', 5);


insert into travellingmamas_consists(hotel_id_id, room_id_id)
    values 
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 2),
        (5, 1), 
        (6, 3),
        (7, 5),
        (8, 6),
        (9, 5),
        (10, 6),
        (11, 3);
        


























INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','SRINAGAR','13:00','13:55','2.5hr', '5900', '10/24/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','SHIMLA','19:20','20:05','1.5hrs','6000', '10/25/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','SHIMLA','05:50','06:35','1.5hrs','6100', '10/26/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','NAINITAL','10:20','10:55','2.5hrs','6200', '10/27/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','NAINITAL','18:10','18:55','2.1hrs','6300', '10/28/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','DARJEELING','22:00','22:50','2.8hrs','6400', '10/29/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','DARJEELING','02:15','02:55','2.9hrs','6500', '10/30/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','COORG','18:50','19:40','2.5hrs','6600', '10/31/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','COORG','23:00','13:45','2.3hrs','6700', '11/01/2019');

INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','OOTY','18:50','19:40','2.8hrs','6800', '11/02/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','OOTY','23:00','13:45','2.3hrs','6900', '11/03/2019');

INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','ALLEPPEY','18:50','19:40','1.5hrs','7000', '11/04/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','ALLEPPEY','23:00','13:45','2.3hrs','7100', '11/05/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('MANGALORE','DHARAMSHALA','18:50','19:40','2 hrs','7200', '11/06/2019');


INSERT INTO travellingmamas_flight_prime(source, destination, arrival_time, departure_time, duration, price, date)
VALUES('BANGALORE','DHARAMSHALA','23:00','13:45','2.3hrs','7300', '11/07/2019');


insert into travellingmamas_has_room(room_id_id, room_number_id) values
(1, 1001), (1, 1002), (1, 1003), (1, 1004), (1, 1005), (1, 1006),
(2, 1001), (2, 1002), (2, 1003), (2, 1004), (2, 1005), (2, 1006),
(3, 1001), (3, 1002), (3, 1003), (3, 1004), (3, 1005), (3, 1006),
(4, 1001), (4, 1002), (4, 1003), (4, 1004), (4, 1005), (4, 1006),
(5, 1001), (5, 1002), (5, 1003), (5, 1004), (5, 1005), (5, 1006),
(6, 1001), (6, 1002), (6, 1003), (6, 1004), (6, 1005), (6, 1006)




insert into travellingmamas_airline(name, head_quarter_city) values
('air india', 'delhi'),
('indigo', 'gurgaon'),
('spicejet', 'gurgoan'),
('vistara', 'gurgoan'),
('goair', 'mumbai'),
('air india', 'kochi'),
('air asia', 'bengaluru')


insert into travellingmamas_contact(contact) values
('1234567890'),
('1122334455'),
('9876543212')
