#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:18:44 2019

@author: ioannismilas
"""
from datetime import date
 
class User:
    pass

user1 = User() # object
user1.first_name = "Dave"  # fields
user1.last_name = "Bowman"

print(user1.first_name, user1.last_name)

user2 = User()
user2.first_name = "Frank"  # fields
user2.last_name = "Poole"

print(user2.first_name, user2.last_name)


user1.age = 37
print(user1.age)

## add methods (functions inside classes)

class User:
    """ store name and birthday"""
    def __init__(self, full_name, birthday):
        self.name = full_name
        self.birthday = birthday #YYYYMMDD
        
        name_pieces = full_name.split(" ")
        self.first_name = name_pieces[0]
        self.last_name = name_pieces[-1]
    def age(self):
        """ Return the age in years"""
        today = date.today()
        yyyy = int(self.birthday[0:4])
        mm = int(self.birthday[4:6])
        dd = int(self.birthday[6:8])
        dob = datetime.date(yyyy, mm, dd)
        age_in_days = (today - dob).days
        age_in_years = age_in_days / 365
        return int(age_in_years)
    
user = User("Dave Bowman", "19811006")
print(user.name)
print(user.first_name)
print(user.last_name)
print(user.birthday)        
print(user.age())
help(User)


