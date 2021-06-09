# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:37:11 2021

@author: Ruijie Song
"""

import random
import string
import numpy as np

# generate an array of 10 random strings of length 5
def random_name():
    string1 = ''
    array1 = []
    for j in range(0,10):
        for i in range(0,5):
            string1 = string1 + random.choice(string.ascii_lowercase)
        array1 = np.append(array1,string1)
        string1 = ''
    return array1

def random_num():
    array2 = []
    for i in range(0,10):
        temp_num = random.randint(0,1000)
        array2 = np.append(array2,temp_num)
    return array2

class People:
    def __init__(self, nformat): # first_name=random_name(),middle_name = random_name(),last_name = random_name()):
        random.seed(0)
        self.first_name = random_name()
        self.middle_name = random_name()
        self.last_name = random_name()
        self.nformat = nformat
        self.index = -1
    def __call__(self):
        name = ''
        for i in range(0,10):
            name = (name + self.last_name[i]+'\n')
        return name 
    def __iter__(self):
        return self
        # return Peopleiterator(self)
    def __next__(self):
        self.index +=1
        if (self.nformat == 'first_name_first'):
            return self.first_name[self.index]+' '+self.middle_name[self.index]+' '+self.last_name[self.index]
        elif (self.nformat == 'last_name_first'):
            return self.last_name[self.index]+' '+self.first_name[self.index]+' '+self.middle_name[self.index]
        elif (self.nformat == 'last_name_with_comma_first'):
            return self.last_name[self.index]+' , '+self.first_name[self.index]+' '+self.middle_name[self.index]
        # print(self.name)
        else:
            raise StopIteration
    next = __next__
    
class PeopleWithMoney(People):
    def __init__(self, nformat='first_name_first'):
        People.__init__(self,nformat)
        random.seed(0)
        self.wealth = random_num()
    def __call__(self):
        sorted_index = np.argsort(self.wealth)
        name_list = ''
        if (self.nformat == 'first_name_first'):
            for i in sorted_index:
                name_list = name_list+'\n'+self.first_name[i]+' '+self.middle_name[i]+' '+self.last_name[i]+' '+str(self.wealth[i])
            return name_list
        elif (self.nformat == 'last_name_first'):
            for i in range(len(sorted_index)-1):
                name_list = name_list+'\n'+self.last_name[i]+' '+self.first_name[i]+' '+self.middle_name[i]+' '+str(self.wealth[i])
            return name_list
        elif (self.nformat == 'last_name_with_comma_first'):
            for i in range(len(sorted_index)-1):
                name_list = name_list+'\n'+self.last_name[i]+' , '+self.first_name[i]+' '+self.middle_name[i]+' '+str(self.wealth[i])
            return name_list
    def __next__(self):
        return (People.__next__(self)+' '+str(self.wealth[self.index]))
        # return temp_string
    next = __next__
        
'''
        self.index +=1
        if (self.nformat == 'first_name_first'):
            return self.first_name[self.index]+' '+self.middle_name[self.index]+' '+self.last_name[self.index]+' '+string(self.wealth[self.index])
        elif (self.nformat == 'last_name_first'):
            return self.last_name[self.index]+' '+self.first_name[self.index]+' '+self.middle_name[self.index]+' '+string(self.wealth[self.index])
        elif (self.nformat == 'last_name_with_comma_first'):
            return self.last_name[self.index]+' , '+self.first_name[self.index]+' '+self.middle_name[self.index]+' '+string(self.wealth[self.index])
        # print(self.name)
        else:
            raise StopIteration
'''
    # next = __next__
        
    
'''

class Peopleiterator:
    def __int__(self,p_obj):
        self.first_name = p_obj.first_name
        self.middle_name = p_obj.middle_name
        self.last_name = p_obj.last_name
        self.nformat = p_obj.nformat
        self.index = -1
        # self.name = ''
    def __iter__(self):
        return self
    def __next__(self):
        self.index +=1
        if (self.nformat == 'first_name_first'):
            return self.first_name[self.index]+' '+self.middle_name[self.index]+' '+self.last_name[self.index]
        elif (self.nformat == 'last_name_first'):
            return self.last_name[self.index]+' '+self.first_name[self.index]+' '+self.middle_name[self.index]
        elif (self.nformat == 'last_name_with_comma_first'):
            return self.last_name[self.index]+', '+self.first_name[self.index]+' '+self.middle_name[self.index]
        # print(self.name)
        else:
            raise StopIteration
    # next = __next__
'''
        

# run

X = People('first_name_first')
iters1 = iter(X)
for i in range(0,10):
    print(iters1.next())
    
print()

Y = People('last_name_first')
iters2 = iter(Y)
for i in range(0,10):
    print(iters2.next())

print()

Z = People('last_name_with_comma_first')
iters3 = iter(Z)
for i in range(0,10):
    print(iters3.next())
    
print()

print(X())

A = PeopleWithMoney()
iters4 = iter(A)
for i in range(0,10):
    print(iters4.next()) 
    
print(A())