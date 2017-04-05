#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Number of persons of interest in the data
persons_of_interest = {k:v for k,v in enron_data.iteritems() if v['poi']}
print len(persons_of_interest)
# print persons_of_interest

# Misc data
print enron_data['PRENTICE JAMES']['total_stock_value']
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

# How much the CEO, CFO, and Board Chairman get paid in the end, numbers listed respectively
print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']

# People with a known salary
people_with_salary = {k:v for k,v in enron_data.iteritems() if v['salary']!='NaN'}
print len(people_with_salary)

# People with a known email address
people_with_email = {k:v for k,v in enron_data.iteritems() if v['email_address']!='NaN'}
print len(people_with_email)

# People with NaN for total total_payments
people_without_total_payments = {k:v for k,v in enron_data.iteritems() if v['total_payments']=='NaN'}
print len(people_without_total_payments)

# POIs with NaN for total total_payments
pois_without_total_payments = {k:v for k,v in enron_data.iteritems() if v['total_payments']=='NaN' and v['poi']}
print len(pois_without_total_payments)