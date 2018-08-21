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

#print enron_data.values()[1]
#enron_data.get('poi')[1]

#print (enron_data)

#print enron_data[person_name]["poi"]==1


n = 0
for k in enron_data:
  n += 1 if (enron_data[k]["total_payments"] =='NaN') & (enron_data[k]['poi']==True) else 0

print n

l = 0
for k in enron_data:
  l += 1 if (enron_data[k]['poi']==True) else 0
print l


print float(n)/float(l)


#enron_data["LASTNAME FIRSTNAME"]["feature_name"]
#enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]
#print enron_data.keys()
#print enron_data["COLWELL WESLEY"]
