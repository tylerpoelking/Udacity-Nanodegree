#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)

#print data_dict.values()
#print data_dict.keys()

#print max(data, key=lambda x:x['exercised_stock_options'])
#print min(data, key=lambda x:x['exercised_stock_options'])

l = ['hey']
for i in data_dict:
	ii =data_dict[i]
	l.append(ii["salary"])
	
l.sort()
print l
exit()
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


