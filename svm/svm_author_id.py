#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
from sklearn.svm import SVC
import math


f = -.666666 * math.log(.666666,2)
s = .333333 * math.log(.333333,2)

print f
print s
ans = f-s
print ans
exit()

clf = SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print "The accuracy with C equal to %d is %s" % (10000, accuracy_score(labels_test, pred))

print type(pred)
#print clf.classificaiton_report(features_test)
#print(pred.count(1))
print sum(pred)

# prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())
# #########################################################


# #!/usr/bin/python

# #from udacityplots import *
# import warnings
# warnings.filterwarnings("ignore")

# import matplotlib 
# matplotlib.use('agg')

# import matplotlib.pyplot as plt
# import pylab as pl
# import numpy as np

# #import numpy as np
# #import matplotlib.pyplot as plt
# #plt.ioff()

# def prettyPicture(clf, X_test, y_test):
#     x_min = 0.0; x_max = 1.0
#     y_min = 0.0; y_max = 1.0

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, m_max]x[y_min, y_max].
#     h = .01  # step size in the mesh
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())

#     plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

#     # Plot also the test points
#     grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
#     bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
#     grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
#     bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

#     plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
#     plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
#     plt.legend()
#     plt.xlabel("bumpiness")
#     plt.ylabel("grade")

#     plt.savefig("test.png")
    
# import base64
# import json
# import subprocess

# def output_image(name, format, bytes):
#     image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
#     image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
#     data = {}
#     data['name'] = name
#     data['format'] = format
#     data['bytes'] = base64.encodestring(bytes)
#     print image_start+json.dumps(data)+image_end