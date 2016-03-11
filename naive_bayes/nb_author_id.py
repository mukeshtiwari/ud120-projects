#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
tf = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - tf, 3), "s"
ts = time()
predicted = clf.predict(features_test)
print "testing time:", round (time() - ts, 3), "s"
from sklearn.metrics import accuracy_score
print "accuracy of score", accuracy_score(labels_test, predicted)
#########################################################


