#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:27:38 2018

@author: gaurav
"""



"""
### data to imported

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

"""



#importing required libraries
import sys
from time import time

#Set the working directory
sys.path.append("../tools/")
from data_preprocess import train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels



#import GaussianNB
from sklearn.linear_model import LogisticRegression


#initialising classifier
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()


#fitting data to the classifer
logisticRegr.fit(train_dataset, train_labels)


## Phase - 4  Predicting phase
# Returns a NumPy Array
# Predict for One image
logisticRegr.predict(test_dataset[0].reshape(1,-1))

#### Predict for Multiple images at Once
logisticRegr.predict(test_dataset[0:10])

# Make predictions on entire test data
predictions = logisticRegr.predict(test_dataset)



### Phase - 5
#printing the accuracy
score = logisticRegr.score(test_dataset, test_labels)
print(score)