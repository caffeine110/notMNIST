#!/usr/bin/env pyth on3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar
        : caffeine110
    
Aim     : Convulational Neural Network to classify Alphabets from A t0 J;
        : Udacity mini project

"""

#Part - 1 : Building the model

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Phase 1 : Initialising the CNN
model_cnn_classifier = Sequential()

#Convolutional 
model_cnn_classifier.add(Conv2D(32,3,3,  input_shape = (28,28,3), activation= 'relu'))

# Phase - 2 : Pooling
model_cnn_classifier.add(MaxPooling2D(pool_size = (2,2)))


# Adding a second convolutional layer
model_cnn_classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
model_cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))



#Phase - 3 : Flattening
model_cnn_classifier.add(Flatten())

#Phase - 4 : 
#Full connection
model_cnn_classifier.add(Dense(output_dim = 128, activation= 'relu'))
#Phase - (middle): Output layer 
#model_cnn_classifier.add(Dense(   ))
model_cnn_classifier.add(Dense(output_dim = 10, activation= 'softmax'))




#Phase - 5 :  compiling the CNN
model_cnn_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




#Part 2: Fitting CNN to the image
#Importing 
from keras.preprocessing.image import ImageDataGenerator

# Train Datagen
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        horizontal_flip=True )

#Test Daragen
test_datagen = ImageDataGenerator(rescale=1./255)


#Training Set
training_set= train_datagen.flow_from_directory(
        'notMNIST_large',
        target_size = (28, 28),
        batch_size = 32,
        class_mode='binary')

#test Set
test_set = test_datagen.flow_from_directory(
        'notMNIST_small',
        target_size = (28,28),
        batch_size = 32,
        class_mode='binary')


#Fitting data
#Training Our model by fitting the data
model_cnn_classifier.fit_generator(training_set,
                         steps_per_epoch = 529119,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 18726)


# testing classifier
test = model_cnn_classifier(test_set)

#Phase - 5 : Prediction on New data
#check model 
predict_set = test_datagen.flow_from_directory(
        'single_prediction',
        target_size = (28,28),
        batch_size = 1,
        class_mode='binary')
#result is stored in predictions

predictions = model_cnn_classifier.predict(predict_set)
