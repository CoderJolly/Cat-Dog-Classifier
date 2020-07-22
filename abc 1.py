import numpy as np 
import sys
import pandas as pd 
import matplotlib.pyplot as py

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Activation

''' now we will be importing the images and also scaling them down '''
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

''' the above code will take the data automatically in the batches and class mode 
is set to binary because of the reason that because we have only 2 classes'''

''' now we will be making our own neural network'''

classifier = Sequential() # as we aew moving from left-to-right, hence sequential

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #input_shape (size, size, 3 dimensions(RGB)
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten()) # making features after  pooling in array form, vertical array

classifier.add(Dense(10))
classifier.add(Activation('sigmoid'))

classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()


classifier.fit(training_set, epochs=1)

classifier.save_weights()

# h = classifier.predict_classes(test_set)
# print(h)

                                            
