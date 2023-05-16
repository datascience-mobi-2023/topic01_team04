import numpy as np
import pandas as pd 
"""
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import BatchNormalization

#add data

model = Sequential()

model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1))) #same passing adds zeros around the image
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1)) # default stride is 2
model.add(BatchNormalization())

model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1)) # default stride is 2
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax')) #softmax gives probability of each class

Notizen
-max pooling mit 2x2 ermöglicht Verdopplung der Filter, ohne die Rechenleistung zu erhöhen"""


#add other stuff


def testallzeros(pixel):
    """Tests if any column is all zeros

    Args:
        pixel (numpy array): array with each picture as a column
    """
    print(pixel.shape)
    k = 0
    for i in range(pixel.shape[1]):
        if np.all(pixel[:,i]==0):
            print("Column" + i + "is all zeros.")
        elif np.all(pixel[:,i]==0) == False:
            k += 1
    if k == pixel.shape[1]:
        print("Congratulations, no column is all zeros.")