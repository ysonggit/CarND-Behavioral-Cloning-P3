import csv
import cv2
import numpy as np
import os
import glob
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# refactor codes used in the video tutorial
lines = []
i = 0
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if i > 0:
            lines.append(line)
        i+=1
images = []
measurements = []
offsets = [0, -0.2, 0.2]
for line in lines:
    for img_idx in range(3):
        source_path = line[img_idx]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' +filename
        orig_image = cv2.imread(current_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        # center if offset = 0, left if offset < 0, right if offset > 0
        measurements.append(measurement+offsets[img_idx])

def createModel():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

X_train = np.array(images)
y_train = np.array(measurements)

print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)

model = createModel()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('sequential_model.h5')
