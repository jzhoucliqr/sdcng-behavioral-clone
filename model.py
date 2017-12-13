import numpy as np
import csv
import cv2
import os.path
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                steering_center = float(line[3])
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_center = mpimg.imread('./data/IMG/' + line[0].split('/')[-1])
                image_left  = mpimg.imread('./data/IMG/' + line[1].split('/')[-1])
                image_right = mpimg.imread('./data/IMG/' + line[2].split('/')[-1])

                images.extend([image_center, image_left, image_right])
                measurements.extend([steering_center, steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(measurements)
            image_flipped = np.fliplr(X_train)
            measurement_flipped = 0.0 - y_train
            X_train = np.append(X_train, image_flipped, axis=0)
            y_train = np.append(y_train, measurement_flipped, axis=0)

            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')

